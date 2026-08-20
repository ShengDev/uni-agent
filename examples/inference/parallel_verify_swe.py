"""Parallel gold-patch verification for SWE-bench.

Runs each dataset row's SWE-bench task in oracle mode (``run_oracle_solution=True``):
apply the gold patch in the sandbox, run the tests, and score. Every instance
should resolve -- it's the data-quality baseline you run before training. Results
are bucketed as resolved (ok) / wrong-answer (wa) / timeout-or-error (tle) and
streamed to a live progress bar.

Pass ``--task-config`` (same YAML as ``parallel_infer_api.py``) so run-level
``sandbox.image_map`` is merged before ``SandboxConfig`` is built.
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import ray
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.logging import LogContext, sample_logging
from uni_agent.tasks import TaskConfigResolver, get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 512))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
SANDBOX_PROVIDER = os.getenv("SANDBOX_PROVIDER", "modal")
RUNTIME_TIMEOUT = float(os.getenv("RUNTIME_TIMEOUT", 3600))


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    def __init__(self, log_dir: str | None):
        self.log_dir = log_dir

    async def run_single(self, task_config: dict) -> dict:
        async with self._semaphore:
            instance_id = task_config["metadata"]["instance_id"]
            log_id = f"verify-{uuid4().hex}"
            log_path = str(Path(self.log_dir).expanduser() / log_id / "task.log") if self.log_dir else None
            async with sample_logging.from_context(LogContext(log_id, log_path)):
                try:
                    result = await get_task(task_config).run()
                    info = result.extra_info or {}
                    resolved = bool(info.get("resolved", result.reward))
                    return {
                        "instance_id": instance_id,
                        "log_id": log_id,
                        "resolved": resolved,
                        "eval_completed": bool(info.get("eval_completed", True)),
                        "eval_execution_time": info.get("eval_execution_time"),
                    }
                except Exception as e:
                    logger.error(f"error verifying {instance_id}: {type(e).__name__}: {e}")
                    return {
                        "instance_id": instance_id,
                        "log_id": log_id,
                        "resolved": False,
                        "eval_completed": False,
                        "eval_execution_time": None,
                        "error": f"{type(e).__name__}: {e}",
                    }


def _prepare_task(sample: dict, resolver: TaskConfigResolver) -> dict:
    """Merge run-level Task Config (including ``sandbox.image_map``) onto the sample, then pin oracle eval."""
    sample_config = sample["extra_info"]["tools_kwargs"]["task"]
    resolved = resolver.resolve(sample_config)
    sandbox = dict(resolved.get("sandbox") or {})
    sandbox["provider"] = SANDBOX_PROVIDER
    sandbox["runtime_timeout"] = RUNTIME_TIMEOUT
    resolved["sandbox"] = sandbox
    resolved["run_oracle_solution"] = True
    return resolved


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--task-config",
        default=None,
        help="Run-level Task Config YAML (same shape as parallel_infer_api). "
        "Carries sandbox.image_map; omit to use the parquet sandbox fields as-is.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only verify the first N samples (smoke testing).")
    parser.add_argument(
        "--log-dir",
        default=os.getenv("UNI_AGENT_LOG_DIR", "/tmp/eval_gold_patch"),
        help="Root directory for per-sample logs; use an empty value to disable file logging.",
    )
    args = parser.parse_args()

    ray.init()

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return

    resolver = TaskConfigResolver.from_file(args.task_config) if args.task_config else TaskConfigResolver()
    try:
        tasks = [_prepare_task(sample, resolver) for sample in samples]
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("failed to resolve Task Config: %s", exc)
        return

    logger.info(f"loaded {len(tasks)} samples from {args.data_path}")
    logger.info(
        "provider=%s workers=%s concurrency=%s config=%s",
        SANDBOX_PROVIDER,
        args.num_workers,
        GLOBAL_CONCURRENCY,
        args.task_config or "parquet",
    )

    num_workers = min(args.num_workers, len(tasks))
    workers = [TestEvalActor.remote(args.log_dir) for _ in range(num_workers)]
    # One future per sample (round-robin across workers) so we can stream
    # per-sample progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % num_workers].run_single.remote(task) for i, task in enumerate(tasks)]
    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results: list = [None] * len(futures)
    ok = wa = tle = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="eval",
        unit="inst",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    ) as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            for d in done:
                res = ray.get(d)
                results[fut_to_idx[d]] = res
                if res.get("resolved"):
                    ok += 1
                elif res.get("eval_completed"):
                    wa += 1
                else:
                    tle += 1
                rate = ok / (pbar.n + 1) * 100
                pbar.set_postfix_str(f"resolved={ok} WA={wa} TLE={tle} | {rate:.0f}% pass")
                pbar.update(1)
    wall = time.time() - begin_time

    all_num = len(results)
    success_num = sum(1 for r in results if r.get("resolved"))
    fail_wa_num = sum(1 for r in results if not r.get("resolved") and r.get("eval_completed"))
    fail_tle_num = sum(1 for r in results if not r.get("resolved") and not r.get("eval_completed"))

    fail_wa_names = [r["instance_id"] for r in results if not r.get("resolved") and r.get("eval_completed")]
    fail_tle_names = [r["instance_id"] for r in results if not r.get("resolved") and not r.get("eval_completed")]

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
    pass_rate = success_num / all_num * 100 if all_num else 0.0

    summary = "\n".join(
        [
            "",
            _rule("eval summary"),
            f"  resolved    {success_num:>4}   ({pass_rate:.1f}%)",
            f"  wrong-ans   {fail_wa_num:>4}",
            f"  timeout/err {fail_tle_num:>4}",
            f"  total       {all_num:>4}",
            _rule(f"avg {avg_exec_time:.1f}s | wall {wall:.1f}s | n={len(exec_times)}"),
            "",
        ]
    )
    print(summary)

    logger.info(f"fail_wa instance names: {fail_wa_names}")
    logger.info(f"fail_tle instance names: {fail_tle_names}")

    errored = [(r["instance_id"], r["error"]) for r in results if r.get("error")]
    if errored:
        logger.warning(f"{len(errored)} samples raised exceptions (showing up to 10):")
        for name, err in errored[:10]:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
