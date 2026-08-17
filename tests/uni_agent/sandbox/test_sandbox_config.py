from __future__ import annotations

import pytest

from uni_agent.sandbox import ImageMap, SandboxConfig
from uni_agent.tasks import TaskConfigResolver

SWE_IMAGE = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
SWE_MAP = {
    "from": "swebench/**:latest",
    "to": "enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/**:v2",
}
SWE_MAPPED = (
    "enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.astropy_1776_astropy-12907:v2"
)


def test_image_map_untagged_as_latest():
    config = SandboxConfig(provider="vefaas", image=SWE_IMAGE, image_map=SWE_MAP)
    assert config.image == SWE_MAPPED


def test_image_map_explicit_latest_tag():
    config = SandboxConfig(
        provider="vefaas",
        image=f"{SWE_IMAGE}:latest",
        image_map=ImageMap.model_validate(SWE_MAP),
    )
    assert config.image == SWE_MAPPED


def test_image_map_without_tag_keeps_untagged_name():
    config = SandboxConfig(
        provider="docker",
        image="swebench/sweb.eval.x86_64.foo",
        image_map={"from": "swebench/**", "to": "my.registry.com/swe/**"},
    )
    assert config.image == "my.registry.com/swe/sweb.eval.x86_64.foo"


def test_image_map_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        SandboxConfig(
            provider="vefaas",
            image="swebench/sweb.eval.x86_64.foo",
            image_map={"from": "swerebench/**:latest", "to": "example.com/rebench/**:latest"},
        )


def test_image_map_list_picks_first_match():
    config = SandboxConfig(
        provider="vefaas",
        image="swerebench/sweb.eval.x86_64.foo",
        image_map=[
            SWE_MAP,
            {
                "from": "swerebench/**:latest",
                "to": "enterprise-public-cn-beijing.cr.volces.com/swe-rebench/**:latest",
            },
        ],
    )
    assert config.image == "enterprise-public-cn-beijing.cr.volces.com/swe-rebench/sweb.eval.x86_64.foo:latest"


def test_task_config_merge_applies_yaml_image_map_to_sample_image():
    resolved = TaskConfigResolver(
        {
            "swe_bench": {
                "name": "swe_bench",
                "sandbox": {
                    "provider": "vefaas",
                    "runtime_timeout": 7200,
                    "image_map": SWE_MAP,
                },
                "agent": {"name": "react"},
            }
        }
    ).resolve(
        {
            "name": "swe_bench",
            "sandbox": {"image": SWE_IMAGE},
            "metadata": {"instance_id": "astropy__astropy-12907"},
        }
    )

    assert resolved["sandbox"]["image"] == SWE_IMAGE
    parsed = SandboxConfig(**resolved["sandbox"])
    assert parsed.image == SWE_MAPPED
    assert parsed.provider == "vefaas"
    assert parsed.runtime_timeout == 7200
