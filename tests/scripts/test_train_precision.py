#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train


@pytest.mark.parametrize(
    "storage, computation, expected",
    [
        ("float32", "bf16", "bf16"),
        ("bfloat16", "no", "no"),
        ("float32", None, "no"),
        ("bfloat16", None, "bf16"),
    ],
)
def test_explicit_accelerate_precision_is_independent_of_weight_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
    computation: str | None,
    expected: str,
) -> None:
    import accelerate

    if computation is None:
        monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    else:
        monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", computation)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/dataset"),
        policy=SmolVLAConfig(device="cpu", dtype=storage, push_to_hub=False),
        output_dir=tmp_path / "output",
    )
    observed = []

    class ReachedAcceleratorError(Exception):
        pass

    def accelerator(**kwargs):
        observed.append(kwargs["mixed_precision"])
        raise ReachedAcceleratorError

    monkeypatch.setattr(accelerate, "Accelerator", accelerator)
    with pytest.raises(ReachedAcceleratorError):
        train(config)
    assert observed == [expected]
