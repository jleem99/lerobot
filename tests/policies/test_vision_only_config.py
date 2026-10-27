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

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition, transition_to_batch


@pytest.mark.parametrize("config_class", [PI05Config, SmolVLAConfig])
def test_vision_only_config_survives_independent_loading(tmp_path, config_class):
    config = config_class(
        use_proprio=False,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
    )
    config.validate_features()
    assert config.robot_state_feature is None
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    loaded.validate_features()
    assert not loaded.use_proprio
    assert loaded.robot_state_feature is None
    assert config_class().use_proprio  # Missing flags keep the original behavior.


def test_saved_pi05_prompt_omits_state_and_preserves_instruction(tmp_path):
    pipeline = PolicyProcessorPipeline(
        steps=[Pi05PrepareStateTokenizerProcessorStep(use_proprio=False)],
        name="policy_preprocessor",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    pipeline.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    for state in [torch.zeros(1, 7), torch.full((1, 7), 999.0)]:
        output = loaded({"task": ["put cube into the bowl"], "observation.state": state})
        assert output["task"] == ["Task: put cube into the bowl;\nAction: "]
