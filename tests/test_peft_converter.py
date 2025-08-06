from pathlib import Path

import pytest


def test_peft_converter(prompt_tune_path, tmp_path):
    # PROMPT_TUNING adapters are no longer supported in vLLM V1
    # This test is disabled as the conversion functionality has been removed
    pytest.skip("PROMPT_TUNING adapters are deprecated in vLLM V1")
