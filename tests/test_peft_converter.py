from pathlib import Path

from vllm_tgis_adapter.tgis_utils.convert_pt_to_prompt import convert_pt_to_peft


def test_peft_converter(prompt_tune_path, tmp_path):
    convert_pt_to_peft(input_dir=prompt_tune_path, output_dir=tmp_path)

    assert Path.exists(tmp_path / "adapter_config.json")
    assert Path.exists(tmp_path / "adapter_model.safetensors")
