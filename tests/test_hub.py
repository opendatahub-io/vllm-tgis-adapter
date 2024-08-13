from pathlib import Path

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm_tgis_adapter.tgis_utils.hub import (
    convert_files,
    download_weights,
    weight_files,
    weight_hub_files,
)

pytestmark = pytest.mark.hf_data


def test_convert_files():
    model_id = "facebook/opt-125m"
    local_pt_files = download_weights(model_id, extension=".bin")
    local_pt_files = [Path(p) for p in local_pt_files]
    local_st_files = [
        p.parent / f"{p.stem.removeprefix('pytorch_')}.safetensors"
        for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])

    found_st_files = weight_files(model_id)

    assert all(str(p) in found_st_files for p in local_st_files)


def test_weight_hub_files():
    filenames = weight_hub_files("facebook/opt-125m")
    assert filenames == ["model.safetensors"]


def test_weight_hub_files_llm():
    filenames = weight_hub_files("bigscience/bloom")
    assert filenames == [f"model_{i:05d}-of-00072.safetensors" for i in range(1, 73)]


def test_weight_hub_files_empty():
    filenames = weight_hub_files("bigscience/bloom", ".errors")
    assert filenames == []


def test_download_weights():
    files = download_weights("facebook/opt-125m")
    local_files = weight_files("facebook/opt-125m")
    assert files == local_files


def test_weight_files_error():
    with pytest.raises(LocalEntryNotFoundError):
        weight_files("bert-base-uncased")
