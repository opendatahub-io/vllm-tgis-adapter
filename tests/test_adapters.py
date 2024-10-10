from pathlib import Path

import pytest

from vllm_tgis_adapter.grpc.adapters import AdapterStore, validate_adapters
from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_validate_adapters():
    adapter_name = "bloom_sentiment_1"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request, AdapterStore(cache_path=FIXTURES_DIR, adapters={})
    )
    assert "prompt_adapter_request" in adapters
    assert adapters["prompt_adapter_request"].prompt_adapter_name == adapter_name
    adapter_path = adapters["prompt_adapter_request"].prompt_adapter_local_path
    assert adapter_path is not None

    assert Path.exists(Path(adapter_path) / "adapter_config.json")
    assert Path.exists(Path(adapter_path) / "adapter_model.safetensors")
