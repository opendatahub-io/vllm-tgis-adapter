from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock

import pytest
from vllm.lora.request import LoRARequest

from vllm_tgis_adapter.grpc.adapters import AdapterStore, validate_adapters
from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
)

try:
    from vllm.entrypoints.openai.protocol import LoadLoRAAdapterRequest
except ImportError:
    from vllm.entrypoints.openai.protocol import (
        LoadLoraAdapterRequest as LoadLoRAAdapterRequest,
    )

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def vllm_model_handler() -> AsyncMock:
    mock_handler = AsyncMock()
    return mock_handler


@pytest.mark.asyncio
async def test_lora_adapter(vllm_model_handler):
    base_model_name = "/granite/granite-3b-base-v2/step_75000_ckpt"
    adapter_name = "granite-3b-code-instruct-lora"
    lora_path = str(FIXTURES_DIR) + "/" + adapter_name
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
        model_id=base_model_name,
    )
    mock_lora_requests = PropertyMock(
        side_effect=[
            [],
            [
                LoRARequest(
                    lora_name=adapter_name,
                    lora_int_id=0,
                    lora_path=lora_path,
                ),
            ],
        ]
    )
    type(vllm_model_handler).lora_requests = mock_lora_requests

    adapters = await validate_adapters(
        request,
        AdapterStore(cache_path=FIXTURES_DIR, adapters={}),
        vllm_model_handler,
    )
    # Ensure we created a LoRA adapter request
    assert "lora_request" in adapters
    assert adapters["lora_request"].lora_name == adapter_name
    assert isinstance(adapters["lora_request"], LoRARequest)
    # Ensure the vllm model handler received the load request
    vllm_model_handler.load_lora_adapter.assert_awaited_with(
        request=LoadLoRAAdapterRequest(
            lora_path=lora_path,
            lora_name=adapter_name,
        ),
        base_model_name=base_model_name,
    )


@pytest.mark.asyncio
async def test_lora_adapters_are_cached_remotely(vllm_model_handler):
    base_model_name = "/granite/granite-3b-base-v2/step_75000_ckpt"
    adapter_name = "granite-3b-code-instruct-lora"
    lora_path = str(FIXTURES_DIR) + "/" + adapter_name
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
        model_id=base_model_name,
    )
    mock_lora_requests = PropertyMock(
        side_effect=[
            [],
            [
                LoRARequest(
                    lora_name=adapter_name,
                    lora_int_id=0,
                    lora_path=lora_path,
                ),
            ],
            [
                LoRARequest(
                    lora_name=adapter_name,
                    lora_int_id=0,
                    lora_path=lora_path,
                ),
            ],
        ]
    )
    type(vllm_model_handler).lora_requests = mock_lora_requests
    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})

    adapters_1 = await validate_adapters(request, adapter_store, vllm_model_handler)
    adapters_2 = await validate_adapters(request, adapter_store, vllm_model_handler)
    # Ensure the vllm model handler only received a single load request
    vllm_model_handler.load_lora_adapter.assert_awaited_once_with(
        request=LoadLoRAAdapterRequest(
            lora_path=str(FIXTURES_DIR) + "/" + adapter_name,
            lora_name=adapter_name,
        ),
        base_model_name=base_model_name,
    )

    # Metadata isn't cached locally
    assert len(adapter_store.adapters) == 0
    # Same unique ID is reused for the second request
    assert (
        adapters_1["lora_request"].lora_int_id == adapters_2["lora_request"].lora_int_id
    )
