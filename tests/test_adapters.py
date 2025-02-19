import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock

import pytest
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

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
async def test_caikit_prompt_adapter(vllm_model_handler):
    # Checks that decoder.pt style adapters from caikit_nlp are loaded correctly
    adapter_name = "bloom_sentiment_1"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request,
        AdapterStore(cache_path=FIXTURES_DIR, adapters={}),
        vllm_model_handler,
    )
    # Ensure we created a prompt adapter request
    assert "prompt_adapter_request" in adapters
    assert adapters["prompt_adapter_request"].prompt_adapter_name == adapter_name
    adapter_path = adapters["prompt_adapter_request"].prompt_adapter_local_path
    assert adapter_path is not None
    assert isinstance(adapters["prompt_adapter_request"], PromptAdapterRequest)

    # make sure the converted adapter is not in the cache directory
    assert str(FIXTURES_DIR) not in adapter_path
    assert "/tmp" in adapter_path

    # Check for the converted artifacts
    assert Path.exists(Path(adapter_path) / "adapter_config.json")
    assert Path.exists(Path(adapter_path) / "adapter_model.safetensors")


@pytest.mark.asyncio
async def test_prompt_adapter(vllm_model_handler):
    adapter_name = "bloomz-560m-prompt-adapter"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request,
        AdapterStore(cache_path=FIXTURES_DIR, adapters={}),
        vllm_model_handler,
    )
    # Ensure we created a prompt adapter request
    assert "prompt_adapter_request" in adapters
    assert adapters["prompt_adapter_request"].prompt_adapter_name == adapter_name
    assert isinstance(adapters["prompt_adapter_request"], PromptAdapterRequest)


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
async def test_prompt_adapters_are_cached_locally(vllm_model_handler):
    adapter_name = "bloomz-560m-prompt-adapter"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})

    adapters_1 = await validate_adapters(request, adapter_store, vllm_model_handler)
    adapters_2 = await validate_adapters(request, adapter_store, vllm_model_handler)

    # Metadata is only fetched and cached once
    assert len(adapter_store.adapters) == 1
    # Same unique ID is reused for the second request
    assert (
        adapters_1["prompt_adapter_request"].prompt_adapter_id
        == adapters_2["prompt_adapter_request"].prompt_adapter_id
    )
    # The vllm model handler isn't queried for prompt adapter requests
    vllm_model_handler.load_lora_adapter.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_store_handles_multiple_adapters(vllm_model_handler):
    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})

    adapter_name = "bloom_sentiment_1"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )
    adapters_1 = await validate_adapters(request, adapter_store, vllm_model_handler)

    adapter_name = "bloomz-560m-prompt-adapter"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )
    adapters_2 = await validate_adapters(request, adapter_store, vllm_model_handler)

    assert len(adapter_store.adapters) == 2
    assert (
        adapters_1["prompt_adapter_request"].prompt_adapter_id
        < adapters_2["prompt_adapter_request"].prompt_adapter_id
    )


@pytest.mark.asyncio
async def test_cache_handles_concurrent_loads(vllm_model_handler):
    # Check that the cache does not hammer the filesystem when accessed concurrently
    # Specifically, when concurrent requests for the same new adapter arrive

    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})
    # Use a caikit-style adapter that requires conversion, to test worst case
    adapter_name = "bloom_sentiment_1"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    # Fire off a bunch of concurrent requests for the same new adapter
    tasks = [
        asyncio.create_task(
            validate_adapters(request, adapter_store, vllm_model_handler)
        )
        for _ in range(1000)
    ]

    # Await all tasks
    await asyncio.gather(*tasks)

    # The adapter store should have only given out one unique ID
    assert adapter_store.next_unique_id == 1000002

    base_model_name = "/granite/granite-3b-base-v2/step_75000_ckpt"
    lora_adapter_name = "granite-3b-code-instruct-lora"
    lora_path = str(FIXTURES_DIR) + "/" + lora_adapter_name
    request = BatchedGenerationRequest(
        adapter_id=lora_adapter_name,
        model_id=base_model_name,
    )
    side_effect = [[]]
    mock_lora_request = [
        [
            LoRARequest(
                lora_name=lora_adapter_name,
                lora_int_id=0,
                lora_path=lora_path,
            ),
        ]
    ]
    side_effect.extend(mock_lora_request * 1000)
    mock_lora_requests = PropertyMock(side_effect=side_effect)
    type(vllm_model_handler).lora_requests = mock_lora_requests

    # Fire off a bunch of concurrent requests for the same new adapter
    tasks = [
        asyncio.create_task(
            validate_adapters(request, adapter_store, vllm_model_handler)
        )
        for _ in range(1000)
    ]

    # Await all tasks
    await asyncio.gather(*tasks)

    # Ensure the vllm model handler only received a single load request
    vllm_model_handler.load_lora_adapter.assert_awaited_once_with(
        request=LoadLoRAAdapterRequest(
            lora_path=str(FIXTURES_DIR) + "/" + lora_adapter_name,
            lora_name=lora_adapter_name,
        ),
        base_model_name=base_model_name,
    )
    # The adapter store id is incremented but not used
    assert adapter_store.next_unique_id == 1000003
    assert len(adapter_store.adapters) == 1
