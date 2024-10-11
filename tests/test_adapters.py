import asyncio
from pathlib import Path

import pytest
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

from vllm_tgis_adapter.grpc.adapters import AdapterStore, validate_adapters
from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_caikit_prompt_adapter():
    # Checks that decoder.pt style adapters from caikit_nlp are loaded correctly
    adapter_name = "bloom_sentiment_1"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request, AdapterStore(cache_path=FIXTURES_DIR, adapters={})
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
async def test_prompt_adapter():
    adapter_name = "bloomz-560m-prompt-adapter"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request, AdapterStore(cache_path=FIXTURES_DIR, adapters={})
    )
    # Ensure we created a prompt adapter request
    assert "prompt_adapter_request" in adapters
    assert adapters["prompt_adapter_request"].prompt_adapter_name == adapter_name
    assert isinstance(adapters["prompt_adapter_request"], PromptAdapterRequest)


@pytest.mark.asyncio
async def test_lora_adapter():
    adapter_name = "granite-3b-code-instruct-lora"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapters = await validate_adapters(
        request, AdapterStore(cache_path=FIXTURES_DIR, adapters={})
    )
    # Ensure we created a LoRA adapter request
    assert "lora_request" in adapters
    assert adapters["lora_request"].lora_name == adapter_name
    assert isinstance(adapters["lora_request"], LoRARequest)


@pytest.mark.asyncio
async def test_adapters_are_cached():
    adapter_name = "granite-3b-code-instruct-lora"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )

    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})

    adapters_1 = await validate_adapters(request, adapter_store=adapter_store)
    adapters_2 = await validate_adapters(request, adapter_store=adapter_store)

    # Metadata is only fetched and cached once
    assert len(adapter_store.adapters) == 1
    # Same unique ID is re-used for the second request
    assert (
        adapters_1["lora_request"].lora_int_id == adapters_2["lora_request"].lora_int_id
    )


@pytest.mark.asyncio
async def test_store_handles_multiple_adapters():
    adapter_store = AdapterStore(cache_path=FIXTURES_DIR, adapters={})

    adapter_name = "granite-3b-code-instruct-lora"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )
    adapters_1 = await validate_adapters(request, adapter_store=adapter_store)

    adapter_name = "bloomz-560m-prompt-adapter"
    request = BatchedGenerationRequest(
        adapter_id=adapter_name,
    )
    adapters_2 = await validate_adapters(request, adapter_store=adapter_store)

    assert len(adapter_store.adapters) == 2
    assert (
        adapters_1["lora_request"].lora_int_id
        < adapters_2["prompt_adapter_request"].prompt_adapter_id
    )


@pytest.mark.asyncio
async def test_cache_handles_concurrent_loads():
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
        asyncio.create_task(validate_adapters(request, adapter_store=adapter_store))
        for _ in range(1000)
    ]

    # Await all tasks
    await asyncio.gather(*tasks)

    # The adapter store should have only given out one unique ID
    assert adapter_store.next_unique_id == 2
