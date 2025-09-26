"""Adapters for requests.

Contains code to map api requests for adapters (e.g. peft prefixes, LoRA)
into valid LLM engine requests
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.protocol import ErrorResponse

from vllm_tgis_adapter.logging import init_logger

from .validation import TGISValidationError

try:
    from vllm.entrypoints.openai.protocol import LoadLoRAAdapterRequest
except ImportError:
    from vllm.entrypoints.openai.protocol import (
        LoadLoraAdapterRequest as LoadLoRAAdapterRequest,
    )

if TYPE_CHECKING:
    from vllm.entrypoints.grpc.pb.generation_pb2 import (
        BatchedGenerationRequest,
        BatchedTokenizeRequest,
        SingleGenerationRequest,
    )
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels
    from vllm.lora.request import LoRARequest

global_thread_pool = None  # used for loading adapter files from disk

VALID_ADAPTER_ID_PATTERN = re.compile("[/\\w\\-]+")

logger = init_logger(__name__)


@dataclasses.dataclass
class AdapterMetadata:
    unique_id: int  # Unique integer for vllm to identify the adapter
    adapter_type: str  # The string name of the peft adapter type, e.g. LORA
    full_path: str
    full_config: dict  # The loaded adapter_config.json dict


@dataclasses.dataclass
class AdapterStore:
    cache_path: str  # Path to local store of adapters to load from
    adapters: dict[str, AdapterMetadata]
    # Pick a large number to avoid colliding with vllm's adapter IDs
    next_unique_id: int = 1000001
    load_locks: dict[str, asyncio.Lock] = dataclasses.field(default_factory=dict)


async def validate_adapters(
    request: SingleGenerationRequest
    | BatchedGenerationRequest
    | BatchedTokenizeRequest,
    adapter_store: AdapterStore | None,
    vllm_model_handler: OpenAIServingModels,
) -> dict[str, LoRARequest]:
    """Validate the adapters.

    Takes the adapter name from the request and constructs a valid
    engine request if one is set. Raises if the requested adapter
    does not exist or adapter type is unsupported

    Returns the kwarg dictionary to add to an engine.generate() call.
    """
    global global_thread_pool  # noqa: PLW0603
    adapter_id = request.adapter_id
    # Backwards compatibility for `prefix_id` arg
    if not adapter_id and request.prefix_id:
        adapter_id = request.prefix_id

    if adapter_id and not adapter_store:
        TGISValidationError.AdaptersDisabled.error()

    if not adapter_id or not adapter_store:
        return {}

    # Guard against concurrent access for the same adapter
    async with adapter_store.load_locks.setdefault(adapter_id, asyncio.Lock()):
        # Check VLLM server lora cache if this request matches an existing
        # LoRA adapter
        if (
            existing_lora_request := _get_lora_request(vllm_model_handler, adapter_id)
        ) is not None:
            return {"lora_request": existing_lora_request}

        # If not already cached, we need to validate that files exist and
        # grab the type out of the adapter_config.json file
        if (adapter_metadata := adapter_store.adapters.get(adapter_id)) is None:
            _reject_bad_adapter_id(adapter_id)
            local_adapter_path = str(Path(adapter_store.cache_path) / adapter_id)

            loop = asyncio.get_running_loop()
            if global_thread_pool is None:
                global_thread_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                )

            # Increment the unique adapter id counter here in async land where we don't
            # need to deal with thread-safety
            unique_id = adapter_store.next_unique_id
            adapter_store.next_unique_id += 1

            adapter_metadata = await loop.run_in_executor(
                global_thread_pool,
                _load_adapter_metadata,
                adapter_id,
                local_adapter_path,
                unique_id,
            )

            # Add to cache
            # Query vllm's cache for lora requests
            if adapter_metadata.adapter_type == "LORA":
                lora_request = await _load_lora_adapter(
                    request,
                    adapter_id,
                    adapter_metadata,
                    vllm_model_handler,
                )
                return {"lora_request": lora_request}
            # Use our cache for everything else
            adapter_store.adapters[adapter_id] = adapter_metadata

    # All other types unsupported
    TGISValidationError.AdapterUnsupported.error(adapter_metadata.adapter_type)  # noqa: RET503


async def _load_lora_adapter(
    request: SingleGenerationRequest
    | BatchedGenerationRequest
    | BatchedTokenizeRequest,
    adapter_id: str,
    adapter_metadata: AdapterMetadata,
    vllm_model_handler: OpenAIServingModels,
) -> LoRARequest:
    load_request = LoadLoRAAdapterRequest(
        lora_path=adapter_metadata.full_path,
        lora_name=adapter_id,
    )
    load_result = await vllm_model_handler.load_lora_adapter(
        request=load_request,
        base_model_name=request.model_id,
    )
    if isinstance(load_result, ErrorResponse):
        raise ValueError(load_result.message)  ## noqa: TRY004
    if (
        existing_lora_request := _get_lora_request(vllm_model_handler, adapter_id)
    ) is not None:
        return existing_lora_request
    raise RuntimeError("vllm server failed to load LoRA adapter")


def _get_lora_request(
    vllm_model_handler: OpenAIServingModels,
    adapter_id: str,
) -> LoRARequest | None:
    lora_requests = vllm_model_handler.lora_requests
    if isinstance(lora_requests, dict):
        # vLLM > 0.9.2
        return lora_requests.get(adapter_id)

    # vLLM <= 0.9.2
    assert isinstance(lora_requests, list)
    for existing_lora_request in lora_requests:
        if existing_lora_request.lora_name == adapter_id:
            return existing_lora_request
    return None


def _load_adapter_metadata(adapter_id: str, adapter_path: str, unique_id: int) -> dict:
    """Get adapter metadata from files.

    Performs all the filesystem access required to deduce the type
    of the adapter. It's run in a separate thread pool executor so that file
    access does not block the main event loop.
    """
    if not Path(adapter_path).exists():
        TGISValidationError.AdapterNotFound.error(
            adapter_id, "directory does not exist"
        )

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not Path(adapter_config_path).exists():
        TGISValidationError.AdapterNotFound.error(
            adapter_id, "invalid adapter: no adapter_config.json found"
        )

    with open(adapter_config_path) as adapter_config_file:
        adapter_config = json.load(adapter_config_file)

    adapter_type = adapter_config.get("peft_type", None)
    adapter_metadata = AdapterMetadata(
        unique_id=unique_id,
        adapter_type=adapter_type,
        full_path=adapter_path,
        full_config=adapter_config,
    )

    return adapter_metadata


def _reject_bad_adapter_id(adapter_id: str) -> None:
    """Reject adapter.

    Raise if the adapter id attempts path traversal or
    has invalid file path characters.
    """
    if not VALID_ADAPTER_ID_PATTERN.fullmatch(adapter_id):
        TGISValidationError.InvalidAdapterID.error(adapter_id)

    cwd = Path().cwd()
    if not Path(adapter_id).resolve().is_relative_to(cwd):
        TGISValidationError.InvalidAdapterID.error(adapter_id)
