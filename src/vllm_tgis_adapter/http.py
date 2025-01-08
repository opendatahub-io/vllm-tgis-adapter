from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from fastapi.responses import JSONResponse
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.protocol import ErrorResponse, LoadLoraAdapterRequest
from vllm.logger import init_logger

from vllm_tgis_adapter.tgis_utils import logs

if TYPE_CHECKING:
    import argparse

    from fastapi import FastAPI, Request, Response
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.protocol import AsyncEngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)


def inject_lora_adapter_middleware(app: FastAPI, args: argparse.Namespace) -> None:  # noqa: C901
    invalid_adapter_names = set()

    @app.middleware("http")
    async def load_all_adapters(request: Request, call_next: Callable) -> Response:
        # For the /v1/models endpoint, we want to pre-load all adapters in the
        # ADAPTER_CACHE directory so that they properly appear in the response
        if "/models" not in request.url.path:
            # Not the models endpoint
            return await call_next(request)

        if not args.adapter_cache or not Path.is_dir(args.adapter_cache):
            logger.debug("No adapter cache, can't load all adapters")
            return await call_next(request)

        if not hasattr(request.app.state, "openai_serving_models"):
            logger.warning(
                "No models handler found- vLLM API is incompatible for adapter "
                "pre-loading"
            )
            return await call_next(request)
        models_handler = request.app.state.openai_serving_models

        loaded_adapters = [lora.name for lora in models_handler.lora_requests]
        nonlocal invalid_adapter_names

        adapter_names = os.listdir(args.adapter_cache)
        for adapter in adapter_names:
            if adapter in invalid_adapter_names or adapter in loaded_adapters:
                continue

            adapter_path = Path(args.adapter_cache) / adapter
            logger.info(
                "Pre-loading adapter '%s' from cache for /v1/models call", adapter
            )
            load_adapter_request = LoadLoraAdapterRequest(
                lora_path=adapter_path, lora_name=adapter
            )
            try:
                result = await models_handler.load_lora_adapter(load_adapter_request)
                if isinstance(result, ErrorResponse):
                    logger.warning(
                        "Adapter '%s' was invalid and could not be loaded", adapter
                    )
                    invalid_adapter_names.add(adapter)
            except Exception:
                logger.exception("Unexpected error loading adapter")
                invalid_adapter_names.add(adapter)

        return await call_next(request)

    @app.middleware("http")
    async def hijack_adapters(request: Request, call_next: Callable) -> Response:  # noqa: PLR0911
        if "/completions" not in request.url.path:
            return await call_next(request)

        if not args.adapter_cache or not Path.is_dir(args.adapter_cache):
            logger.debug("No adapter cache, cannot pre-load adapter for inference call")
            return await call_next(request)

        # Get the model id
        body_json = await request.json()
        model_id = body_json.get("model", None)
        if not model_id:
            logger.debug("Model id not found in request")
            return await call_next(request)

        # Grab the models handler and check if this model exists
        if not hasattr(request.app.state, "openai_serving_models"):
            logger.warning(
                "No models handler found- vLLM API is incompatible for adapter "
                "pre-loading"
            )
            return await call_next(request)
        models_handler = request.app.state.openai_serving_models

        # 1. Is this the base model?
        if models_handler.model_name() == model_id:
            return await call_next(request)

        # 2. Is it a loaded adapter?
        if model_id in {lora.name for lora in models_handler.lora_requests}:
            return await call_next(request)

        # 3. If not- is this the name of a valid adapter stored in ADAPTER_CACHE?
        adapter_path = Path(args.adapter_cache) / model_id
        if adapter_path.exists():
            try:
                logger.info(
                    "Pre-loading adapter '%s' from cache for inference call", model_id
                )
                load_adapter_request = LoadLoraAdapterRequest(
                    lora_path=adapter_path, lora_name=model_id
                )
                result = await models_handler.load_lora_adapter(load_adapter_request)
                # If there was an error loading the adapter, then short-circuit and
                # send _that_ error back to the user. Otherwise they'll only get
                # a 404 from the /chat/completions call
                if isinstance(result, ErrorResponse):
                    return JSONResponse(
                        content=result.model_dump(), status_code=result.code
                    )

            except Exception:
                # This shouldn't happen! Ignore so the request can be handled anyway
                logger.exception("Unexpected error loading adapter")

        # If loading an adapter was successful, the call can now use it
        return await call_next(request)


async def run_http_server(
    args: argparse.Namespace,
    engine: AsyncLLMEngine | AsyncEngineClient,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    # modified copy of vllm.entrypoints.openai.api_server.run_server that
    # allows passing of the engine

    app = build_app(args)

    @app.middleware("http")
    async def set_correlation_id(request: Request, call_next: Callable) -> Response:
        # If a correlation ID header is set, then use it as the request ID
        correlation_id = request.headers.get("X-Correlation-ID", None)
        if correlation_id:
            # NB: Setting a header here requires using byte arrays and lowercase
            headers = dict(request.scope["headers"])
            headers[b"x-request-id"] = correlation_id.encode()
            request.scope["headers"] = list(headers.items())
            # Tell the logger that the request ID is the correlation ID for this
            # request
            logs.set_correlation_id(correlation_id, correlation_id)

        return await call_next(request)

    model_config = await engine.get_model_config()
    init_app_state(engine, model_config, app.state, args)
    inject_lora_adapter_middleware(app, args)

    serve_kwargs = {
        "host": args.host,
        "port": args.port,
        "log_level": args.uvicorn_log_level,
        "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
        "ssl_keyfile": args.ssl_keyfile,
        "ssl_certfile": args.ssl_certfile,
        "ssl_ca_certs": args.ssl_ca_certs,
        "ssl_cert_reqs": args.ssl_cert_reqs,
    }
    serve_kwargs.update(uvicorn_kwargs)

    shutdown_coro = await serve_http(app, **serve_kwargs)

    # launcher.serve_http returns a shutdown coroutine to await
    # (The double await is intentional)
    await shutdown_coro
