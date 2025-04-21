from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.logger import init_logger

from vllm_tgis_adapter.tgis_utils import logs

if TYPE_CHECKING:
    import argparse
    import socket

    from fastapi import FastAPI, Request, Response
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.protocol import AsyncEngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)


async def build_http_server(
    args: argparse.Namespace,
    engine: AsyncLLMEngine | AsyncEngineClient,
) -> FastAPI:
    # builds the vllm api server so we can pass reference to it
    # within the tgis adapter

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

    if hasattr(engine, "get_vllm_config"):
        vllm_config = await engine.get_vllm_config()
        maybe_coroutine = init_app_state(engine, vllm_config, app.state, args)
    else:
        model_config = await engine.get_model_config()
        maybe_coroutine = init_app_state(engine, model_config, app.state, args)
    if inspect.isawaitable(maybe_coroutine):
        await maybe_coroutine

    return app


async def run_http_server(
    args: argparse.Namespace,
    app: FastAPI,
    sock: socket.socket | None = None,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    # modified copy of vllm.entrypoints.openai.api_server.run_server that
    # allows passing of the engine

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

    # should only be used in versions of vllm >= 0.7.3
    if "sock" in inspect.getfullargspec(serve_http).args:
        serve_kwargs["sock"] = sock

    shutdown_coro = await serve_http(app, **serve_kwargs)

    # launcher.serve_http returns a shutdown coroutine to await
    # (The double await is intentional)
    await shutdown_coro
