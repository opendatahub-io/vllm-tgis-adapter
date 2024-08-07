from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import uvicorn
from vllm.entrypoints.openai.api_server import (
    init_app,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    import argparse

    from fastapi import FastAPI
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.protocol import AsyncEngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)


async def serve_http(
    app: FastAPI,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Gracefully stopping http server")
        await server.shutdown()


async def run_http_server(
    args: argparse.Namespace,
    engine: AsyncLLMEngine | AsyncEngineClient,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    # modified copy of vllm.entrypoints.openai.api_server.run_server that
    # allows passing of the engine

    app = await init_app(engine, args)  # type: ignore[arg-type]

    await serve_http(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )
