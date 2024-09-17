from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import build_app
from vllm.logger import init_logger

try:
    from vllm.entrypoints.openai.api_server import init_app
except ImportError:  # vllm > 0.6.1.post2
    from vllm.entrypoints.openai.api_server import init_app_state


if TYPE_CHECKING:
    import argparse

    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.protocol import AsyncEngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)


async def run_http_server(
    args: argparse.Namespace,
    engine: AsyncLLMEngine | AsyncEngineClient,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    # modified copy of vllm.entrypoints.openai.api_server.run_server that
    # allows passing of the engine

    try:
        app = await init_app(engine, args)  # type: ignore[arg-type]
    except NameError:  # vllm > 0.6.1.post2
        app = build_app(args)
        model_config = await engine.get_model_config()
        init_app_state(engine, model_config, app.state, args)

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

    try:
        shutdown_coro = await serve_http(app, engine, **serve_kwargs)
    except TypeError:
        # vllm 0.5.4 backwards compatibility
        # HTTP server will not shut itself down when the engine dies
        shutdown_coro = await serve_http(app, **serve_kwargs)

    # launcher.serve_http returns a shutdown coroutine to await
    # (The double await is intentional)
    await shutdown_coro
