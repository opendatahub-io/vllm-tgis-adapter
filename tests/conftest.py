from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING

import pytest
import requests
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.__main__ import run_http_server
from vllm_tgis_adapter.grpc import run_grpc_server
from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService
from vllm_tgis_adapter.healthcheck import health_check
from vllm_tgis_adapter.tgis_utils.args import (
    EnvVarArgumentParser,
    add_tgis_args,
    postprocess_tgis_args,
)

from .utils import get_random_port, wait_until

if TYPE_CHECKING:
    import argparse

    from vllm.config import ModelConfig


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def lora_available() -> bool:
    # lora does not work on cpu
    return not vllm.config.is_cpu()


@pytest.fixture(scope="session")
def lora_adapter_name(request: pytest.FixtureRequest):
    if not request.getfixturevalue("lora_available"):
        pytest.skip("Lora is not available with this configuration")

    return "lora-test"


@pytest.fixture(scope="session")
def lora_adapter_path(request: pytest.FixtureRequest):
    if not request.getfixturevalue("lora_available"):
        pytest.skip("Lora is not available with this configuration")

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    return path


@pytest.fixture(scope="session")
def args(
    request: pytest.FixtureRequest,
    monkeysession,
    grpc_server_thread_port,
    http_server_thread_port,
    lora_available,
) -> argparse.Namespace:
    """Return parsed CLI arguments for the adapter/vLLM."""
    # avoid parsing pytest arguments as vllm/vllm_tgis_adapter arguments

    extra_args: list[str] = []
    if lora_available:
        name = request.getfixturevalue("lora_adapter_name")
        path = request.getfixturevalue("lora_adapter_path")

        extra_args.extend(("--enable-lora", f"--lora-modules={name}={path}"))

    monkeysession.setattr(
        sys,
        "argv",
        [
            "__main__.py",
            f"--grpc-port={grpc_server_thread_port}",
            f"--port={http_server_thread_port}",
            *extra_args,
        ],
    )

    parser = FlexibleArgumentParser("testing parser")
    parser = EnvVarArgumentParser(parser=make_arg_parser(parser))
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())

    return args


@pytest.fixture(scope="session")
def engine_args(args) -> AsyncEngineArgs:
    """Return AsyncEngineArgs from cli args."""
    return AsyncEngineArgs.from_cli_args(args)


@pytest.fixture(scope="session")
def engine(engine_args) -> AsyncLLMEngine:
    """Return a vLLM engine from the engine args."""
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,  # type: ignore[arg-type]
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    return engine


@pytest.fixture(scope="session")
def model_config(engine) -> ModelConfig:
    """Return a vLLM ModelConfig."""
    return asyncio.run(engine.get_model_config())


@pytest.fixture(scope="session")
def grpc_server_thread_port() -> int:
    """Port for the grpc server."""
    return get_random_port()


@pytest.fixture(scope="session")
def grpc_server_url(grpc_server_thread_port) -> str:
    """Url for the grpc server."""
    return f"localhost:{grpc_server_thread_port}"


@pytest.fixture(scope="session")
def _grpc_server(engine, args, grpc_server_url) -> None:
    """Spins up the grpc server in a background thread."""

    def _health_check():
        assert health_check(
            server_url=grpc_server_url,
            insecure=True,
            timeout=1,
            service=TextGenerationService.SERVICE_NAME,
        )

    loop = asyncio.new_event_loop()
    task: asyncio.Task | None = None

    def target():
        nonlocal task

        task = loop.create_task(run_grpc_server(engine, args, disable_log_stats=False))
        loop.run_until_complete(task)

    t = threading.Thread(target=target)
    t.start()

    try:
        wait_until(_health_check)
        yield
    finally:
        task.cancel()
        t.join()


@pytest.fixture(scope="session")
def http_server_thread_port(scope="session") -> int:
    """Port for the http server."""
    return get_random_port()


@pytest.fixture(scope="session")
def http_server_url(http_server_thread_port) -> str:
    """Url for the http server."""
    return f"http://localhost:{http_server_thread_port}"


@pytest.fixture(scope="session")
def _http_server(engine, model_config, engine_args, args, http_server_url) -> None:
    """Spins up the http server in a background thread."""

    def _health_check() -> None:
        requests.get(
            f"{http_server_url}/health",
            timeout=1,
        ).raise_for_status()

    global server  # noqa: PLW0602

    loop = asyncio.new_event_loop()

    task: asyncio.Task | None = None

    def target():
        nonlocal task

        task = loop.create_task(run_http_server(engine, args, model_config))
        loop.run_until_complete(task)

    t = threading.Thread(target=target)
    t.start()

    try:
        wait_until(_health_check)
        yield
    finally:
        task.cancel()
        t.join()
