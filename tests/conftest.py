import asyncio
import sys
import threading

import pytest
from vllm import AsyncLLMEngine
from vllm.config import DeviceConfig
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser

from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService, start_grpc_server
from vllm_tgis_adapter.healthcheck import health_check
from vllm_tgis_adapter.tgis_utils.args import (
    EnvVarArgumentParser,
    add_tgis_args,
    postprocess_tgis_args,
)

from .utils import get_random_port, wait_until


@pytest.fixture()
def engine(mocker, monkeypatch):
    """Return a mocked vLLM engine."""
    engine = mocker.Mock(spec=AsyncLLMEngine)
    engine.engine = mocker.Mock(spec=_AsyncLLMEngine)
    mocker.patch("torch.cuda.memory_summary", return_value="mocked")
    engine.engine.device_config = mocker.Mock(spec=DeviceConfig)
    engine.engine.device_config.device = "cuda"

    engine.engine.stat_logger = "mocked"

    return engine


@pytest.fixture()
def args(monkeypatch, grpc_server_thread_port):
    # avoid parsing pytest arguments as vllm/vllm_tgis_adapter arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "__main__.py",
            f"--grpc-port={grpc_server_thread_port}",
        ],
    )

    parser = EnvVarArgumentParser(parser=make_arg_parser())
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())

    return args


@pytest.fixture()
def grpc_server_thread_port():
    """Port for grpc server."""
    return get_random_port()


@pytest.fixture()
def grpc_server_url(grpc_server_thread_port):
    """Port for grpc server."""
    return f"localhost:{grpc_server_thread_port}"


@pytest.fixture()
def grpc_server(engine, args, grpc_server_url):
    """Spins up grpc server in a background thread."""

    def _health_check():
        assert health_check(
            server_url=grpc_server_url,
            insecure=True,
            timeout=1,
            service=TextGenerationService.SERVICE_NAME,
        )

    global server  # noqa: PLW0602

    loop = asyncio.new_event_loop()

    async def run_server():
        global server  # noqa: PLW0603

        server = await start_grpc_server(engine, args)
        while server._server.is_running():  # noqa: SLF001
            await asyncio.sleep(1)

    def target():
        loop.run_until_complete(run_server())

    t = threading.Thread(target=target)
    t.start()

    async def stop():
        global server  # noqa: PLW0602

        await server.stop(grace=None)
        await server.wait_for_termination()

    try:
        wait_until(_health_check)
        yield server
    finally:
        loop.create_task(stop())  # noqa: RUF006
        t.join()
