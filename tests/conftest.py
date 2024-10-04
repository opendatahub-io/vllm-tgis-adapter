from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeVar

import pytest
import requests
import vllm
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.__main__ import run_and_catch_termination_cause, start_servers
from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService
from vllm_tgis_adapter.healthcheck import health_check
from vllm_tgis_adapter.tgis_utils.args import (
    EnvVarArgumentParser,
    add_tgis_args,
    postprocess_tgis_args,
)

from .utils import TaskFailedError, get_random_port, wait_until

if TYPE_CHECKING:
    import argparse
    from collections.abc import Generator

    T = TypeVar("T")

    YieldFixture = Generator[T, None, None]
    ArgFixture = Annotated[T, pytest.fixture]


@pytest.fixture
def prompt_tune_path():
    return Path(__file__).parent / "fixtures" / "bloom_sentiment_1"


@pytest.fixture
def lora_available() -> bool:
    # lora does not work on cpu
    return not vllm.config.current_platform.is_cpu()


@pytest.fixture
def lora_adapter_name(request: pytest.FixtureRequest) -> str:
    if not request.getfixturevalue("lora_available"):
        pytest.skip("Lora is not available with this configuration")

    return "lora-test"


@pytest.fixture
def lora_adapter_path(request: pytest.FixtureRequest) -> str:
    if not request.getfixturevalue("lora_available"):
        pytest.skip("Lora is not available with this configuration")

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    return path


@pytest.fixture(
    params=[
        # pytest.param(True, id="disable-frontend-multiprocessing=True"),
        pytest.param(False, id="disable-frontend-multiprocessing=False"),
    ]
)
def disable_frontend_multiprocessing(request):
    """Enable or disable the frontend-multiprocessing feature."""
    return request.param


@pytest.fixture
def server_args(request: pytest.FixtureRequest):
    return request.param if hasattr(request, "param") else []


@pytest.fixture
def args(  # noqa: PLR0913
    request: pytest.FixtureRequest,
    monkeypatch,
    grpc_server_port: ArgFixture[int],
    http_server_port: ArgFixture[int],
    lora_available: ArgFixture[bool],
    disable_frontend_multiprocessing,
    server_args: ArgFixture[list[str]],
) -> argparse.Namespace:
    """Return parsed CLI arguments for the adapter/vLLM."""
    # avoid parsing pytest arguments as vllm/vllm_tgis_adapter arguments

    # Extra server init flags
    extra_args: list[str] = [*server_args]
    if lora_available:
        name = request.getfixturevalue("lora_adapter_name")
        path = request.getfixturevalue("lora_adapter_path")

        extra_args.extend(("--enable-lora", f"--lora-modules={name}={path}"))

    if disable_frontend_multiprocessing:
        extra_args.append("--disable-frontend-multiprocessing")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "__main__.py",
            f"--grpc-port={grpc_server_port}",
            f"--port={http_server_port}",
            *extra_args,
        ],
    )

    parser = FlexibleArgumentParser("testing parser")
    parser = EnvVarArgumentParser(parser=make_arg_parser(parser))
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())

    return args


@pytest.fixture
def grpc_server_port() -> int:
    """Port for the grpc server."""
    return get_random_port()


@pytest.fixture
def grpc_server_address(grpc_server_port: ArgFixture[int]) -> str:
    """Address for the grpc server."""
    return f"localhost:{grpc_server_port}"


@pytest.fixture
def http_server_port() -> int:
    """Port for the http server."""
    return get_random_port()


@pytest.fixture
def http_server_url(http_server_port: ArgFixture[int]) -> str:
    """Url for the http server."""
    return f"http://localhost:{http_server_port}"


@pytest.fixture
def _servers(
    args: ArgFixture[argparse.Namespace],
    grpc_server_address: ArgFixture[str],
    http_server_url: ArgFixture[str],
    monkeypatch,
) -> YieldFixture[None]:
    """Run the servers in an asyncio loop in a background thread."""
    global server  # noqa: PLW0602

    loop = asyncio.new_event_loop()

    task: asyncio.Task | None = None

    def _health_check() -> None:
        if not task:
            raise TaskFailedError

        if task.done():
            exc = task.exception()
            if exc:
                raise TaskFailedError from exc

            raise TaskFailedError

        requests.get(
            f"{http_server_url}/health",
            timeout=1,
        ).raise_for_status()

        assert health_check(
            server_url=grpc_server_address,
            insecure=True,
            timeout=1,
            service=TextGenerationService.SERVICE_NAME,
        )

    # patch the add_signal_handler method so that instantiating the servers
    # does not try to modify signal handlers in a child thread, which cannot be done
    def dummy_signal_handler(*args, **kwargs):
        pass

    monkeypatch.setattr(loop, "add_signal_handler", dummy_signal_handler)

    def target():
        nonlocal task
        task = loop.create_task(start_servers(args))
        run_and_catch_termination_cause(loop, task)

    t = threading.Thread(target=target)
    t.start()

    try:
        wait_until(_health_check)
        yield
    finally:
        if task:
            task.cancel()

        t.join()

    # workaround: Instantiating the TGISStatLogger multiple times creates
    # multiple Gauges etc which can only be instantiated once.
    # By unregistering the Collectors from the REGISTRY we can
    # work around this problem.

    from prometheus_client.registry import REGISTRY

    for name in list(REGISTRY._collector_to_names.keys()):  # noqa: SLF001
        REGISTRY.unregister(name)
