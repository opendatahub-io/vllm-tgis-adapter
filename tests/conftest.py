import asyncio
import sys
import threading

import pytest
import requests
from vllm import AsyncLLMEngine
from vllm.config import DeviceConfig, ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding

import vllm_tgis_adapter
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
def args(monkeypatch, grpc_server_thread_port, http_server_thread_port):
    # avoid parsing pytest arguments as vllm/vllm_tgis_adapter arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "__main__.py",
            f"--grpc-port={grpc_server_thread_port}",
            f"--port={http_server_thread_port}",
        ],
    )

    parser = EnvVarArgumentParser(parser=make_arg_parser())
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())

    return args


@pytest.fixture()
def model_config(mocker):
    """Return a mocked vLLM ModelConfig."""

    def modelconfig_init(self, *args):
        self.model = "dummy_model"
        self.tokenizer = "mock_tokenizer"
        self.tokenizer_mode = "mock_tokenizer"
        self.trust_remote_code = False
        self.dtype = "bloat"
        self.seed = 42
        self.skip_tokenizer_init = True
        self.max_model_len = 42
        self.tokenizer_revision = "dummy_revision"

    mocker.patch("vllm.config.ModelConfig.__init__", modelconfig_init)
    return ModelConfig()


@pytest.fixture()
def grpc_server_thread_port():
    """Port for the grpc server."""
    return get_random_port()


@pytest.fixture()
def grpc_server_url(grpc_server_thread_port):
    """Url for the grpc server."""
    return f"localhost:{grpc_server_thread_port}"


@pytest.fixture()
def _grpc_server(engine, args, grpc_server_url):
    """Spins up the grpc server in a background thread."""

    def _health_check():
        assert health_check(
            server_url=grpc_server_url,
            insecure=True,
            timeout=1,
            service=TextGenerationService.SERVICE_NAME,
        )

    loop = asyncio.new_event_loop()

    global task  # noqa: PLW0602

    def target():
        global task  # noqa: PLW0603

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


@pytest.fixture()
def http_server_thread_port():
    """Port for the http server."""
    return get_random_port()


@pytest.fixture()
def http_server_url(http_server_thread_port):
    """Url for the http server."""
    return f"http://localhost:{http_server_thread_port}"


@pytest.fixture()
def _http_server(engine, args, http_server_url, model_config, mocker, monkeypatch):  # noqa: PLR0913
    """Spins up the http server in a background thread."""
    chat = mocker.Mock(spec=OpenAIServingChat)
    completion = mocker.Mock(spec=OpenAIServingCompletion)
    embedding = mocker.Mock(spec=OpenAIServingEmbedding)
    chat.engine = engine
    completion.engine = engine
    embedding.engine = engine

    mocker.patch("vllm_tgis_adapter.__main__.OpenAIServingChat", return_value=chat)
    mocker.patch(
        "vllm_tgis_adapter.__main__.OpenAIServingCompletion", return_value=completion
    )
    mocker.patch(
        "vllm_tgis_adapter.__main__.OpenAIServingEmbedding", return_value=embedding
    )

    def _health_check():
        requests.get(
            f"{http_server_url}/health",
            timeout=1,
        ).raise_for_status()

    global server  # noqa: PLW0602

    loop = asyncio.new_event_loop()

    monkeypatch.setattr(
        vllm_tgis_adapter.__main__,
        "engine_args",
        mocker.Mock(spec=AsyncEngineArgs),
        raising=False,
    )

    global task  # noqa: PLW0602

    def target():
        global task  # noqa: PLW0603

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
