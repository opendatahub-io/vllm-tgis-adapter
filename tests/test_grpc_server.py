import asyncio
from typing import TYPE_CHECKING

import pytest
from vllm import __version_tuple__ as vllm_version

from vllm_tgis_adapter.grpc.pb.generation_pb2 import DecodingParameters

from .utils import GrpcClient

if TYPE_CHECKING:
    from vllm.sampling_params import GuidedDecodingParams


simplified_sql_grammar = """
    root ::= select_statement

    select_statement ::= "SELECT " column " from " table " where " condition

    column ::= "col_1 " | "col_2 "

    table ::= "table_1 " | "table_2 "

    condition ::= column "= " number

    number ::= "1 " | "2 "
"""


@pytest.fixture
def grpc_client(grpc_server_address, _servers):
    """Return a grpc client connected to the grpc server."""
    host, port = grpc_server_address.split(":")
    with GrpcClient(
        host=host,
        port=port,
        insecure=True,
    ) as client:
        yield client


def test_generation_request(grpc_client):
    response = grpc_client.make_request(
        "The answer to life the universe and everything is "
    )

    assert response.text
    assert response.generated_token_count
    assert response.stop_reason is not None


def test_tokenize_request(grpc_client):
    response_tokenize = grpc_client.make_request_tokenize(
        text="Please answer the following question.\nhow far is Paris from New York?",
    )

    assert response_tokenize.token_count


def test_generation_request_stream(grpc_client):
    streaming_response = grpc_client.make_request_stream(
        "The answer to life the universe and everything is ",
    )

    text_chunks: list[str] = [chunk.text for chunk in streaming_response]

    assert text_chunks
    assert len(text_chunks) == 11
    assert "".join(text_chunks)


def test_batched_generation_request(grpc_client):
    responses = list(
        grpc_client.make_request(
            [
                "The answer to life the universe and everything is ",
                "Medicinal herbs ",
            ],
        )
    )

    assert len(responses) == 2
    assert all(response.text for response in responses)


def test_lora_request(grpc_client, lora_adapter_name):
    response = grpc_client.make_request("hello", adapter_id=lora_adapter_name)

    assert response.text


def test_request_id(grpc_client, mocker):
    from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService, uuid
    from vllm_tgis_adapter.tgis_utils.logs import logger

    request_id_spy = mocker.spy(TextGenerationService, "request_id")
    # `caplog` doesn't appear to work here
    # So we can instead spy directly on the logger from tgis_utils.logs
    logger_spy = mocker.spy(logger, "info")

    # Test that the request ID is set to `x-correlation-id` if supplied
    response = grpc_client.make_request(
        "The answer to life the universe and everything is ",
        metadata=[("x-correlation-id", "dummy-correlation-id")],
    )
    assert response.text

    request_id_spy.assert_called_once()
    assert request_id_spy.spy_return == "dummy-correlation-id"
    request_id_spy.reset_mock()
    logger_spy.assert_called_once()
    log_statement = logger_spy.call_args[0][0] % tuple(logger_spy.call_args[0][1:])
    assert "correlation_id=dummy-correlation-id" in log_statement
    logger_spy.reset_mock()

    # Test that the request ID is set to a new uuid if `x-correlation-id` is not
    # supplied
    request_id = uuid.uuid4()
    mocker.patch.object(uuid, "uuid4", return_value=request_id)

    response = grpc_client.make_request(
        "The answer to life the universe and everything is ",
    )
    assert response.text

    request_id_spy.assert_called_once()
    assert request_id_spy.spy_return == request_id.hex
    logger_spy.assert_called_once()
    log_statement = logger_spy.call_args[0][0] % tuple(logger_spy.call_args[0][1:])
    assert "correlation_id=None" in log_statement
    logger_spy.reset_mock()


def test_error_handling(mocker):
    from vllm.engine.multiprocessing import MQEngineDeadError

    from vllm_tgis_adapter.grpc.grpc_server import _handle_exception, logger

    def dummy_func():
        pass

    class DummyEngine:
        errored = False
        is_running = True

    class DummyArg:
        engine = DummyEngine()

    # General error handling
    key_error = KeyError()
    dummy_arg_0 = DummyArg()
    with pytest.raises(KeyError):
        asyncio.run(_handle_exception(key_error, dummy_func, dummy_arg_0))

    engine_error = MQEngineDeadError("foo:bar")

    # Engine error handling
    spy = mocker.spy(logger, "error")

    # Does not raises exception
    asyncio.run(_handle_exception(engine_error, dummy_func, dummy_arg_0))
    spy.assert_called_once_with(engine_error)


@pytest.mark.parametrize(
    "decoding_params",
    [
        pytest.param(
            DecodingParameters(format="JSON"),
            id="guided_decoding_json",
        ),
        pytest.param(
            DecodingParameters(
                json_schema='{"schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}}'  # noqa: E501
            ),
            id="guided_decoding_json_schema",
        ),
        pytest.param(
            DecodingParameters(regex=r"\d.\d+"),
            id="guided_decoding_regex",
        ),
        pytest.param(
            DecodingParameters(choice={"choices": ["1", "2", "3", "4"]}),
            id="guided_decoding_choices",
        ),
        pytest.param(
            DecodingParameters(grammar=simplified_sql_grammar),
            id="guided_decoding_grammar",
            marks=pytest.mark.xfail(
                vllm_version <= (0, 10, 0),
                reason="grammar not supported in v0.10.0",
            ),
        ),
    ],
)
def test_guided_decoding_request(
    grpc_client,
    mocker,
    decoding_params: DecodingParameters,
):
    from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService

    spy = mocker.spy(TextGenerationService, "_make_generator")
    response = grpc_client.make_request(
        "hello", params_kwargs={"decoding": decoding_params}
    )
    assert response.text

    spy.assert_called_once()
    spied_sampling_params = spy.call_args.kwargs["sampling_params"]

    spied_decoding_params: GuidedDecodingParams = spied_sampling_params.guided_decoding
    if decoding_params.format == DecodingParameters.ResponseFormat.JSON:
        assert spied_decoding_params.json_object
    elif decoding_params.json_schema:
        assert spied_decoding_params.json == decoding_params.json_schema
    elif decoding_params.regex:
        assert spied_decoding_params.regex == decoding_params.regex
    elif decoding_params.choice:
        # choices are converted to a grammar
        assert all(
            value in spied_decoding_params.grammar
            for value in decoding_params.choice.choices
        )
    elif decoding_params.grammar:
        assert spied_decoding_params.grammar == decoding_params.grammar
    else:
        raise ValueError
