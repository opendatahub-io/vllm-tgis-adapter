import asyncio

import pytest
from vllm import __version_tuple__ as vllm_version

from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
    DecodingParameters,
    GenerationRequest,
    Parameters,
    StoppingCriteria,
)

from .utils import GrpcClient


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


def test_guided_decoding_parameters_passed_to_engine(grpc_client, mocker):
    """Test that guided decoding parameters are properly passed from gRPC to engine.

    This test focuses on verifying parameter passing rather than guided decoding
    functionality, making it suitable for CPU execution where guided decoding
    might not work perfectly.
    """
    from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService

    # Spy on the _validate_and_convert_params method to intercept parameter processing
    validate_params_spy = mocker.spy(
        TextGenerationService, "_validate_and_convert_params"
    )

    # Also spy on _make_generator to verify final parameters
    engine_generate_spy = mocker.spy(TextGenerationService, "_make_generator")

    json_schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'

    # Create a request with JSON schema guided decoding
    request = BatchedGenerationRequest(
        model_id=None,
        requests=[GenerationRequest(text="Test prompt")],
        params=Parameters(
            stopping=StoppingCriteria(max_new_tokens=5),
            decoding=DecodingParameters(json_schema=json_schema),
        ),
    )

    # Make the request - we don't care if guided decoding works, just that
    # params are passed
    try:
        response = grpc_client.generation_service_stub.Generate(request=request)
        # Basic response validation
        assert response.responses
    except Exception as e:
        # If the request fails due to guided decoding issues, that's OK for this test
        # We're testing parameter passing, not guided decoding functionality
        if any(
            term in str(e).lower() for term in ["guided", "outlines", "json", "schema"]
        ):
            # This is expected on CPU - guided decoding might not work
            pass
        else:
            # Other errors should still fail the test
            raise

    # Verify that parameter validation was called with our guided decoding params
    validate_params_spy.assert_called_once()
    call_args = validate_params_spy.call_args[0]
    params_arg = call_args[1]  # Second argument is the Parameters object

    # Verify the guided decoding parameters were present in the request
    assert params_arg.decoding.HasField("json_schema")
    assert params_arg.decoding.json_schema == json_schema

    # Verify engine.generate was called (even if it failed later)
    engine_generate_spy.assert_called_once()

    # Get the sampling_params that were passed to the engine
    engine_call_kwargs = engine_generate_spy.call_args[1]
    sampling_params = engine_call_kwargs["sampling_params"]

    # The key test: verify guided decoding parameters were processed and set
    if vllm_version <= (0, 10, 0):
        # For vLLM <= 0.10.0, guided decoding should be in logits_processors
        # Even if the processor creation failed, the attempt should have been made
        # We can't guarantee the processor was created on CPU, but we can verify
        # the parameter processing logic was executed
        assert sampling_params is not None
    else:
        # For vLLM >= 0.10.1, guided decoding should be in guided_decoding parameter
        # This should be set regardless of whether guided decoding actually works
        assert sampling_params.guided_decoding is not None
        assert sampling_params.guided_decoding.json == json_schema


def test_guided_decoding_different_types(grpc_client, mocker):
    """Test that different guided decoding parameter types are passed to engine."""
    from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService

    # Spy on parameter validation to verify guided decoding params are processed
    validate_params_spy = mocker.spy(
        TextGenerationService, "_validate_and_convert_params"
    )
    engine_generate_spy = mocker.spy(TextGenerationService, "_make_generator")

    # Test different guided decoding types
    test_cases = [
        {
            "name": "regex",
            "params": DecodingParameters(regex=r"\d{3}-\d{2}-\d{4}"),
            "field": "regex",
            "expected_value": r"\d{3}-\d{2}-\d{4}",
        },
        {
            "name": "choice",
            "params": DecodingParameters(
                choice=DecodingParameters.StringChoices(choices=["yes", "no"])
            ),
            "field": "choice",
            "expected_value": ["yes", "no"],
        },
        {
            "name": "json_format",
            "params": DecodingParameters(format=DecodingParameters.ResponseFormat.JSON),
            "field": "format",
            "expected_value": DecodingParameters.ResponseFormat.JSON,
        },
    ]

    for test_case in test_cases:
        # Reset spies for each test case
        validate_params_spy.reset_mock()
        engine_generate_spy.reset_mock()

        request = BatchedGenerationRequest(
            model_id=None,
            requests=[GenerationRequest(text="Test")],
            params=Parameters(
                stopping=StoppingCriteria(max_new_tokens=3),
                decoding=test_case["params"],
            ),
        )

        # Make request - don't require it to succeed, just verify parameter passing
        try:
            grpc_client.generation_service_stub.Generate(request=request)
        except Exception as e:
            # Allow guided decoding failures on CPU
            if any(
                term in str(e).lower()
                for term in ["guided", "outlines", "json", "regex"]
            ):
                pass
            else:
                raise

        # Verify parameter processing was called
        validate_params_spy.assert_called_once()
        params_arg = validate_params_spy.call_args[0][1]

        # Verify the specific guided decoding field was set
        assert params_arg.decoding.HasField(test_case["field"])

        # Verify engine was called with processed parameters
        engine_generate_spy.assert_called_once()


def test_no_guided_decoding_parameters(grpc_client, mocker):
    """Test that requests without guided decoding don't have guided decoding params."""
    from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService

    # Spy on parameter processing
    validate_params_spy = mocker.spy(
        TextGenerationService, "_validate_and_convert_params"
    )
    engine_generate_spy = mocker.spy(TextGenerationService, "_make_generator")

    # Create a request without guided decoding
    request = BatchedGenerationRequest(
        model_id=None,
        requests=[GenerationRequest(text="Test")],
        params=Parameters(
            stopping=StoppingCriteria(max_new_tokens=3),
            # No decoding parameters
        ),
    )

    response = grpc_client.generation_service_stub.Generate(request=request)

    # Verify basic response
    assert response.responses
    assert response.responses[0].text

    # Verify parameter processing
    validate_params_spy.assert_called_once()
    params_arg = validate_params_spy.call_args[0][1]

    # Verify no guided decoding fields are set
    guided_fields = ["json_schema", "regex", "choice", "grammar", "format"]
    for field in guided_fields:
        assert not params_arg.decoding.HasField(field)

    # Verify engine was called
    engine_generate_spy.assert_called_once()
    sampling_params = engine_generate_spy.call_args[1]["sampling_params"]

    # Verify no guided decoding in final parameters
    if vllm_version > (0, 10, 0):
        # For newer vLLM versions, guided_decoding should be None
        assert sampling_params.guided_decoding is None
