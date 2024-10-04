import pytest

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

    spy = mocker.spy(TextGenerationService, "request_id")
    response = grpc_client.make_request(
        "The answer to life the universe and everything is ",
        metadata=[("x-correlation-id", "dummy-correlation-id")],
    )
    assert response.text

    spy.assert_called_once()
    assert spy.spy_return == "dummy-correlation-id"

    spy.reset_mock()

    request_id = uuid.uuid4()
    mocker.patch.object(uuid, "uuid4", return_value=request_id)

    response = grpc_client.make_request(
        "The answer to life the universe and everything is ",
    )
    assert response.text

    spy.assert_called_once()
    assert spy.spy_return == request_id.hex
