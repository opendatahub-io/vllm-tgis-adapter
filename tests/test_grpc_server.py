import pytest

from .utils import GrpcClient


@pytest.fixture()
def grpc_client(grpc_server_thread_port, _grpc_server):
    """Return a grpc client connected to the grpc server."""
    with GrpcClient(
        host="localhost",
        port=grpc_server_thread_port,
        insecure=True,
    ) as client:
        yield client


def test_generation_request(grpc_client, grpc_server_thread_port):
    response = grpc_client.make_request(
        "The answer to life the universe and everything is "
    )

    assert response.text
    assert response.generated_token_count
    assert response.stop_reason is not None


def test_generation_request_stream(grpc_client, grpc_server_thread_port):
    streaming_response = grpc_client.make_request_stream(
        "The answer to life the universe and everything is ",
    )

    text_chunks: list[str] = [chunk.text for chunk in streaming_response]

    assert text_chunks
    assert len(text_chunks) == 11
    assert "".join(text_chunks)


def test_batched_generation_request(grpc_client, grpc_server_thread_port):
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
