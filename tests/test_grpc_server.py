import pytest


@pytest.mark.usefixtures("_grpc_server")
def test_startup():
    """Test that the grpc_server fixture starts up properly."""
