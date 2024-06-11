def test_startup(grpc_server):
    """Test that the grpc_server fixture starts up properly."""
    assert grpc_server._server.is_running()  # noqa: SLF001
