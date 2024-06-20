import pytest
import requests


@pytest.mark.usefixtures("_http_server")
def test_startup(http_server_url):
    """Test that the http_server fixture starts up properly."""
    requests.get(f"{http_server_url}/health").raise_for_status()
    # requests.get(f"{http_server_url}/v1/models").raise_for_status()
