import requests


def test_startup(http_server_url, _http_server):
    """Test that the http_server fixture starts up properly."""
    requests.get(f"{http_server_url}/health").raise_for_status()


def test_completions(http_server_url, _http_server):
    response = requests.get(f"{http_server_url}/v1/models")
    response.raise_for_status()

    data = response.json()["data"]
    model_id = data[0]["id"]

    response = requests.post(
        f"{http_server_url}/v1/completions",
        json={
            "prompt": "The answer tho life the universe and everything is ",
            "model": model_id,
        },
    )
    response.raise_for_status()

    completion = response.json()

    generated_text = completion["choices"][0]["text"]

    assert generated_text
