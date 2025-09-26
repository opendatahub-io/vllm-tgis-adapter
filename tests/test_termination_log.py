import pytest

from .utils import TaskFailedError


@pytest.fixture
def termination_log_fpath(tmp_path, monkeypatch):
    # create termination log before server starts
    temp_file = tmp_path / "termination_log.txt"
    monkeypatch.setenv("TERMINATION_LOG_DIR", str(temp_file))
    yield temp_file
    temp_file.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "server_args",
    [
        pytest.param(["--enable-lora"], id="enable-lora"),
        pytest.param(["--max-model-len=10241024"], id="huge-model-len"),
        pytest.param(["--model=google-bert/bert-base-uncased"], id="unsupported-model"),
    ],
    indirect=True,
)
def test_startup_fails(request, args, termination_log_fpath, lora_available):
    """Test that common set-up errors crash the server on startup.

    These errors should be properly reported in the termination log.

    """
    if lora_available and args.enable_lora:
        pytest.skip("This test requires a non-lora supported device to run")

    # Server fixture is called explicitly so that we can handle thrown exception
    with pytest.raises(TaskFailedError):
        _ = request.getfixturevalue("_servers")

    # read termination logs
    assert termination_log_fpath.exists()
