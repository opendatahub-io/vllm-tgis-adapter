import os
from pathlib import Path

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = "lint", "tests"
nox.options.default_venv_backend = "uv"
locations = "src", "tests"

versions = [
    "3.12",
    "3.11",
    "3.10",
    "3.9",
]


@nox.session(python=versions)
def build_vllm(session: nox.Session) -> None:
    if vllm_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        session.install(vllm_version)


@nox.session(python=versions)
def tests(session: nox.Session) -> None:
    if vllm_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        session.install(vllm_version)

    session.install(".[tests]")

    # Re-install without dependencies since `.[tests]` brings in
    # overrides that may bring the vllm version down
    if vllm_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        session.install(vllm_version, "--no-deps")

    session.run(
        "pytest",
        "--cov",
        "--cov-config=pyproject.toml",
        "--no-cov-on-fail",
        *session.posargs,
        env={"COVERAGE_FILE": f".coverage.{session.python}"},
    )


@nox.session(python=versions)
def lint(session: nox.Session) -> None:
    session.install("pre-commit")
    session.install("-e", ".[dev]")

    if run_mypy := "--mypy" in session.posargs:
        session.posargs.remove("--mypy")

    args = *(session.posargs or ("--show-diff-on-failure",)), "--all-files"
    session.run("pre-commit", "run", *args)

    if run_mypy:
        session.run("python", "-m", "mypy")


@nox.session(python=versions)
def build(session: nox.Session) -> None:
    session.install("build", "setuptools", "twine")

    session.run("python", "-m", "build")

    dists = Path("dist").glob("*")
    session.run("twine", "check", *dists, silent=True)


@nox.session(python=versions)
def dev(session: nox.Session) -> None:
    """Set up a python development environment for the project."""
    args = session.posargs or ("venv",)
    venv_dir = Path(args[0])

    session.log(f"Setting up virtual environment in {venv_dir}")
    session.install("uv")
    session.run("uv", "venv", venv_dir, silent=True)

    python = venv_dir / "bin/python"
    session.run(
        *f"{python} -m uv pip install -e .[dev]".split(),
        external=True,
    )
