import importlib
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import nox

try:
    import tomllib
except ImportError:
    import tomli  # nox installs toml for python<3.11

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


@lru_cache
def _get_build_dependencies_from_file(pyproject: Path) -> list[str]:
    if not pyproject.exists():
        raise FileNotFoundError(f"{pyproject} does not exist.")

    with pyproject.open("rb") as fh:
        lib = tomllib if importlib.util.find_spec("tomllib") else tomli
        pyproject_data = lib.load(fh)

    build_system = pyproject_data.get("build-system", {})
    requires = build_system.get("requires", [])

    if not requires:
        raise ValueError("[build-system.requires] is empty")

    return requires


@lru_cache
def _get_build_dependencies_from_repo(repo_url: str, ref: str) -> list[str]:
    """Retrieve build dependencies from pyproject.toml for the given github repo/ref."""
    assert repo_url.startswith("https://github.com/"), (
        "this can only work with github urls"
    )

    raw_url = (
        repo_url.replace("github.com", "raw.githubusercontent.com")
        + f"/{ref}/pyproject.toml"
    )

    response = urllib.request.urlopen(raw_url)  # noqa: S310
    data = response.read()

    lib = tomllib if importlib.util.find_spec("tomllib") else tomli
    pyproject_data = lib.loads(data.decode())

    build_system = pyproject_data.get("build-system", {})
    requires = build_system.get("requires", [])

    if not requires:
        raise ValueError("[build-system.requires] is empty")

    return requires


def install_vllm_build_deps(session: nox.Session, vllm_version: str) -> None:
    """Installs vllm build deps parsing them from the given vllm repo path/url.

    This is required because the vllm CPU build for PEP517/PEP518 style build is broken
    and we have to manually install build dependencies and use --no-build-isolation.
    """
    maybe_repo_path = Path(vllm_version)
    if maybe_repo_path.exists() and (maybe_repo_path / "pyproject.toml").exists():
        pyproject = Path(vllm_version) / "pyproject.toml"
        build_deps = _get_build_dependencies_from_file(pyproject)
    elif vllm_version.startswith("git+https://github.com/"):
        url = vllm_version.lstrip("git+")
        if "@" in vllm_version:
            url, ref = url.split("@")
        else:
            import warnings

            ref = "main"
            warnings.warn(f"Ref not specified, assuming to be {ref}", stacklevel=2)

        build_deps = _get_build_dependencies_from_repo(url, ref)

    else:
        raise NotImplementedError(
            f"{vllm_version=} does not exist or url scheme is unsupported"
        )

    if not build_deps:
        raise ValueError("build deps are empty?")

    session.install(*build_deps)


def install_vllm_if_overridden(session: nox.Session) -> None:
    if vllm_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        install_vllm_build_deps(session, vllm_version)
        session.install("--no-build-isolation", vllm_version)


@nox.session(python=versions)
def build_vllm(session: nox.Session) -> None:
    install_vllm_if_overridden(session)


@nox.session(python=versions)
def tests(session: nox.Session) -> None:
    install_vllm_if_overridden(session)

    session.install(".[tests]")

    # Re-install vllm since `.[tests]` brings in
    # overrides that may bring the vllm version down.
    # Should be a no-op most of the time.
    install_vllm_if_overridden(session)

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
