#!/usr/bin/env -S uv run --active --script
# /// script
# dependencies = [
#   "tomli",
# ]
# ///
from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run

import tomli


def _get_build_dependencies_from_file(pyproject: Path) -> list[str]:
    if not pyproject.exists():
        raise FileNotFoundError(f"{pyproject} does not exist.")

    with pyproject.open("rb") as fh:
        pyproject_data = tomli.load(fh)

    build_system = pyproject_data.get("build-system", {})
    requires = build_system.get("requires", [])

    if not requires:
        raise ValueError("[build-system.requires] is empty")

    return requires


def install_vllm_build_deps(
    pyproject_file: str | None = None,
    constraints: str | Path | None = None,
) -> None:
    """Installs vllm build deps parsing them from the given vllm repo path/url.

    This is required because the vllm CPU build for PEP517/PEP518 style build is broken
    and we have to manually install build dependencies and use --no-build-isolation.
    """
    pyproject = Path(pyproject_file or "pyproject.toml")
    if not constraints:
        constraints = pyproject.parent / "requirements" / "cpu.txt"
    build_deps = _get_build_dependencies_from_file(pyproject)

    run(  # noqa: S603
        (
            "uv",
            "pip",
            "install",
            f"--overrides={constraints.resolve()}",
            *build_deps,
        ),
        check=True,
    )


if __name__ == "__main__":
    install_vllm_build_deps(
        sys.argv[1] if sys.argv[1:] else None,  # pyproject path
        sys.argv[2] if sys.argv[2:] else None,  # constraint file (default=cpu)
    )
