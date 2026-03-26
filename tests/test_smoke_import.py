from evokernel import __version__
from evokernel.cli import main


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_entrypoint_is_importable() -> None:
    assert callable(main)
