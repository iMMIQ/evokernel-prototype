from evokernel import __version__
from evokernel.cli import main


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_entrypoint_prints_placeholder_message(
    capsys,
) -> None:
    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out.strip() == "evokernel CLI is not implemented yet."
