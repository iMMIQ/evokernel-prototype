import pytest

from evokernel import __version__
from evokernel.cli import main


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_entrypoint_requires_config_and_task_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "--config" in captured.err
    assert "--task" in captured.err
