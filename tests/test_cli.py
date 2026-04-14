from pychmp.cli import main


def test_cli_version(capsys) -> None:
    """Print a non-empty version string for the CLI entrypoint."""
    assert main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out
