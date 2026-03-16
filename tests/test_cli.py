from pychmp.cli import main


def test_cli_version(capsys) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out
