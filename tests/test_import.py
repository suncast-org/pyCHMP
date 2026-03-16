import pychmp


def test_import_version() -> None:
    assert isinstance(pychmp.__version__, str)
    assert pychmp.__version__
