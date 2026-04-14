import pychmp


def test_import_version() -> None:
    """Expose a non-empty package version at import time."""
    assert isinstance(pychmp.__version__, str)
    assert pychmp.__version__
