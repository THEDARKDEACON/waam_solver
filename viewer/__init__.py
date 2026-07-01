"""Interactive melt-pool viewer (Taichi GGUI)."""

__all__ = ["main", "run"]


def main() -> None:
    from .app import main as _main
    _main()


def run(argv=None):
    from .app import run as _run
    return _run(argv)
