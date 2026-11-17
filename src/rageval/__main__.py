"""CLI entry point: ``python -m rageval run | sweep | check``."""
import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
