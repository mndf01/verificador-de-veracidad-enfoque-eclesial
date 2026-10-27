"""`python -m src.cli` — Interfaz de terminal del verificador."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())