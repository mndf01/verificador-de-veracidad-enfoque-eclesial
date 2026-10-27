"""Entry point de consola para el verificador (`python -m verificador`).

Re-exporta el CLI de `src.cli`. Permite lanzar el orquestador como:

    python -m verificador evaluar https://...

Además, en `pyproject.toml` se registra el script `verificador` para que, tras
`pip install -e .`, el comando quede disponible en la terminal.
"""
import sys
from pathlib import Path

# Asegura que la raíz del proyecto esté en el path para importar `src.*` y `config`.
_RAIZ = Path(__file__).resolve().parent
if _RAIZ not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())