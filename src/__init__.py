"""Paquete `src` del verificador.

Agrupa las capas desacopladas del sistema (base de datos, ETL, expansión,
IVR, modelo, web). El resto del proyecto se ejecuta desde la raíz con imports
prefijados `src.` y módulos `python -m`; este `__init__` permite también que
`pip install -e .` registre los paquetes como instalables (entry point
`verificador` en pyproject.toml).

No expone lógica aquí: cada capa tiene su propia API pública.
"""