"""Parámetros del módulo. Los valores por defecto replican la sección [expansion_dinamica]
de config.toml; así las etapas puras se pueden usar y probar sin cargar pydantic ni Chroma."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParametrosExpansion:
    # --- límites de entrada (el sistema no hace OCR) ---
    max_mb: float = 20.0
    max_paginas: int = 300
    min_caracteres_por_pagina: int = 100
    max_fraccion_paginas_vacias: float = 0.5
    # --- maquetación ---
    detectar_columnas: bool = True             # reordena PDFs de dos columnas (requiere pdfplumber)
    # --- segmentación ---
    palabras_max_chunk: int = 500
    palabras_min_chunk: int = 20
    # --- metadatos por defecto ---
    nivel_autoridad_default: float = 0.33      # Nivel 1 = normativa local (Rol.pdf)
    categoria_default: str = "Normativa Externa (carga dinámica)"
    # --- Chroma ---
    espacio_hnsw: str = "cosine"
    batch_size: int = 200
    max_results: int = 30
    telemetria: bool = False

    @classmethod
    def desde_config(cls) -> "ParametrosExpansion":
        """Lee config.toml vía config.py (import perezoso: solo la capa de sesión lo necesita)."""
        from config import obtener_configuraciones

        s = obtener_configuraciones()
        e, c = s.expansion_dinamica, s.chroma
        return cls(
            max_mb=e.max_mb,
            max_paginas=e.max_paginas,
            min_caracteres_por_pagina=e.min_caracteres_por_pagina,
            max_fraccion_paginas_vacias=e.max_fraccion_paginas_vacias,
            detectar_columnas=e.detectar_columnas,
            palabras_max_chunk=e.palabras_max_chunk,
            palabras_min_chunk=e.palabras_min_chunk,
            nivel_autoridad_default=e.nivel_autoridad_default,
            categoria_default=e.categoria_default,
            espacio_hnsw=c.espacio_hnsw,
            batch_size=c.batch_size,
            max_results=c.max_results,
            telemetria=c.telemetria,
        )
