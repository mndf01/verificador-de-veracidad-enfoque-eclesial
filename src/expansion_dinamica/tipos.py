"""Estructuras de datos compartidas por las etapas del módulo (sin dependencias externas)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaginaPDF:
    numero: int          # 1-indexado
    texto: str


@dataclass
class ResultadoExtraccion:
    paginas: list[PaginaPDF]
    total_paginas: int
    paginas_sin_texto: int
    sha256: str
    advertencias: list[str] = field(default_factory=list)
    paginas_reordenadas: int = 0       # páginas en dos columnas cuyo orden de lectura se reconstruyó


@dataclass
class Fragmento:
    """Unidad lista para indexar. Mantiene el mismo espíritu que un item del corpus base:
    texto + contexto jerárquico + id de norma, más la página para poder citar."""
    texto: str
    id_norma: int                  # nº de artículo/canon/numeral; 0 si la unidad no está numerada
    etiqueta: str                  # "Artículo 12", "Preámbulo", "Fragmento 3"...
    contexto: list[str]            # ["TÍTULO II – ...", "CAPÍTULO I – ..."]
    pagina_inicio: int
    pagina_fin: int
    parte: int = 1                 # si una unidad larga se parte en ventanas
    partes_total: int = 1


@dataclass
class ResultadoSegmentacion:
    fragmentos: list[Fragmento]
    esquema: str                   # "articulado" | "numerado" | "ventanas"
    advertencias: list[str] = field(default_factory=list)
