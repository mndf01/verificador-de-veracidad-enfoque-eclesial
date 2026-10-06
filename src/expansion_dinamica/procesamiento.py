"""Pipeline puro: PDF → texto limpio → fragmentos. No usa Chroma ni embeddings.

Se separa de la sesión a propósito (arquitectura desacoplada): esta parte se puede
probar, ejecutar desde consola y reutilizar sin cargar el modelo SBERT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errores import DocumentoSinContenidoError
from .extractor_pdf import FuentePDF, extraer_paginas, leer_bytes
from .limpieza import preparar_lineas
from .parametros import ParametrosExpansion
from .segmentador import segmentar
from .tipos import ResultadoSegmentacion


@dataclass
class DocumentoProcesado:
    nombre: str
    sha256: str
    total_paginas: int
    segmentacion: ResultadoSegmentacion
    advertencias: list[str] = field(default_factory=list)   # extracción + segmentación

    @property
    def doc_id(self) -> str:
        return self.sha256[:10]


def _nombre_para(fuente: FuentePDF, sugerido: str | None, sha256: str, nombre: str | None) -> str:
    if nombre:
        return nombre.strip()
    if sugerido:
        return Path(sugerido).stem
    return f"documento_{sha256[:8]}"


def procesar_pdf(
    fuente: FuentePDF,
    nombre: str | None = None,
    params: ParametrosExpansion | None = None,
) -> DocumentoProcesado:
    """Extrae, limpia y segmenta un PDF.

    Raises:
        PdfInvalidoError, PdfExcedeLimiteError, PdfSinTextoNativoError: ver extractor_pdf.
        DocumentoSinContenidoError: no quedó ningún fragmento utilizable.
    """
    params = params or ParametrosExpansion()
    datos, sugerido = leer_bytes(fuente)          # se lee una sola vez (los uploads de Streamlit son streams)
    extraccion = extraer_paginas(datos, params)

    lineas = preparar_lineas(extraccion.paginas)
    segmentacion = segmentar(
        lineas,
        palabras_max=params.palabras_max_chunk,
        palabras_min=params.palabras_min_chunk,
    )
    if not segmentacion.fragmentos:
        raise DocumentoSinContenidoError(
            "No se pudo obtener texto utilizable del PDF después de limpiarlo."
        )

    return DocumentoProcesado(
        nombre=_nombre_para(fuente, sugerido, extraccion.sha256, nombre),
        sha256=extraccion.sha256,
        total_paginas=extraccion.total_paginas,
        segmentacion=segmentacion,
        advertencias=extraccion.advertencias + segmentacion.advertencias,
    )
