"""Módulo de expansión dinámica: carga transitoria de un PDF normativo externo.

API pública:
    SesionExpansion        → cargar_pdf / consultar / consultar_combinado / cerrar
    procesar_pdf           → PDF → fragmentos, sin Chroma (útil para depurar)
    ParametrosExpansion    → límites y tamaños (se lee de config.toml)
    ExpansionError y subclases para mostrar mensajes claros en la interfaz
"""
from .errores import (
    DocumentoSinContenidoError,
    ExpansionError,
    NivelAutoridadInvalidoError,
    PdfExcedeLimiteError,
    PdfInvalidoError,
    PdfSinTextoNativoError,
)
from .parametros import ParametrosExpansion
from .procesamiento import DocumentoProcesado, procesar_pdf
from .sesion import NIVELES_AUTORIDAD, TIPO_CORPUS, ResumenDocumento, SesionExpansion

__all__ = [
    "SesionExpansion", "ResumenDocumento", "ParametrosExpansion", "DocumentoProcesado",
    "procesar_pdf", "TIPO_CORPUS", "NIVELES_AUTORIDAD",
    "ExpansionError", "PdfInvalidoError", "PdfExcedeLimiteError", "PdfSinTextoNativoError",
    "DocumentoSinContenidoError", "NivelAutoridadInvalidoError",
]
