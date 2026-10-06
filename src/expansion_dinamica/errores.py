"""Excepciones del módulo de expansión dinámica.

Todas heredan de ExpansionError, de modo que la interfaz (Streamlit) puede
capturar una sola clase y mostrar el mensaje al usuario tal cual.
"""


class ExpansionError(Exception):
    """Error base del módulo de expansión dinámica."""


class PdfInvalidoError(ExpansionError):
    """El archivo no es un PDF legible (corrupto, vacío o protegido con contraseña)."""


class PdfExcedeLimiteError(ExpansionError):
    """El PDF supera el tamaño o el número de páginas permitidos."""


class PdfSinTextoNativoError(ExpansionError):
    """El PDF no tiene capa de texto extraíble (escaneado o formado por imágenes).

    El sistema NO aplica OCR: es una exclusión explícita del alcance de la tesis.
    """


class DocumentoSinContenidoError(ExpansionError):
    """Tras limpiar y segmentar no quedó ningún fragmento utilizable."""


class NivelAutoridadInvalidoError(ExpansionError, ValueError):
    """El nivel de autoridad no es uno de los valores normalizados (0.33, 0.66, 1.0)."""
