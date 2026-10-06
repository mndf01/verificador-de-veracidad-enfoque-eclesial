"""Extracción de texto nativo desde un PDF cargado por el usuario.

Esta etapa es PURA: no toca Chroma, ni config, ni el modelo de embeddings.
Recibe una fuente (ruta, bytes o archivo subido) y devuelve texto por página.

Decisión de alcance (tesis, sección 1.4): NO se hace OCR. Un PDF escaneado se rechaza
con un mensaje claro en lugar de producir texto vacío o ruidoso.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from statistics import median

from .columnas import Palabra, canal_dos_columnas, detectar_dos_columnas, ordenar_lectura
from .errores import PdfExcedeLimiteError, PdfInvalidoError, PdfSinTextoNativoError
from .parametros import ParametrosExpansion
from .tipos import PaginaPDF, ResultadoExtraccion

log = logging.getLogger(__name__)

# Para decidir si un documento es de dos columnas se miran pocas páginas repartidas por todo
# el PDF (analizar la geometría de cada página es lento); si la mayoría lo es, se reordenan
# todas las que lo necesiten.
_MUESTRA_PAGINAS = 8
_FRACCION_MUESTRA_DOS_COLUMNAS = 0.4

FuentePDF = str | Path | bytes | bytearray | BinaryIO


def leer_bytes(fuente: FuentePDF) -> tuple[bytes, str | None]:
    """Normaliza la fuente a bytes y devuelve (contenido, nombre_sugerido).

    Acepta una ruta, bytes, o cualquier objeto con .read() (p. ej. el UploadedFile
    de Streamlit, que además trae .name).
    """
    if isinstance(fuente, (bytes, bytearray)):
        return bytes(fuente), None

    if isinstance(fuente, (str, Path)):
        ruta = Path(fuente)
        if not ruta.is_file():
            raise PdfInvalidoError(f"No se encontró el archivo: {ruta}")
        return ruta.read_bytes(), ruta.name

    if hasattr(fuente, "read"):
        if hasattr(fuente, "seek"):
            fuente.seek(0)
        datos = fuente.read()
        if not isinstance(datos, (bytes, bytearray)):
            raise PdfInvalidoError("El archivo cargado no es binario.")
        return bytes(datos), getattr(fuente, "name", None)

    raise PdfInvalidoError(f"Tipo de fuente no soportado: {type(fuente).__name__}")


def _palabras_de(pagina) -> list[Palabra]:
    """Palabras con coordenadas de una página de pdfplumber."""
    try:
        return [
            Palabra(w["text"], float(w["x0"]), float(w["x1"]), float(w["top"]), float(w["bottom"]))
            for w in pagina.extract_words()
        ]
    finally:
        flush = getattr(pagina, "flush_cache", None)
        if flush:
            flush()


def _muestra_uniforme(indices: list[int], n: int) -> list[int]:
    if len(indices) <= n:
        return indices
    paso = (len(indices) - 1) / (n - 1)
    return [indices[round(i * paso)] for i in range(n)]


def _reordenar_columnas(
    datos: bytes, paginas: list[PaginaPDF], params: ParametrosExpansion, advertencias: list[str]
) -> int:
    """Si el PDF es de dos columnas, reemplaza el texto de esas páginas por su orden de lectura.

    Devuelve cuántas páginas se reordenaron (0 = el documento se deja tal cual). Cualquier fallo
    aquí es NO fatal: se conserva el texto normal y se avisa.
    """
    try:
        import pdfplumber
    except ImportError:
        advertencias.append(
            "No se analizó la maquetación en columnas porque falta la librería pdfplumber; "
            "si el PDF es de dos columnas el texto puede salir mezclado."
        )
        return 0

    try:
        with pdfplumber.open(io.BytesIO(datos)) as pdf:
            con_texto = [
                i for i, p in enumerate(paginas) if len(p.texto.strip()) >= params.min_caracteres_por_pagina
            ]
            if not con_texto:
                return 0

            # 1) Se analizan pocas páginas repartidas por el documento (con criterio estricto).
            muestra = _muestra_uniforme(con_texto, _MUESTRA_PAGINAS)
            palabras_muestra = {i: _palabras_de(pdf.pages[i]) for i in muestra}
            canales = {i: canal_dos_columnas(w) for i, w in palabras_muestra.items()}
            estrictas = [c for c in canales.values() if c is not None]
            if not estrictas:
                return 0

            # 2) El canal típico del documento sirve para reconocer páginas dispersas (p. ej. la
            #    última, con la columna derecha casi vacía) que no pasarían el criterio estricto.
            canal_doc = (median(c[0] for c in estrictas), median(c[1] for c in estrictas))
            tolerantes = sum(
                detectar_dos_columnas(w, canal_doc) for i, w in palabras_muestra.items() if canales[i] is None
            )
            if (len(estrictas) + tolerantes) / len(muestra) < _FRACCION_MUESTRA_DOS_COLUMNAS:
                return 0

            # 3) Se reordena cada página que lo necesite (estricto primero; si no, tolerante).
            reordenadas = 0
            for i in con_texto:
                palabras = palabras_muestra.get(i) or _palabras_de(pdf.pages[i])
                lineas = ordenar_lectura(palabras) or ordenar_lectura(palabras, canal_doc)
                if lineas:
                    paginas[i] = PaginaPDF(numero=paginas[i].numero, texto="\n".join(lineas))
                    reordenadas += 1
    except Exception as e:
        log.warning("No se pudo analizar la maquetación en columnas: %s", e)
        advertencias.append(
            "No se pudo analizar la maquetación en columnas; el texto se extrajo en el orden normal."
        )
        return 0

    if reordenadas:
        advertencias.append(
            f"Se detectó maquetación en dos columnas en {reordenadas} de {len(paginas)} páginas; "
            "el texto se reordenó para leer cada columna completa. Revise los fragmentos si el "
            "documento tiene tablas o notas al margen."
        )
    return reordenadas


def extraer_paginas(fuente: FuentePDF, params: ParametrosExpansion | None = None) -> ResultadoExtraccion:
    """Extrae el texto de cada página y valida que el PDF sea utilizable.

    Raises:
        PdfInvalidoError: no es PDF, está corrupto, vacío o protegido con contraseña.
        PdfExcedeLimiteError: supera max_mb o max_paginas.
        PdfSinTextoNativoError: la mayoría de las páginas no tiene texto extraíble.
    """
    params = params or ParametrosExpansion()
    datos, _ = leer_bytes(fuente)

    if not datos:
        raise PdfInvalidoError("El archivo está vacío.")
    if len(datos) > params.max_mb * 1024 * 1024:
        raise PdfExcedeLimiteError(
            f"El PDF pesa {len(datos) / 1024 / 1024:.1f} MB; el máximo permitido es {params.max_mb:g} MB."
        )
    if b"%PDF-" not in datos[:1024]:
        raise PdfInvalidoError("El archivo no tiene formato PDF.")

    try:
        lector = PdfReader(io.BytesIO(datos), strict=False)
        if lector.is_encrypted:
            # Muchos PDF "protegidos" solo restringen copiar/imprimir y abren con clave vacía.
            if not lector.decrypt(""):
                raise PdfInvalidoError("El PDF está protegido con contraseña.")
        total = len(lector.pages)
    except PdfInvalidoError:
        raise
    except Exception as e:  # pypdf lanza varias excepciones propias según el tipo de corrupción
        raise PdfInvalidoError(f"No se pudo abrir el PDF (¿está dañado?): {e}") from e

    if total == 0:
        raise PdfInvalidoError("El PDF no contiene páginas.")
    if total > params.max_paginas:
        raise PdfExcedeLimiteError(
            f"El PDF tiene {total} páginas; el máximo permitido es {params.max_paginas}."
        )

    paginas: list[PaginaPDF] = []
    sin_texto = 0
    fallidas: list[int] = []
    for i, pagina in enumerate(lector.pages, start=1):
        try:
            texto = pagina.extract_text() or ""
        except Exception as e:
            log.warning("No se pudo extraer la página %d: %s", i, e)
            fallidas.append(i)
            texto = ""
        if len(texto.strip()) < params.min_caracteres_por_pagina:
            sin_texto += 1
        paginas.append(PaginaPDF(numero=i, texto=texto))

    if sin_texto / total > params.max_fraccion_paginas_vacias:
        raise PdfSinTextoNativoError(
            "El PDF no tiene texto extraíble: parece escaneado o formado por imágenes "
            f"({sin_texto} de {total} páginas sin texto). El sistema no aplica OCR; "
            "use un PDF con texto seleccionable."
        )

    advertencias: list[str] = []
    reordenadas = 0
    if params.detectar_columnas:
        reordenadas = _reordenar_columnas(datos, paginas, params, advertencias)

    if sin_texto:
        advertencias.append(
            f"{sin_texto} de {total} páginas tienen poco o ningún texto extraíble "
            "(imágenes, portadas o páginas en blanco); su contenido no se indexará."
        )
    if fallidas:
        advertencias.append(f"No se pudo leer el texto de las páginas: {fallidas[:10]}.")

    # Señal de codificación defectuosa (fuentes sin mapeo Unicode): mucha basura o pocas letras.
    muestra = "".join(p.texto for p in paginas)
    sin_espacios = [c for c in muestra if not c.isspace()]
    if sin_espacios:
        letras = sum(c.isalpha() for c in sin_espacios) / len(sin_espacios)
        reemplazos = muestra.count("\ufffd") / len(sin_espacios)
        if reemplazos > 0.02 or letras < 0.4:
            advertencias.append(
                "El texto extraído parece tener problemas de codificación; "
                "revise la calidad de los fragmentos antes de confiar en los resultados."
            )

    return ResultadoExtraccion(
        paginas=paginas,
        total_paginas=total,
        paginas_sin_texto=sin_texto,
        sha256=hashlib.sha256(datos).hexdigest(),
        advertencias=advertencias,
        paginas_reordenadas=reordenadas,
    )
