"""Limpieza del texto extraído de un PDF (etapa pura).

Un PDF solo guarda instrucciones para ubicar caracteres en la página, no estructura
lógica (Zhu & Cole, 2022). Lo que sale de la extracción trae ruido que dañaría tanto la
segmentación por expresiones regulares como los embeddings:

  * ligaturas ("ﬁ"), guiones blandos, espacios no separables, caracteres de control
  * encabezados y pies de página repetidos en cada hoja
  * números de página sueltos
  * palabras partidas con guion al final de línea ("matri-" / "monio")

La salida es una lista plana de (numero_pagina, linea) ya depurada, que conserva la
página de origen para poder citar "p. N" después.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .tipos import PaginaPDF

# Nota: NO se usa NFKC porque convierte "º" (ordinal) en "o" y rompería "Art. 1º".
_TABLA = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "\u00ad": "", "\u200b": "", "\ufeff": "",
    "\xa0": " ", "\u2009": " ", "\u202f": " ",
})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACIOS = re.compile(r"[ \t]+")

# "12", "Página 12", "Pág. 3 de 10", "3/10", "- 4 -"
_NUM_PAGINA = re.compile(
    r"^(?:p[áa]g(?:ina|\.)?\s*)?\d{1,4}(?:\s*(?:de|/)\s*\d{1,4})?$|^[-–—]\s*\d{1,4}\s*[-–—]$",
    re.IGNORECASE,
)


def normalizar_unicode(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.translate(_TABLA)
    return _CONTROL.sub("", texto)


def _lineas(texto: str) -> list[str]:
    texto = normalizar_unicode(texto).replace("\r\n", "\n").replace("\r", "\n")
    lineas = (_ESPACIOS.sub(" ", l).strip() for l in texto.split("\n"))
    return [l for l in lineas if l]


def _clave_repeticion(linea: str) -> str:
    """'Página 3 de 10' y 'Página 4 de 10' deben contarse como la misma línea repetida."""
    return re.sub(r"\d+", "#", linea.lower())


def quitar_encabezados_y_pies(paginas_lineas: list[list[str]], bordes: int = 2) -> list[list[str]]:
    """Elimina líneas que se repiten en los bordes de la página y números de página sueltos.

    Solo mira las primeras y últimas `bordes` líneas de cada página: así nunca borra
    contenido del cuerpo aunque una frase se repita.

    Qué es un encabezado/pie se APRENDE únicamente de las páginas "completas" (con al menos
    2 líneas de cuerpo fuera de los bordes): en una página de 3 líneas todo sería "borde" y
    se podría tomar texto legítimo por encabezado. Lo aprendido se aplica luego a TODAS las
    páginas, de modo que la última página (casi siempre corta) también queda limpia. Los
    números de página sueltos se quitan siempre.

    Límite conocido: para reconocer "Página 3 de 10" los dígitos se ignoran al comparar,
    así que una línea de borde que se repita idéntica salvo por los números en ≥ 50 % de
    las páginas completas se tratará como encabezado/pie.
    """

    def indices_borde(lineas: list[str]) -> set[int]:
        if not lineas:
            return set()
        return set(range(min(bordes, len(lineas)))) | set(range(max(0, len(lineas) - bordes), len(lineas)))

    completas = [l for l in paginas_lineas if len(l) >= 2 * bordes + 2]

    conteo: Counter[str] = Counter()
    for lineas in completas:
        conteo.update({_clave_repeticion(lineas[i]) for i in indices_borde(lineas)})

    # Con 1 sola página completa no hay base para decir que algo "se repite". Con 2, la línea
    # debe estar en AMBAS; desde 3, en al menos la mitad (y no menos de 3).
    n = len(completas)
    umbral = None if n < 2 else (2 if n == 2 else max(3, math.ceil(0.5 * n)))

    limpias = []
    for lineas in paginas_lineas:
        borde = indices_borde(lineas)
        conservadas = []
        for i, linea in enumerate(lineas):
            if i in borde:
                if _NUM_PAGINA.match(linea):
                    continue
                if umbral is not None and conteo[_clave_repeticion(linea)] >= umbral:
                    continue
            conservadas.append(linea)
        limpias.append(conservadas)
    return limpias


_PALABRA = re.compile(r"[^\W\d_]+")
_COMPUESTA = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)+")
_FIN_CON_GUION = re.compile(r"([^\W\d_]+)-$")


def unir_guiones(lineas: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Resuelve las palabras partidas con guion al final de línea (también entre páginas).

    Un guion al final de línea es ambiguo: puede ser partición silábica ("matri-" / "monio" →
    "matrimonio") o un compuesto que se cortó justo en su guion ("económico-" / "social" →
    "económico-social"). Se decide con el propio documento, sin diccionario: si el compuesto
    con guion aparece completo en otra parte y la forma unida no, se conserva el guion; en
    cualquier otro caso se une como palabra partida.

    Solo se actúa si el guion va pegado a una letra y la línea siguiente empieza en minúscula;
    así no se tocan guiones de diálogo, rangos ("1 - 6") ni títulos.
    """
    compuestas = {m.lower() for _, l in lineas for m in _COMPUESTA.findall(l)}
    palabras = {m.lower() for _, l in lineas for m in _PALABRA.findall(l)}

    resultado: list[tuple[int, str]] = []
    i = 0
    while i < len(lineas):
        pagina, linea = lineas[i]
        while (
            linea.endswith("-")
            and len(linea) > 1
            and linea[-2].isalpha()
            and i + 1 < len(lineas)
            and lineas[i + 1][1][:1].islower()
        ):
            siguiente = lineas[i + 1][1]
            izq = _FIN_CON_GUION.search(linea)
            der = _PALABRA.match(siguiente)
            conservar = bool(
                izq and der
                and f"{izq.group(1)}-{der.group(0)}".lower() in compuestas
                and f"{izq.group(1)}{der.group(0)}".lower() not in palabras
            )
            linea = (linea if conservar else linea[:-1]) + siguiente
            i += 1
        resultado.append((pagina, linea))
        i += 1
    return resultado


def preparar_lineas(paginas: list[PaginaPDF]) -> list[tuple[int, str]]:
    """Pipeline completo de limpieza → lista plana de (pagina, linea)."""
    por_pagina = [_lineas(p.texto) for p in paginas]
    por_pagina = quitar_encabezados_y_pies(por_pagina)
    planas = [(p.numero, l) for p, lineas in zip(paginas, por_pagina) for l in lineas]
    return unir_guiones(planas)
