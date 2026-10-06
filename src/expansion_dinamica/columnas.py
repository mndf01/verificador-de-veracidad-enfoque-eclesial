"""Detección de maquetación en dos columnas y reconstrucción del orden de lectura.

Etapa PURA: recibe palabras con coordenadas (como las entrega pdfplumber) y devuelve las
líneas en el orden en que un humano las leería. No importa ninguna librería de PDF.

Por qué hace falta: un PDF no sabe qué es una "columna", solo ubica caracteres en la página.
Leer línea por línea de izquierda a derecha mezcla una frase de la columna izquierda con una
de la derecha y arruina tanto la segmentación como los embeddings.

Método (corte tipo XY, conservador):
  1. Se agrupan las palabras en filas por su posición vertical.
  2. Se busca un CANAL: una franja vertical vacía, en la zona central de la página, que casi
     ninguna fila atraviese (se toleran unas pocas filas que sí, como un título a todo el ancho).
  3. Se valida que sean columnas de verdad y no otra cosa (nota al margen, tabla de contenido
     con números a la derecha, tres columnas...): ambas deben tener texto suficiente, ancho
     suficiente y coincidir en vertical.
  4. Se lee por bloques: las filas "a todo el ancho" (título, encabezado de sección, pie)
     cortan la página en bandas; dentro de cada banda se lee primero la columna izquierda
     completa y luego la derecha.

Páginas dispersas: la última página de un documento suele tener la columna derecha casi vacía
y no cumpliría los criterios estrictos. Si el documento ya se sabe de dos columnas (porque
otras páginas lo confirmaron) se puede pasar su `canal` y entonces se aplica un criterio más
tolerante, que aun así exige que las líneas de la derecha estén alineadas con las de la izquierda.

Ante cualquier duda se devuelve None y el llamador conserva la lectura normal: es preferible
no reordenar a reordenar mal.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

# --- umbrales (en puntos tipográficos; una hoja A4 mide ~595 pt de ancho) -------------------
ANCHO_CANAL_MIN = 10.0          # un canal de columnas mide típicamente 12–40 pt
CRUCES_MAX_FRACCION = 0.10      # filas que pueden atravesar el canal (títulos, pies)...
CRUCES_MIN_PERMITIDOS = 2       # ...y como mínimo se toleran 2
BANDA_CENTRAL = 0.25            # el canal debe caer entre el 25 % y el 75 % del ancho del texto
ANCHO_TEXTO_MIN = 150.0         # con menos ancho no se analiza
PALABRAS_MIN_PAGINA = 60        # páginas casi vacías: no hay base para decidir
LINEAS_MIN_POR_COLUMNA = 6
FRACCION_MIN_PALABRAS = 0.20    # cada columna debe aportar al menos el 20 % de las palabras
ANCHO_MIN_COLUMNA = 0.25        # y ocupar al menos el 25 % del ancho del texto
SOLAPE_VERTICAL_MIN = 0.5       # las dos columnas deben coincidir al menos la mitad en vertical
FRACCION_MIN_FILAS_DOBLES = 0.30
FRACCION_MAX_FILAS_ANCHAS = 0.30   # si más filas atraviesan el canal, no es una página de columnas
# --- criterio tolerante (canal ya conocido por el resto del documento) ---
PALABRAS_MIN_CON_CANAL = 12
LINEAS_MIN_CON_CANAL = 2
FRACCION_MIN_ALINEADAS = 0.50      # filas de la derecha que comparten línea con la izquierda


@dataclass(frozen=True)
class Palabra:
    texto: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def centro_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def centro_y(self) -> float:
        return (self.top + self.bottom) / 2


def _agrupar_filas(palabras: list[Palabra]) -> list[list[Palabra]]:
    """Agrupa palabras en filas (de arriba abajo) y ordena cada fila de izquierda a derecha."""
    if not palabras:
        return []
    alto = median(p.bottom - p.top for p in palabras)
    tolerancia = max(2.0, 0.6 * alto)

    ordenadas = sorted(palabras, key=lambda p: p.centro_y)
    filas: list[list[Palabra]] = []
    fila = [ordenadas[0]]
    referencia = ordenadas[0].centro_y
    for p in ordenadas[1:]:
        if abs(p.centro_y - referencia) <= tolerancia:
            fila.append(p)
            referencia = sum(w.centro_y for w in fila) / len(fila)
        else:
            filas.append(sorted(fila, key=lambda w: w.x0))
            fila, referencia = [p], p.centro_y
    filas.append(sorted(fila, key=lambda w: w.x0))
    return filas


def _buscar_canal(filas: list[list[Palabra]]) -> tuple[float, float] | None:
    """Franja vertical más ancha, en la zona central, que casi ninguna fila atraviesa."""
    palabras = [p for fila in filas for p in fila]
    xmin = min(p.x0 for p in palabras)
    xmax = max(p.x1 for p in palabras)
    ancho = xmax - xmin
    if ancho < ANCHO_TEXTO_MIN:
        return None

    # Cuántas filas cubren cada punto x (diferencias acumuladas).
    n = int(ancho) + 3
    dif = [0] * (n + 1)
    for fila in filas:
        for p in fila:
            dif[int(p.x0 - xmin)] += 1
            dif[int(p.x1 - xmin) + 1] -= 1
    cobertura, acumulado = [], 0
    for i in range(n):
        acumulado += dif[i]
        cobertura.append(acumulado)

    permitidos = max(CRUCES_MIN_PERMITIDOS, CRUCES_MAX_FRACCION * len(filas))
    desde, hasta = int(BANDA_CENTRAL * ancho), int((1 - BANDA_CENTRAL) * ancho)

    mejor: tuple[int, int, int] | None = None       # (ancho, inicio, fin)
    inicio: int | None = None
    for i in range(desde, hasta + 1):
        libre = cobertura[i] <= permitidos
        if libre and inicio is None:
            inicio = i
        if inicio is not None and (not libre or i == hasta):
            fin = i if libre else i - 1
            if mejor is None or fin - inicio + 1 > mejor[0]:
                mejor = (fin - inicio + 1, inicio, fin)
            inicio = None

    if mejor is None or mejor[0] < ANCHO_CANAL_MIN:
        return None
    return xmin + mejor[1], xmin + mejor[2] + 1


def _es_fila_ancha(fila: list[Palabra], g0: float, g1: float, gc: float) -> bool:
    """Fila que no pertenece a una columna: una palabra abarca el canal, o las dos mitades están
    pegadas (el hueco entre ellas es solo un espacio entre palabras, no un canal)."""
    a = [p for p in fila if p.centro_x < gc]
    b = [p for p in fila if p.centro_x >= gc]
    abarca = any(p.x0 < g0 and p.x1 > g1 for p in fila)
    pegadas = bool(a and b) and min(p.x0 for p in b) - max(p.x1 for p in a) < ANCHO_CANAL_MIN
    return abarca or pegadas


def _analizar(
    palabras: list[Palabra], canal_conocido: tuple[float, float] | None = None
) -> tuple[tuple[float, float], list[list[Palabra]]] | None:
    """Devuelve (canal, filas) si la página es de dos columnas, o None.

    Sin `canal_conocido` se buscan y validan las columnas de forma estricta. Con él (el canal
    de otras páginas del mismo documento) se aplica el criterio tolerante para páginas dispersas.
    """
    minimo = PALABRAS_MIN_PAGINA if canal_conocido is None else PALABRAS_MIN_CON_CANAL
    if len(palabras) < minimo:
        return None
    filas = _agrupar_filas(palabras)

    if canal_conocido is None:
        canal = _buscar_canal(filas)
        if canal is None:
            return None
    else:
        canal = canal_conocido
    g0, g1 = canal
    gc = (g0 + g1) / 2

    # Las filas a todo el ancho (títulos, pies) se excluyen del análisis de cada columna:
    # de lo contrario sus palabras, que desbordan el canal, distorsionan todas las medidas.
    normales = [f for f in filas if not _es_fila_ancha(f, g0, g1, gc)]
    if len(filas) - len(normales) > FRACCION_MAX_FILAS_ANCHAS * len(filas):
        return None
    izq = [p for f in normales for p in f if p.centro_x < gc]
    der = [p for f in normales for p in f if p.centro_x >= gc]
    if not izq or not der:
        return None

    filas_izq, filas_der = _agrupar_filas(izq), _agrupar_filas(der)

    if canal_conocido is not None:
        # Criterio tolerante: la columna derecha puede ser corta, pero sus líneas deben estar
        # alineadas con las de la izquierda (así una firma o un título a la derecha no cuentan).
        if len(filas_izq) < LINEAS_MIN_CON_CANAL or len(filas_der) < LINEAS_MIN_CON_CANAL:
            return None
        con_der = [f for f in normales if any(p.centro_x >= gc for p in f)]
        alineadas = [f for f in con_der if any(p.centro_x < gc for p in f)]
        if len(alineadas) < FRACCION_MIN_ALINEADAS * len(con_der):
            return None
        return canal, filas

    # --- criterio estricto ---
    xmin = min(p.x0 for p in palabras)
    xmax = max(p.x1 for p in palabras)
    ancho = xmax - xmin
    total = len(izq) + len(der)
    if len(izq) < FRACCION_MIN_PALABRAS * total or len(der) < FRACCION_MIN_PALABRAS * total:
        return None
    if max(p.x1 for p in izq) - xmin < ANCHO_MIN_COLUMNA * ancho:
        return None
    if xmax - min(p.x0 for p in der) < ANCHO_MIN_COLUMNA * ancho:
        return None
    if len(filas_izq) < LINEAS_MIN_POR_COLUMNA or len(filas_der) < LINEAS_MIN_POR_COLUMNA:
        return None

    # Deben coincidir en vertical (una sangría en la mitad derecha no es una columna).
    t_i, b_i = min(p.top for p in izq), max(p.bottom for p in izq)
    t_d, b_d = min(p.top for p in der), max(p.bottom for p in der)
    if min(b_i, b_d) - max(t_i, t_d) < SOLAPE_VERTICAL_MIN * min(b_i - t_i, b_d - t_d):
        return None

    # Muchas filas deben tener texto a ambos lados del canal, bien separado.
    dobles = sum(
        1 for f in normales
        if any(p.centro_x < gc for p in f) and any(p.centro_x >= gc for p in f)
    )
    if dobles < max(3, FRACCION_MIN_FILAS_DOBLES * len(normales)):
        return None

    # Si dentro de una columna aparece otro canal, el diseño es ambiguo (3+ columnas, tabla):
    # mejor no tocarlo.
    for lado in (filas_izq, filas_der):
        if len(lado) >= LINEAS_MIN_POR_COLUMNA and _buscar_canal(lado) is not None:
            return None

    return canal, filas


def canal_dos_columnas(palabras: list[Palabra]) -> tuple[float, float] | None:
    """El canal (x0, x1) entre las dos columnas si la página las tiene de forma clara, o None."""
    resultado = _analizar(palabras)
    return resultado[0] if resultado else None


def detectar_dos_columnas(palabras: list[Palabra], canal: tuple[float, float] | None = None) -> bool:
    """¿Es una página de dos columnas? Con `canal` (de otras páginas) se usa el criterio tolerante."""
    return _analizar(palabras, canal) is not None


def _unir(palabras: list[Palabra]) -> str:
    return " ".join(p.texto for p in palabras)


def ordenar_lectura(
    palabras: list[Palabra], canal: tuple[float, float] | None = None
) -> list[str] | None:
    """Líneas de la página en orden de lectura, o None si no es de dos columnas."""
    resultado = _analizar(palabras, canal)
    if resultado is None:
        return None
    (g0, g1), filas = resultado
    gc = (g0 + g1) / 2

    salida: list[str] = []
    col_izq: list[str] = []
    col_der: list[str] = []

    def cerrar_banda() -> None:
        salida.extend(col_izq)
        salida.extend(col_der)
        col_izq.clear()
        col_der.clear()

    for fila in filas:
        if _es_fila_ancha(fila, g0, g1, gc):
            cerrar_banda()
            salida.append(_unir(fila))
        else:
            a = [p for p in fila if p.centro_x < gc]
            b = [p for p in fila if p.centro_x >= gc]
            if a:
                col_izq.append(_unir(a))
            if b:
                col_der.append(_unir(b))
    cerrar_banda()
    return salida
