"""Agregación de las ventanas de una noticia en el vector de 16 features del RF
y asignación del veredicto en las 3 clases del proyecto.

Esquema de clases (decisión D2/D3 del mapa del proyecto):

    proximo       → la noticia habla de lo que el corpus cubre y lo respalda
    distante      → la noticia es del ámbito eclesial pero el corpus no la respalda
    no_definido   → el corpus no cubre el tema de la noticia (asignado por construcción)

Origen de cada etiqueta:

  * Las noticias de `afin` / `critica` / `neutral` (postura del MEDIO) se etiquetan
    por umbral del IVR agregado: `>= UMBRAL_PROXIMO` → proximo, resto → distante.
  * Las noticias de la categoría `no_definido` (temas ajenos al corpus: deportes,
    tecnología, política general...) reciben `no_definido` POR CONSTRUCCIÓN, sin
    mirar el IVR. Esto evita la circularidad: la etiqueta no sale de las variables.

`categoria_medio` se conserva como columna de ANÁLISIS (no es etiqueta) para poder
demostrar que el sistema no aprende la postura editorial del medio.

Los umbrales son PROVISIONALES: se recalcularán con los percentiles reales del
dataset completo (2000 noticias) antes del entrenamiento definitivo.
"""
from __future__ import annotations

from .generador import CATEGORIA_FUERA_DE_TEMA, NoticiaProcesada

# --- Umbral por percentiles (decisión C) ---------------------------------------------
# `proximo` = noticias en el PERCENTIL_PROXIMO superior de similitud DENTRO de los
# medios eclesiales (afin/critica/neutral). No es un valor absoluto: así el corte se
# adapta a la escala real del IVR y garantiza que la clase tenga ejemplos suficientes
# para que el Random Forest pueda aprenderla.
PERCENTIL_PROXIMO = 85

# Fallback si no se pudo calcular el percentil (p. ej. dataset mínimo).
UMBRAL_PROXIMO_FALLBACK = 0.60

# Clases del proyecto.
CLASES = ("proximo", "distante", "no_definido")


def calcular_umbral_proximo(valores_ivr: list[float], percentil: float = PERCENTIL_PROXIMO) -> float:
    """Umbral que deja el `percentil` superior de los valores por encima.

    Se aplica solo sobre las noticias de medios eclesiales (las de `no_definido`
    están fuera del corpus y no deben influir en el corte).

    :param valores_ivr: IVR agregado de las noticias del ámbito eclesial.
    :param percentil: 0-100. Con 85, el 15 % de mayor similitud queda como `proximo`.
    :return: valor de corte; `UMBRAL_PROXIMO_FALLBACK` si la lista está vacía.
    """
    if not valores_ivr:
        return UMBRAL_PROXIMO_FALLBACK
    ordenados = sorted(float(v) for v in valores_ivr)
    if len(ordenados) == 1:
        return ordenados[0]
    # Interpolación lineal del percentil (equivalente a numpy.percentile sin depender de numpy).
    pos = (len(ordenados) - 1) * (percentil / 100.0)
    bajo = int(pos)
    alto = min(bajo + 1, len(ordenados) - 1)
    peso = pos - bajo
    return ordenados[bajo] * (1 - peso) + ordenados[alto] * peso


def agregar_features(res: NoticiaProcesada) -> dict[str, float]:
    """Convierte el detalle por ventana en el vector de 16 features del RF.

    Se apoya en la mejor ventana para las señales finas (SB, literal, ontológico,
    autoridad) y usa los agregados multi-ventana para IVR y conteos.
    """
    from src.modelo.rf import NOMBRES_CARACTERISTICAS

    detalle = [v for v in res.ventanas_detalle if v.get("ok")]
    if not detalle:
        return {nombre: 0.0 for nombre in NOMBRES_CARACTERISTICAS}

    ivrs = [float(v["ivr"]) for v in detalle]
    ivr_max = max(ivrs)
    ivr_medio = sum(ivrs) / len(ivrs)
    ivr_min = min(ivrs)
    ordenados = sorted(ivrs, reverse=True)
    top3 = sum(ordenados[:3]) / min(3, len(ordenados))
    n_evid_max = max(int(v.get("n_evidencias", 0)) for v in detalle)

    senales_max = res.senales_max or {}
    features = {
        "n_evidencias": float(n_evid_max),
        "n_conceptos_premisa": float(len(res.conceptos_premisa) if res.conceptos_premisa else 0),
        "ivr_max": ivr_max,
        "ivr_medio": ivr_medio,
        "ivr_min": ivr_min,
        "ivr_brecha": ivr_max - ivr_medio,
        "sb_max": float(senales_max.get("sb") or ivr_max),
        "sb_medio": ivr_medio,
        "sb_min": ivr_min,
        "sb_brecha": ivr_max - ivr_min,
        "literal_max": float(senales_max.get("literal") or 0.0),
        "literal_medio": ivr_medio,
        "ontologico_max": float(senales_max.get("ontologico") or 0.0),
        "so_disponible": 1.0 if senales_max.get("ontologico") is not None else 0.0,
        "terminos_compartidos_max": float(senales_max.get("terminos_compartidos") or 0.0),
        "autoridad_max": float(senales_max.get("autoridad") or 0.0),
    }
    return {nombre: float(features.get(nombre, 0.0)) for nombre in NOMBRES_CARACTERISTICAS}


def asignar_veredicto(
    res: NoticiaProcesada,
    estrategia: str = "construccion",
    umbral_proximo: float | None = None,
) -> str:
    """Asigna la clase del proyecto a una noticia procesada.

    Estrategias:
      * `construccion` (por defecto, decisión D3): las noticias de la categoría
        `no_definido` se etiquetan así SIN mirar el IVR (el corpus no cubre su
        tema). El resto se etiqueta por umbral del IVR: proximo / distante.
        Es la estrategia recomendada porque evita la circularidad.
      * `umbral`: etiqueta todo por umbral del IVR, ignorando la categoría.

    :param umbral_proximo: corte por encima del cual una noticia es `proximo`.
        Si es None se usa `UMBRAL_PROXIMO_FALLBACK`; el CLI calcula este valor por
        percentiles sobre el conjunto completo antes de etiquetar.
    """
    corte = UMBRAL_PROXIMO_FALLBACK if umbral_proximo is None else umbral_proximo
    ivr = res.ivr_top3 if res.ivr_top3 > 0 else res.ivr_max

    if estrategia == "umbral":
        return "proximo" if ivr >= corte else "distante"

    # estrategia "construccion"
    if res.categoria == CATEGORIA_FUERA_DE_TEMA:
        return "no_definido"
    return "proximo" if ivr >= corte else "distante"
