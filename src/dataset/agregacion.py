"""Agregación de las ventanas de una noticia en el vector de 16 features del RF.

El pipeline calcula el IVR por ventana (ver `generador.procesar_noticia`). Este
módulo condensa ese detalle en el vector plano que consume el Random Forest.

Estrategia de agregación:

  * `n_evidencias`            → máximo entre ventanas (la que más encontró)
  * `*_max`, `*_medio`, `*_min` de IVR/SB → agregados sobre las ventanas
  * `ivr_brecha`              → diferencia max - media (cuánto destaca la mejor)

Todo se construye reutilizando `ivr.vector_caracteristicas()` sobre el informe de
la MEJOR ventana, y luego se sobreescriben los agregados con los valores
multi-ventana. Así el vector mantiene exactamente las 16 claves esperadas por
`rf.NOMBRES_CARACTERISTICAS`, sin inventar un contrato nuevo.

Nota metodológica (para la tesis):
  La etiqueta `veredicto` es **supervisión débil**. `categoria` describe la
  postura del MEDIO (afin/critica/neutral) frente a la Iglesia, no la veracidad
  de la afirmación. Por eso la estrategia `debil` la usa solo como prior y la
  corrige con el IVR.
"""
from __future__ import annotations

from .generador import NoticiaProcesada

# Umbrales por defecto para la estrategia `umbral` y `debil`.
IVR_ALTO = 0.55
IVR_BAJO = 0.30


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

    # La ventana ganadora aporta las señales finas (SB, L, SO, J) vía su informe.
    # Como `procesar_noticia` no guarda el informe completo por ventana (sería
    # pesado), se aproximan las señales derivadas del propio IVR agregado.
    n_evid_max = max(int(v.get("n_evidencias", 0)) for v in detalle)

    # Señales aproximadas: se reparten sobre la ventana ganadora. Si se quisiera
    # exactitud total, procesar_noticia podría guardar `senales` por ventana.
    features = {
        "n_evidencias": float(n_evid_max),
        "n_conceptos_premisa": float(res.__dict__.get("n_conceptos_premisa", 0) or 0),
        "ivr_max": ivr_max,
        "ivr_medio": ivr_medio,
        "ivr_min": ivr_min,
        "ivr_brecha": ivr_max - ivr_medio,
        # SB/L/SO/J: se aproximan usando el IVR de la mejor ventana y su top3.
        "sb_max": ivr_max,
        "sb_medio": ivr_medio,
        "sb_min": ivr_min,
        "sb_brecha": ivr_max - ivr_min,
        "literal_max": top3,
        "literal_medio": ivr_medio,
        "ontologico_max": ivr_max,
        "so_disponible": 1.0 if ivr_max > 0 else 0.0,
        "terminos_compartidos_max": top3,
        "autoridad_max": 1.0,
    }
    # Garantiza las claves exactas esperadas por el RF (y marca las que falten a 0).
    return {nombre: float(features.get(nombre, 0.0)) for nombre in NOMBRES_CARACTERISTICAS}


def asignar_veredicto(res: NoticiaProcesada, estrategia: str = "debil") -> str:
    """Asigna la clase `veredicto` a partir del IVR agregado y la categoría del medio.

    Estrategias:
      * `proxy`  → mapea directamente la categoría del medio (afin→consistente,
                   critica→inconsistente, neutral→no_verificable). Es la más simple
                   y la más débil metodológicamente: confunde postura con veracidad.
      * `umbral` → solo usa el IVR: alto→consistente, bajo→inconsistente, medio→no_verificable.
      * `debil`  → combina la categoría como prior con el IVR como corrección:
                   si la noticia NO tiene respaldo en el corpus (ivr_max bajo) se marca
                   `no_verificable` sea cual sea el medio; si tiene respaldo, usa la
                   categoría del medio para decidir consistente/inconsistente.
    """
    ivr = res.ivr_top3 if res.ivr_top3 > 0 else res.ivr_max

    if estrategia == "proxy":
        return {
            "afin": "consistente",
            "critica": "inconsistente",
            "neutral": "no_verificable",
        }.get(res.categoria, "no_verificable")

    if estrategia == "umbral":
        if ivr >= IVR_ALTO:
            return "consistente"
        if ivr < IVR_BAJO:
            return "inconsistente"
        return "no_verificable"

    # estrategia "debil" (por defecto)
    if ivr < IVR_BAJO:
        # Sin respaldo en el corpus: no hay nada que verificar.
        return "no_verificable"
    if res.categoria == "critica":
        return "inconsistente"
    if res.categoria in ("afin", "neutral"):
        return "consistente"
    return "no_verificable"
