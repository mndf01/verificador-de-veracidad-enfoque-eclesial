"""Calibración de los pesos del IVR a partir de la importancia de variables del RF.

Contexto (ver MAPA-PROYECTO.md, tarea 7):

Los pesos `SB=0.40, L=0.25, SO=0.20, J=0.15` son una **hipótesis de partida**: el
propio `config.IvrConfig` documenta que se calibran en una fase posterior con la
importancia de variables del Random Forest.

Este módulo implementa esa calibración en tres pasos:

  1. **Ablación.** Entrena el RF SIN las columnas `ivr_*` (que son combinaciones
     lineales de SB/L/SO/J con los pesos vigentes). Incluirlas al medir la
     importancia de las señales sería circular.
  2. **Agregación por familia.** Suma la importancia de todas las features que
     provienen de una misma señal (`sb_max`, `sb_medio`, `sb_min`, `sb_brecha` → SB).
  3. **Comparación de candidatos por AUC.** Mide la capacidad discriminante del
     IVR de cada juego de pesos para separar `proximo` del resto, sobre una
     partición **agrupada por dominio** (fuera de muestra).

El paso 3 es el decisivo: la importancia de variables es un proxy imperfecto
(mide uso en el bosque, no contribución causal). Por eso la adopción de nuevos
pesos se condiciona a una mejora real de AUC fuera de muestra.

IMPORTANTE — circularidad de las etiquetas:
Las clases `proximo`/`distante` se asignan por umbral del IVR (P85). Recalibrar el
IVR contra etiquetas derivadas del IVR es circular en el nivel de la fórmula. Por
eso este módulo **no adopta** pesos automáticamente: reporta los números para que
el equipo decida. La vía que rompería la circularidad del todo sería etiquetar a
mano una submuestra.
"""
from __future__ import annotations

import pandas as pd

from .rf import (
    CARACTERISTICAS_SIN_IVR,
    FAMILIAS_SENALES,
    NOMBRES_CARACTERISTICAS,
    _nuevo_clf,
)

# Pesos actuales del IVR (config.toml → [ivr]) y candidatos a comparar.
PESOS_ACTUALES = {"sb": 0.40, "literal": 0.25, "ontologico": 0.20, "autoridad": 0.15}

PESOS_UNIFORME = {"sb": 0.25, "literal": 0.25, "ontologico": 0.25, "autoridad": 0.25}

PESOS_SOLO_TEXTUALES = {"sb": 0.60, "literal": 0.40, "ontologico": 0.0, "autoridad": 0.0}


def _preparar_X(dataset: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    X = dataset[columnas].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def importancias_por_familia(dataset: pd.DataFrame) -> dict:
    """Paso 1+2: entrena sin `ivr_*` y agrega las importancias por familia de señal.

    :return: dict con
        * `importancias_crudas`: {feature: importancia} de las 12 señales puras
        * `importancias_familia`: {sb|literal|ontologico|autoridad: suma}
        * `pesos_propuestos`: esas sumas normalizadas a 1
        * `accuracy`: exactitud de este modelo reducido (referencia)
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    X = _preparar_X(dataset, CARACTERISTICAS_SIN_IVR)
    le = LabelEncoder()
    y = le.fit_transform(dataset["veredicto"])

    if "dominio" in dataset.columns and dataset["dominio"].nunique() > 2:
        grupos = dataset["dominio"].astype(str).values
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                      .split(X, y, groups=grupos))
    else:                                    # sin dominio: partición estratificada simple
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(
            range(len(X)), test_size=0.25, random_state=42, stratify=y
        )

    clf = _nuevo_clf()
    clf.fit(X.iloc[tr], y[tr])
    acc = float(accuracy_score(y[te], clf.predict(X.iloc[te])))

    crudas = dict(zip(CARACTERISTICAS_SIN_IVR, clf.feature_importances_.tolist()))
    familia = {
        senal: float(sum(crudas.get(col, 0.0) for col in cols))
        for senal, cols in FAMILIAS_SENALES.items()
    }
    total = sum(familia.values()) or 1.0
    propuestos = {senal: valor / total for senal, valor in familia.items()}

    return {
        "importancias_crudas": dict(sorted(crudas.items(), key=lambda kv: kv[1], reverse=True)),
        "importancias_familia": familia,
        "pesos_propuestos": propuestos,
        "accuracy_sin_ivr": acc,
        "n_features": len(CARACTERISTICAS_SIN_IVR),
    }


def calcular_ivr_con_pesos(fila: dict, pesos: dict) -> float:
    """Recalcula el IVR de una fila usando OTROS pesos, sin volver a consultar el RAG.

    Aproxima las 4 señales desde las features agregadas disponibles en el CSV:

      SB ≈ sb_max,  L ≈ literal_max,  SO ≈ ontologico_max,  J ≈ autoridad_max

    Es una aproximación: el IVR original se calcula por evidencia y aquí solo
    tenemos agregados. Sirve para comparar la capacidad discriminante relativa
    de distintos juegos de pesos, no para sustituir `ivr.calcular_ivr`.
    """
    senales = {
        "sb": float(fila.get("sb_max", 0.0) or 0.0),
        "literal": float(fila.get("literal_max", 0.0) or 0.0),
        "ontologico": (
            None if not fila.get("so_disponible") else float(fila.get("ontologico_max", 0.0) or 0.0)
        ),
        "autoridad": float(fila.get("autoridad_max", 0.0) or 0.0),
    }
    num = 0.0
    den = 0.0
    for nombre, peso in pesos.items():
        valor = senales.get(nombre)
        if valor is None or peso <= 0:
            continue
        num += peso * min(max(valor, 0.0), 1.0)
        den += peso
    return num / den if den else 0.0


def auc_para_pesos(dataset: pd.DataFrame, pesos: dict) -> float:
    """AUC de separar `proximo` del resto usando el IVR recalculado con `pesos`.

    :return: AUC en [0, 1]; 0.5 = sin capacidad discriminante, 1.0 = perfecto.
    """
    from sklearn.metrics import roc_auc_score

    y_bin = (dataset["veredicto"] == "proximo").astype(int).values
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")

    scores = [
        calcular_ivr_con_pesos(fila.to_dict(), pesos)
        for _, fila in dataset.iterrows()
    ]
    return float(roc_auc_score(y_bin, scores))


def comparar_candidatos(dataset: pd.DataFrame, propuestos: dict | None = None) -> dict:
    """Paso 3: compara el AUC de cada juego de pesos sobre el MISMO dataset.

    :param propuestos: pesos derivados de las importancias; si es None se calculan.
    :return: {nombre_candidato: {"pesos": ..., "auc": ...}} ordenado por AUC desc.
    """
    candidatos = {
        "actual": dict(PESOS_ACTUALES),
        "uniforme": dict(PESOS_UNIFORME),
        "solo_textuales": dict(PESOS_SOLO_TEXTUALES),
    }
    if propuestos:
        candidatos["importancias"] = dict(propuestos)

    resultados = {}
    for nombre, pesos in candidatos.items():
        resultados[nombre] = {
            "pesos": pesos,
            "auc": auc_para_pesos(dataset, pesos),
        }
    return dict(sorted(resultados.items(), key=lambda kv: kv[1]["auc"], reverse=True))
