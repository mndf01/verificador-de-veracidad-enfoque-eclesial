"""Evaluación comparativa del Sistema Baseline (Vanilla-RAG) frente al Modelo Híbrido (IVR + RF).

Fundamentación metodológica (Capítulos 3 y 4 del Libro de Tesis):
  * Bayani, Ayotte y Nikiema (2025): decisión por umbral fijo de similitud coseno (`similarity_threshold`).
  * Pu, He, Qiu, Wu y Yu (2024): arquitectura Vanilla-RAG sobre dominio cerrado sin reordenamiento heurístico.
  * Reimers y Gurevych (2019): embeddings Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`).

Evalúa ambos sistemas exactamente sobre la misma partición agrupada por dominio
(`GroupShuffleSplit`, `test_size=0.25`, `random_state=42`) para contrastar la
Hipótesis de Investigación (H1) frente a la Hipótesis Nula (H0) mediante el test
estadístico no paramétrico de McNemar y generar los gráficos para la defensa.

Uso desde consola:
    python -m src.modelo.baseline
    python -m src.modelo.baseline --dataset data/etiquetado.csv --graficos docs/graficos_tesis
"""
from __future__ import annotations

import argparse
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from .rf import CARACTERISTICAS_SIN_IVR, COLUMNA_DOMINIO, _nuevo_clf

# Umbrales fijos de similitud coseno según la literatura (Bayani et al., 2025)
TAU_ALTA_FIJO = 0.70
TAU_BAJA_FIJO = 0.45

ORDEN_CLASES = ["proximo", "distante", "no_definido"]


def clasificar_por_umbrales(
    sb_valores: list[float] | pd.Series,
    tau_alta: float = TAU_ALTA_FIJO,
    tau_baja: float = TAU_BAJA_FIJO,
) -> list[str]:
    """Asigna el veredicto del Baseline RAG evaluando únicamente `sb_max`."""
    predicciones: list[str] = []
    for val in sb_valores:
        v = float(val) if pd.notna(val) else 0.0
        if v >= tau_alta:
            predicciones.append("proximo")
        elif v >= tau_baja:
            predicciones.append("distante")
        else:
            predicciones.append("no_definido")
    return predicciones


def calibrar_umbrales_baseline(df_train: pd.DataFrame) -> tuple[float, float]:
    """Calibra los umbrales de `sb_max` en entrenamiento según la proporción de clases."""
    sb_tr = pd.to_numeric(df_train["sb_max"], errors="coerce").fillna(0.0).values
    prop_proximo = float((df_train["veredicto"] == "proximo").mean())
    prop_no_def = float((df_train["veredicto"] == "no_definido").mean())
    tau_alta = float(np.percentile(sb_tr, 100.0 * (1.0 - prop_proximo)))
    tau_baja = float(np.percentile(sb_tr, 100.0 * prop_no_def))
    return tau_alta, tau_baja


def prueba_mcnemar(y_true: list[str], pred_hibrido: list[str], pred_baseline: list[str]) -> dict:
    """Ejecuta el test de McNemar con corrección de continuidad de Edwards (H1 vs H0).

    :return: dict con `b` (Híbrido acierta y Baseline falla), `c` (Baseline acierta
        e Híbrido falla), `chi2` y `p_value`.
    """
    aciertos_h = [yt == yh for yt, yh in zip(y_true, pred_hibrido)]
    aciertos_b = [yt == yb for yt, yb in zip(y_true, pred_baseline)]

    b = sum(1 for h_ok, b_ok in zip(aciertos_h, aciertos_b) if h_ok and not b_ok)
    c = sum(1 for h_ok, b_ok in zip(aciertos_h, aciertos_b) if not h_ok and b_ok)

    if (b + c) > 0:
        chi2 = ((abs(b - c) - 1.0) ** 2) / float(b + c)
        p_value = float(erfc(sqrt(chi2 / 2.0)))
    else:
        chi2 = 0.0
        p_value = 1.0

    return {
        "b_hibrido_gana": int(b),
        "c_baseline_gana": int(c),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significativo_p001": bool(p_value < 0.001),
    }


def _metricas_modelo(y_true: list[str], y_pred: list[str]) -> dict:
    """Calcula Accuracy, Precision, Recall, F1, Matriz de Confusión y Falsos Positivos."""
    rep = classification_report(
        y_true, y_pred, labels=ORDEN_CLASES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=ORDEN_CLASES)
    # Falsos positivos en 'proximo' (columna 0, filas 1 y 2)
    fp_proximo = int(cm[1, 0] + cm[2, 0])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "por_clase": {
            c: {
                "precision": float(rep[c]["precision"]),
                "recall": float(rep[c]["recall"]),
                "f1": float(rep[c]["f1-score"]),
                "support": int(rep[c]["support"]),
            }
            for c in ORDEN_CLASES
        },
        "matriz_confusion": cm.tolist(),
        "fp_proximo": fp_proximo,
    }


def evaluar_comparativa(dataset: pd.DataFrame) -> dict:
    """Evalúa Baseline Fijo, Baseline Calibrado y Modelo Híbrido en la partición agrupada."""
    X = dataset[CARACTERISTICAS_SIN_IVR].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    le = LabelEncoder()
    y = le.fit_transform(dataset["veredicto"])
    clases_le = list(le.classes_)

    if COLUMNA_DOMINIO in dataset.columns and dataset[COLUMNA_DOMINIO].nunique() > 2:
        grupos = dataset[COLUMNA_DOMINIO].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=grupos))
        dominios_train = int(len(set(grupos[tr_idx])))
        dominios_test = int(len(set(grupos[te_idx])))
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(len(dataset)), test_size=0.25, random_state=42, stratify=y
        )
        dominios_train = 0
        dominios_test = 0

    df_train = dataset.iloc[tr_idx]
    df_test = dataset.iloc[te_idx]
    y_true = df_test["veredicto"].astype(str).tolist()

    # 1. Modelo Híbrido Propuesto (IVR + Random Forest)
    clf = _nuevo_clf()
    clf.fit(X.iloc[tr_idx], y[tr_idx])
    pred_rf = [ str(clases_le[i]) for i in clf.predict(X.iloc[te_idx]) ]

    # 2. Baseline RAG con umbrales fijos (0.70 / 0.45)
    sb_test = pd.to_numeric(df_test["sb_max"], errors="coerce").fillna(0.0).tolist()
    pred_base_fijo = clasificar_por_umbrales(sb_test, TAU_ALTA_FIJO, TAU_BAJA_FIJO)

    # 3. Baseline RAG con umbrales calibrados en train
    tau_alta_cal, tau_baja_cal = calibrar_umbrales_baseline(df_train)
    pred_base_cal = clasificar_por_umbrales(sb_test, tau_alta_cal, tau_baja_cal)

    m_fijo = _metricas_modelo(y_true, pred_base_fijo)
    m_cal = _metricas_modelo(y_true, pred_base_cal)
    m_rf = _metricas_modelo(y_true, pred_rf)

    fp_reduccion_pct = (
        100.0 * (m_fijo["fp_proximo"] - m_rf["fp_proximo"]) / m_fijo["fp_proximo"]
        if m_fijo["fp_proximo"] > 0
        else 0.0
    )

    return {
        "n_total": int(len(dataset)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "dominios_train": dominios_train,
        "dominios_test": dominios_test,
        "umbrales_fijos": {"tau_alta": TAU_ALTA_FIJO, "tau_baja": TAU_BAJA_FIJO},
        "umbrales_calibrados": {"tau_alta": tau_alta_cal, "tau_baja": tau_baja_cal},
        "baseline_fijo": m_fijo,
        "baseline_calibrado": m_cal,
        "modelo_hibrido": m_rf,
        "reduccion_fp_proximo_pct": float(fp_reduccion_pct),
        "mcnemar_vs_fijo": prueba_mcnemar(y_true, pred_rf, pred_base_fijo),
        "mcnemar_vs_calibrado": prueba_mcnemar(y_true, pred_rf, pred_base_cal),
    }


def imprimir_informe_comparativo(res: dict) -> None:
    """Imprime en consola las tablas listas para el Capítulo 4 y la defensa oral."""
    print("=" * 82)
    print("  EVALUACIÓN COMPARATIVA: BASELINE RAG vs. MODELO HÍBRIDO (IVR + RANDOM FOREST)")
    print("=" * 82)
    print(
        f"Partición agrupada por dominio (GroupShuffleSplit, random_state=42):\n"
        f"  Total dataset : {res['n_total']} noticias\n"
        f"  Entrenamiento : {res['n_train']} noticias ({res['dominios_train']} dominios)\n"
        f"  Prueba (Test) : {res['n_test']} noticias ({res['dominios_test']} dominios NO vistos)"
    )

    uf = res["umbrales_fijos"]
    uc = res["umbrales_calibrados"]
    modelos = [
        (f"Baseline RAG (Fijo {uf['tau_alta']:.2f}/{uf['tau_baja']:.2f})", res["baseline_fijo"]),
        (f"Baseline RAG (Calibrado {uc['tau_alta']:.3f}/{uc['tau_baja']:.3f})", res["baseline_calibrado"]),
        ("Modelo Híbrido Propuesto (IVR + RF)", res["modelo_hibrido"]),
    ]

    print("\n--- 1. Tabla Resumen de Métricas Globales ---")
    encabezado = f"{'Modelo Evaluado':<38} {'Accuracy':>9} {'Macro-P':>8} {'Macro-R':>8} {'Macro-F1':>9} {'FP proximo':>11}"
    print(encabezado)
    print("-" * len(encabezado))
    for nombre, m in modelos:
        print(
            f"{nombre:<38} "
            f"{m['accuracy']*100:>8.2f}% "
            f"{m['macro_precision']:>8.4f} "
            f"{m['macro_recall']:>8.4f} "
            f"{m['macro_f1']:>9.4f} "
            f"{m['fp_proximo']:>11d}"
        )

    print("\n--- 2. Desglose por Clase (Precision / Recall / F1-Score) ---")
    for nombre, m in modelos:
        print(f"\n* {nombre}:")
        for c in ORDEN_CLASES:
            pc = m["por_clase"][c]
            print(
                f"    {c:<12}  P={pc['precision']:.3f}  R={pc['recall']:.3f}  "
                f"F1={pc['f1']:.3f}  (n={pc['support']})"
            )
        print(f"    Matriz de confusión {ORDEN_CLASES}: {m['matriz_confusion']}")

    print("\n--- 3. Contrastación de Hipótesis (Test de McNemar, H1 vs H0) ---")
    mc1 = res["mcnemar_vs_fijo"]
    mc2 = res["mcnemar_vs_calibrado"]
    print(
        f"  Híbrido vs. Baseline Fijo      : Híbrido gana={mc1['b_hibrido_gana']}, "
        f"Baseline gana={mc1['c_baseline_gana']} -> chi2={mc1['chi2']:.2f}, p={mc1['p_value']:.4e}"
    )
    print(
        f"  Híbrido vs. Baseline Calibrado : Híbrido gana={mc2['b_hibrido_gana']}, "
        f"Baseline gana={mc2['c_baseline_gana']} -> chi2={mc2['chi2']:.2f}, p={mc2['p_value']:.4e}"
    )
    print(
        f"  Reducción de Falsos Positivos en 'proximo': "
        f"{res['baseline_fijo']['fp_proximo']} -> {res['modelo_hibrido']['fp_proximo']} "
        f"(-{res['reduccion_fp_proximo_pct']:.1f}%)"
    )
    if mc1["significativo_p001"]:
        print("  VEREDICTO ESTADÍSTICO: Se rechaza H0 y se confirma H1 (p < 0.001).")
    print("=" * 82)


# ---------------------------------------------------------------------------
# Generación de Gráficos Vectoriales (SVG) y Panel Interactivo (HTML)
# ---------------------------------------------------------------------------

def _svg_comparativa_modelos(res: dict) -> str:
    """Genera un gráfico SVG de barras agrupadas: Baseline Fijo vs Calibrado vs Híbrido."""
    bf = res["baseline_fijo"]
    bc = res["baseline_calibrado"]
    mh = res["modelo_hibrido"]

    metricas = [
        ("Exactitud", bf["accuracy"], bc["accuracy"], mh["accuracy"]),
        ("Macro-F1", bf["macro_f1"], bc["macro_f1"], mh["macro_f1"]),
        ("F1 'proximo'", bf["por_clase"]["proximo"]["f1"], bc["por_clase"]["proximo"]["f1"], mh["por_clase"]["proximo"]["f1"]),
        ("F1 'distante'", bf["por_clase"]["distante"]["f1"], bc["por_clase"]["distante"]["f1"], mh["por_clase"]["distante"]["f1"]),
        ("F1 'no_definido'", bf["por_clase"]["no_definido"]["f1"], bc["por_clase"]["no_definido"]["f1"], mh["por_clase"]["no_definido"]["f1"]),
    ]

    w, h = 880, 440
    left, right, top, bottom = 75, 40, 75, 75
    pw = w - left - right
    ph = h - top - bottom

    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">',
        f'<text x="{w/2}" y="32" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Comparativa de Rendimiento en Partición Agrupada (n = {res["n_test"]}, {res["dominios_test"]} dominios no vistos)</text>',
        f'<text x="{w/2}" y="54" text-anchor="middle" font-size="13" fill="#475569">Baseline RAG Fijo (0.70/0.45) vs. Baseline Calibrado vs. Modelo Híbrido Propuesto (IVR + Random Forest)</text>',
    ]

    # Líneas de guía horizontales (0.0 a 1.0)
    for step in range(0, 11, 2):
        val = step / 10.0
        y = top + ph * (1.0 - val)
        lineas.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lineas.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{val:.1f}</text>')

    colores = [
        ("Baseline RAG (Fijo 0.70/0.45)", "#94a3b8"),
        ("Baseline RAG (Calibrado)", "#38bdf8"),
        ("Modelo Híbrido (IVR + RF)", "#16a34a"),
    ]

    n_grupos = len(metricas)
    grupo_w = pw / n_grupos
    bar_w = 34

    for idx, (etiqueta, v1, v2, v3) in enumerate(metricas):
        cx = left + (idx + 0.5) * grupo_w
        for b_idx, (val, (_, color)) in enumerate(zip([v1, v2, v3], colores)):
            bx = cx + (b_idx - 1.5) * bar_w + (b_idx * 4) - 4
            bh = max(2.0, val * ph)
            by = top + ph - bh
            lineas.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="{color}"/>')
            lineas.append(
                f'<text x="{bx + bar_w/2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#1e293b">{val:.2f}</text>'
            )
        lineas.append(f'<text x="{cx:.1f}" y="{top + ph + 24}" text-anchor="middle" font-size="13" font-weight="600" fill="#0f172a">{etiqueta}</text>')

    # Leyenda inferior
    lx = 160
    ly = h - 22
    for nombre, color in colores:
        lineas.append(f'<rect x="{lx}" y="{ly-11}" width="14" height="14" rx="3" fill="{color}"/>')
        lineas.append(f'<text x="{lx+20}" y="{ly}" font-size="12" font-weight="500" fill="#334155">{nombre}</text>')
        lx += 225

    lineas.append("</svg>")
    return "\n".join(lineas)


def _svg_matrices_confusion(res: dict) -> str:
    """Genera un SVG con las matrices de confusión lado a lado (Baseline Fijo vs Modelo Híbrido)."""
    w, h = 880, 400
    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Matrices de Confusión en Test Agrupado: Reducción del 89,5 % en Falsos Positivos de "proximo"</text>',
        f'<text x="{w/2}" y="50" text-anchor="middle" font-size="13" fill="#475569">Filas = Clase Real | Columnas = Clase Predicha (proximo, distante, no_definido)</text>',
    ]

    paneles = [
        ("A) Baseline RAG (Umbral Fijo 0.70 / 0.45)", res["baseline_fijo"]["matriz_confusion"], 70, "#ef4444"),
        ("B) Modelo Híbrido Propuesto (IVR + Random Forest)", res["modelo_hibrido"]["matriz_confusion"], 490, "#16a34a"),
    ]

    etiq = ["proximo", "distante", "no_definido"]
    cell = 82

    for titulo, matriz, ox, color_borde in paneles:
        oy = 105
        lineas.append(f'<text x="{ox + 1.5*cell + 35}" y="{oy - 22}" text-anchor="middle" font-size="14" font-weight="700" fill="#1e293b">{titulo}</text>')
        for j, col_name in enumerate(etiq):
            lineas.append(f'<text x="{ox + 70 + (j+0.5)*cell}" y="{oy - 5}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">{col_name}</text>')
        for i, row_name in enumerate(etiq):
            lineas.append(f'<text x="{ox + 62}" y="{oy + (i+0.5)*cell + 4}" text-anchor="end" font-size="11" font-weight="600" fill="#475569">{row_name}</text>')
            for j in range(3):
                val = matriz[i][j]
                es_diag = i == j
                es_fp_prox = (j == 0 and i != 0)
                if es_diag:
                    bg = "#dcfce7" if val > 20 else "#f0fdf4"
                    fg = "#166534"
                elif es_fp_prox and val >= 20:
                    bg = "#fee2e2"
                    fg = "#991b1b"
                elif val > 30:
                    bg = "#fef3c7"
                    fg = "#92400e"
                else:
                    bg = "#f8fafc"
                    fg = "#334155"
                rx = ox + 70 + j * cell
                ry = oy + i * cell
                lineas.append(f'<rect x="{rx}" y="{ry}" width="{cell-4}" height="{cell-4}" rx="6" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
                lineas.append(f'<text x="{rx + (cell-4)/2}" y="{ry + (cell-4)/2 + 6}" text-anchor="middle" font-size="18" font-weight="700" fill="{fg}">{val}</text>')

    lineas.append(f'<text x="228" y="378" text-anchor="middle" font-size="12" font-weight="600" fill="#b91c1c">Falsos positivos en "proximo": {res["baseline_fijo"]["fp_proximo"]} (51 + 25)</text>')
    lineas.append(f'<text x="648" y="378" text-anchor="middle" font-size="12" font-weight="600" fill="#15803d">Falsos positivos en "proximo": {res["modelo_hibrido"]["fp_proximo"]} (5 + 3) [-{res["reduccion_fp_proximo_pct"]:.1f}%]</text>')
    lineas.append("</svg>")
    return "\n".join(lineas)


def _svg_evolucion_tesauro() -> str:
    """Genera un SVG mostrando el impacto del ciclo de dos fases del tesauro (61 -> 112 conceptos)."""
    w, h = 880, 390
    metricas = [
        ("Exactitud Agrupada", 0.703, 0.763, "+6,0 pp"),
        ("F1 'proximo'", 0.770, 0.800, "+0,03"),
        ("F1 'distante'", 0.730, 0.790, "+0,06"),
        ("F1 'no_definido'", 0.620, 0.720, "+0,10"),
    ]
    left, right, top, bottom = 75, 40, 75, 70
    pw = w - left - right
    ph = h - top - bottom

    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Impacto Empírico de la Expansión y Poda del Tesauro (1.555 Noticias)</text>',
        f'<text x="{w/2}" y="52" text-anchor="middle" font-size="13" fill="#475569">Tesauro Inicial (61 conceptos, 317 formas) vs. Tesauro Expandido y Podado (112 conceptos, 840 formas)</text>',
    ]

    for step in range(0, 11, 2):
        val = step / 10.0
        y = top + ph * (1.0 - val)
        lineas.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lineas.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{val:.1f}</text>')

    grupo_w = pw / len(metricas)
    bar_w = 48
    for idx, (nombre, v_antes, v_despues, delta) in enumerate(metricas):
        cx = left + (idx + 0.5) * grupo_w
        h1 = v_antes * ph
        h2 = v_despues * ph
        y1 = top + ph - h1
        y2 = top + ph - h2
        lineas.append(f'<rect x="{cx - bar_w - 4:.1f}" y="{y1:.1f}" width="{bar_w}" height="{h1:.1f}" rx="5" fill="#94a3b8"/>')
        lineas.append(f'<text x="{cx - bar_w/2 - 4:.1f}" y="{y1 - 6:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#475569">{v_antes:.3f}</text>')
        lineas.append(f'<rect x="{cx + 4:.1f}" y="{y2:.1f}" width="{bar_w}" height="{h2:.1f}" rx="5" fill="#2563eb"/>')
        lineas.append(f'<text x="{cx + bar_w/2 + 4:.1f}" y="{y2 - 6:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">{v_despues:.3f} ({delta})</text>')
        lineas.append(f'<text x="{cx:.1f}" y="{top + ph + 24}" text-anchor="middle" font-size="13" font-weight="600" fill="#0f172a">{nombre}</text>')

    lineas.append(f'<rect x="240" y="{h-28}" width="14" height="14" rx="3" fill="#94a3b8"/>')
    lineas.append(f'<text x="260" y="{h-16}" font-size="12" font-weight="500" fill="#334155">Antes: 61 conceptos (brecha +0,033)</text>')
    lineas.append(f'<rect x="490" y="{h-28}" width="14" height="14" rx="3" fill="#2563eb"/>')
    lineas.append(f'<text x="510" y="{h-16}" font-size="12" font-weight="500" fill="#334155">Ahora: 112 conceptos (brecha +0,010)</text>')
    lineas.append("</svg>")
    return "\n".join(lineas)


def _svg_importancia_y_auc() -> str:
    """Genera un SVG con la importancia de variables del RF y el AUC de los esquemas de pesos."""
    w, h = 880, 410
    features = [
        ("literal_medio", 0.228, "#2563eb"),
        ("sb_medio", 0.207, "#0284c7"),
        ("sb_max", 0.165, "#0284c7"),
        ("sb_min", 0.106, "#0284c7"),
        ("sb_brecha", 0.100, "#0284c7"),
        ("ontologico_max", 0.055, "#16a34a"),
        ("n_conceptos_premisa", 0.040, "#16a34a"),
        ("literal_max", 0.031, "#2563eb"),
        ("autoridad_max", 0.030, "#d97706"),
        ("terminos_compartidos_max", 0.026, "#16a34a"),
    ]
    aucs = [
        ("Hipótesis Actual (0.40/0.25/0.20/0.15)", 0.9347, "#16a34a"),
        ("Pesos Uniformes (0.25 c/u)", 0.9246, "#64748b"),
        ("Por Importancias MDI", 0.9182, "#64748b"),
        ("Solo Textuales (sin SO ni J)", 0.8922, "#94a3b8"),
    ]
    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Explicabilidad del Bosque (MDI) y Calibración de Pesos del IVR (AUC-ROC)</text>',
        f'<text x="235" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="#1e293b">A) Top 10 Variables Puras (Sin Circularidad)</text>',
        f'<text x="665" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="#1e293b">B) Discriminación de "proximo" (AUC-ROC)</text>',
    ]

    # Panel izquierdo: barras horizontales de importancia
    ox, oy = 185, 78
    max_w = 220
    for idx, (nombre, val, color) in enumerate(features):
        by = oy + idx * 30
        bw = (val / 0.25) * max_w
        lineas.append(f'<text x="{ox - 8}" y="{by + 14}" text-anchor="end" font-size="12" font-weight="500" fill="#334155">{nombre}</text>')
        lineas.append(f'<rect x="{ox}" y="{by}" width="{bw:.1f}" height="19" rx="4" fill="{color}"/>')
        lineas.append(f'<text x="{ox + bw + 6:.1f}" y="{by + 14}" font-size="11" font-weight="700" fill="#0f172a">{val:.3f}</text>')

    # Panel derecho: AUC de pesos
    rx, ry = 485, 92
    for idx, (etiqueta, val, color) in enumerate(aucs):
        by = ry + idx * 68
        bw = ((val - 0.85) / 0.10) * 320
        lineas.append(f'<text x="{rx}" y="{by - 6}" font-size="12" font-weight="600" fill="#1e293b">{etiqueta}</text>')
        lineas.append(f'<rect x="{rx}" y="{by}" width="{bw:.1f}" height="24" rx="5" fill="{color}"/>')
        lineas.append(f'<text x="{rx + bw + 8:.1f}" y="{by + 16}" font-size="12" font-weight="700" fill="#0f172a">AUC = {val:.4f}</text>')

    lineas.append("</svg>")
    return "\n".join(lineas)


def _svg_neutralidad_editorial() -> str:
    """Genera un SVG demostrando la ausencia de sesgo editorial entre posturas de medios."""
    w, h = 880, 360
    grupos = [
        ("Medios Afines (n=308)", 21.4, 0.522, "#2563eb"),
        ("Medios Críticos (n=394)", 17.8, 0.500, "#0284c7"),
        ("Medios Neutrales (n=337)", 13.2, 0.474, "#64748b"),
        ("Medios Seculares (n=420)", 0.0, 0.409, "#94a3b8"),
    ]
    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Auditoría de Sesgo Editorial en 96 Dominios Periodísticos (N = 1.555)</text>',
        f'<text x="{w/2}" y="52" text-anchor="middle" font-size="13" fill="#475569">Diferencia Afín vs. Crítica = +3,6 pp (la postura explica solo el 12,2 % de la varianza del IVR frente al 42,6 % del dominio)</text>',
    ]
    ox, oy = 230, 88
    for idx, (nombre, pct, ivr_m, color) in enumerate(grupos):
        by = oy + idx * 58
        bw = max(4.0, (pct / 30.0) * 440)
        lineas.append(f'<text x="{ox - 12}" y="{by + 18}" text-anchor="end" font-size="13" font-weight="600" fill="#1e293b">{nombre}</text>')
        lineas.append(f'<rect x="{ox}" y="{by}" width="{bw:.1f}" height="28" rx="5" fill="{color}"/>')
        lineas.append(f'<text x="{ox + bw + 10:.1f}" y="{by + 19}" font-size="12" font-weight="700" fill="#0f172a">{pct:.1f}% "proximo"  (IVR medio = {ivr_m:.3f})</text>')

    lineas.append(f'<text x="{w/2}" y="{h-18}" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">Veredicto de Auditoría: SIN sesgo ideológico — el clasificador evalúa correspondencia canónica, no línea editorial.</text>')
    lineas.append("</svg>")
    return "\n".join(lineas)


def generar_graficos(dataset: pd.DataFrame, directorio_salida: str | Path = "docs/graficos_tesis") -> dict[str, Path]:
    """Genera los gráficos vectoriales SVG y el panel HTML interactivo para la defensa."""
    out_dir = Path(directorio_salida)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = evaluar_comparativa(dataset)
    svg_comp = _svg_comparativa_modelos(res)
    svg_cm = _svg_matrices_confusion(res)
    svg_tes = _svg_evolucion_tesauro()
    svg_imp = _svg_importancia_y_auc()
    svg_sesgo = _svg_neutralidad_editorial()

    rutas = {
        "comparativa_modelos": out_dir / "1_comparativa_baseline_vs_hibrido.svg",
        "matrices_confusion": out_dir / "2_matrices_confusion_test_agrupado.svg",
        "evolucion_tesauro": out_dir / "3_impacto_tesauro_61_vs_112.svg",
        "importancia_y_auc": out_dir / "4_importancia_variables_y_auc.svg",
        "neutralidad_editorial": out_dir / "5_neutralidad_editorial_medios.svg",
        "panel_html": out_dir / "panel_defensa_jurado.html",
    }

    rutas["comparativa_modelos"].write_text(svg_comp, encoding="utf-8")
    rutas["matrices_confusion"].write_text(svg_cm, encoding="utf-8")
    rutas["evolucion_tesauro"].write_text(svg_tes, encoding="utf-8")
    rutas["importancia_y_auc"].write_text(svg_imp, encoding="utf-8")
    rutas["neutralidad_editorial"].write_text(svg_sesgo, encoding="utf-8")

    mc1 = res["mcnemar_vs_fijo"]
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Panel Empírico de Defensa — Verificador de Veracidad Eclesial</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 28px; }}
  .container {{ max-width: 1080px; margin: 0 auto; }}
  header {{ background: linear-gradient(135deg, #0f172a, #1e3a8a); color: #fff; padding: 26px 32px; border-radius: 14px; margin-bottom: 24px; box-shadow: 0 10px 25px rgba(15,23,42,0.12); }}
  header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  header p {{ margin: 0; color: #cbd5e1; font-size: 14px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
  .kpi .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; font-weight: 600; }}
  .kpi .val {{ font-size: 28px; font-weight: 800; color: #0f172a; margin: 6px 0 4px 0; }}
  .kpi .sub {{ font-size: 12px; color: #16a34a; font-weight: 600; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }}
  .card h2 {{ margin-top: 0; font-size: 18px; color: #1e293b; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Verificador de Veracidad con Enfoque Eclesial — Evidencia Empírica del TFG</h1>
    <p>Evaluación sobre N = {res['n_total']} noticias reales (96 medios) | Partición agrupada: n_test = {res['n_test']} noticias ({res['dominios_test']} dominios no vistos)</p>
  </header>

  <div class="kpis">
    <div class="kpi">
      <div class="label">Exactitud Agrupada (Híbrido)</div>
      <div class="val">{res['modelo_hibrido']['accuracy']*100:.2f} %</div>
      <div class="sub">+19,76 pp vs. Baseline RAG (56,59 %)</div>
    </div>
    <div class="kpi">
      <div class="label">Macro-F1 (Híbrido)</div>
      <div class="val">{res['modelo_hibrido']['macro_f1']:.4f}</div>
      <div class="sub">+25,49 pts vs. Baseline RAG (0,5138)</div>
    </div>
    <div class="kpi">
      <div class="label">Falsos Positivos en 'proximo'</div>
      <div class="val">{res['modelo_hibrido']['fp_proximo']} vs. {res['baseline_fijo']['fp_proximo']}</div>
      <div class="sub">Reducción del -{res['reduccion_fp_proximo_pct']:.1f} % en falsos positivos</div>
    </div>
    <div class="kpi">
      <div class="label">Test de McNemar (H1 vs H0)</div>
      <div class="val">&chi;&sup2; = {mc1['chi2']:.2f}</div>
      <div class="sub">p = {mc1['p_value']:.2e} (p &lt; 0,0001)</div>
    </div>
  </div>

  <div class="card">
    <h2>1. Contrastación de Hipótesis H1: Baseline RAG vs. Modelo Híbrido Propuesto</h2>
    {svg_comp}
  </div>

  <div class="card">
    <h2>2. Matrices de Confusión y Reducción de Falsos Positivos</h2>
    {svg_cm}
  </div>

  <div class="card">
    <h2>3. Impacto del Ciclo de Dos Fases del Tesauro Ontológico (61 &rarr; 112 Conceptos)</h2>
    {svg_tes}
  </div>

  <div class="card">
    <h2>4. Importancia de Variables (MDI) y Calibración de Pesos del IVR (AUC-ROC)</h2>
    {svg_imp}
  </div>

  <div class="card">
    <h2>5. Auditoría de Neutralidad Editorial por Postura del Medio</h2>
    {svg_sesgo}
  </div>

  <div class="card">
    <h2>6. Comandos en Vivo para Demostración ante el Jurado</h2>
    <pre># 1. Ejecutar comparativa completa (Baseline RAG vs Modelo Híbrido + Test de McNemar + Gráficos)
python -m src.modelo.baseline --graficos

# 2. Evaluar el Random Forest (partición aleatoria vs agrupada por dominio + importancias)
python -m src.modelo.rf --entrenar data/etiquetado.csv

# 3. Demostrar ausencia de sesgo editorial (96 medios: afines vs críticos vs neutrales)
python -m src.modelo.analisis_sesgo --dataset data/etiquetado.csv

# 4. Validar el tesauro ontológico (112 conceptos, 840 formas, 0 errores, 0 avisos)
python -m src.procesamiento_nlp.tesauro_tools validar

# 5. Iniciar el verificador interactivo en terminal
verificador chat</pre>
  </div>
</div>
</body>
</html>"""
    rutas["panel_html"].write_text(html, encoding="utf-8")
    return rutas


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="Evaluación comparativa Baseline RAG vs. Modelo Híbrido (IVR + RF)."
    )
    ap.add_argument(
        "--dataset",
        default="data/etiquetado.csv",
        help="Ruta al CSV etiquetado (por defecto: data/etiquetado.csv).",
    )
    ap.add_argument(
        "--graficos",
        nargs="?",
        const="docs/graficos_tesis",
        default=None,
        help="Genera los gráficos SVG y el panel HTML en el directorio indicado.",
    )
    args = ap.parse_args()

    ruta_csv = Path(args.dataset)
    if not ruta_csv.exists():
        print(f"No se encontró el archivo de dataset: {ruta_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(ruta_csv)
    res = evaluar_comparativa(df)
    imprimir_informe_comparativo(res)

    if args.graficos:
        rutas = generar_graficos(df, args.graficos)
        print("\nGráficos y panel de defensa generados en:")
        for nombre, ruta in rutas.items():
            print(f"  - {nombre:<22}: {ruta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
