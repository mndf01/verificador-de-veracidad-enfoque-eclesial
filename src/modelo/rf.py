"""Random Forest para la clasificación del veredicto (capa de decisión).

El Random Forest consume el VECTOR DE CARACTERÍSTICAS que produce el IVR
(`src/vectorizacion/ivr.vector_caracteristicas()`), no los textos ni los dicts
anidados del informe. Así el modelo está desacoplado de Chroma/SBERT igual que
el propio IVR lo está de la base vectorial.

Arquitectura (igual que el resto del proyecto):

    informe = ivr.calcular_ivr(premisa)
    X_txt   = ivr.vector_caracteristicas(informe)   # dict nombre -> float
    X_row   = [X_txt[c] for c in NOMBRES_CARACTERISTICAS]  # orden estable
    y = rf.predecir(X_row)                           # -> "consistente" | "inconsistente" | "no_verificable"

El modelo se entrena con un CSV etiquetado a mano (calibración del TFG) que
relaciona una premisa con su veredicto esperado. Cada fila = una premisa real
etiquetada; las columnas son las características del IVR.

Uso:
    # Entrenar y guardar el modelo en disco (una vez, tras tener el dataset)
    python -m src.modelo.rf --entrenar data/etiquetado.csv --salida data/modelo_rf.joblib

    # Predecir desde consola
    python -m src.modelo.rf --predecir "El bautismo esta prohibido"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Orden FIJO de las columnas. Debe coincidir EXACTAMENTE con las claves que
# devuelve ivr.vector_caracteristicas() (ver test_ivr.TestInformeIvr).
NOMBRES_CARACTERISTICAS = [
    "n_evidencias", "n_conceptos_premisa",
    "ivr_max", "ivr_medio", "ivr_min", "ivr_brecha",
    "sb_max", "sb_medio", "sb_min", "sb_brecha",
    "literal_max", "literal_medio",
    "ontologico_max", "so_disponible",
    "terminos_compartidos_max", "autoridad_max",
]

# Clases del veredicto (similitud con el corpus, no verdad absoluta).
CLASES_VEREDICTO = ["proximo", "distante", "no_definido"]

# Columna que identifica el medio. Se usa para AGRUPAR en la evaluación, nunca
# como feature: si el modelo viera el dominio memorizaría medios en vez de aprender
# del contenido (data leakage).
COLUMNA_DOMINIO = "dominio"


def _X_de_informe(informe: dict) -> list[float]:
    """Vector de floats (orden estable) a partir del informe de calcular_ivr().

    Usa vector_caracteristicas() del IVR para no duplicar la lógica de agregación
    y para garantizar que las ausencias (SO no disponible, etc.) se marquen igual
    aquí y en el dataset de entrenamiento. El import es local para no acoplar el
    top del módulo sklearn al resto del pipeline (igual que hace sesion.py).
    """
    from src.vectorizacion.ivr import vector_caracteristicas

    caracteristicas = vector_caracteristicas(informe)
    return [float(caracteristicas.get(nombre, 0.0)) for nombre in NOMBRES_CARACTERISTICAS]


def entrena(dataset: pd.DataFrame) -> tuple[RandomForestClassifier, LabelEncoder | None]:
    """Entrena el RF con un CSV que tiene las características del IVR + columna `veredicto`.

    :param dataset: DataFrame con las 16 columnas de NOMBRES_CARACTERISTICAS y
        una columna categórica `veredicto`.
    :return: (clf, le). le es None si el veredicto ya llega codificado.
    """
    X = dataset[NOMBRES_CARACTERISTICAS].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    y = dataset["veredicto"]
    le = None
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf, le


def predecir(clf: RandomForestClassifier, le: LabelEncoder | None, X_row: list[float]) -> dict:
    """Clasifica el vector de características y devuelve el veredicto + confianza.

    :return: {"clase": ..., "probabilidad": float en [0,1], "probabilidades": {clase: p}}.
    """
    # Se envuelve en un DataFrame con los nombres de columna del entrenamiento
    # para que scikit-learn no emita warnings de "X does not have valid feature names".
    X = pd.DataFrame([X_row], columns=NOMBRES_CARACTERISTICAS)
    probs = clf.predict_proba(X)[0]
    indice = int(clf.predict(X)[0])
    clase = le.inverse_transform([indice])[0] if le is not None else str(indice)
    return {
        "clase": clase,
        "probabilidad": float(probs[indice]),
        "probabilidades": {
            (le.inverse_transform([i])[0] if le is not None else str(i)): float(p)
            for i, p in enumerate(probs)
        },
    }


def evaluar(dataset: pd.DataFrame, columna_dominio: str = COLUMNA_DOMINIO) -> dict:
    """Evalúa el modelo con DOS particiones para medir generalización real.

    Por qué dos:
      * **Aleatoria** (``train_test_split`` estratificado): mide el rendimiento
        cuando el mismo medio puede aparecer en train y en test. Si el modelo se
        apoya en el medio, aquí infla la exactitud.
      * **Agrupada por dominio** (``GroupShuffleSplit``): los dominios del test NO
        aparecen en train. Refleja el uso real (un medio nuevo que el sistema
        nunca vio).

    La diferencia entre ambas cuantifica cuánto del rendimiento es memorización
    del medio y cuánto generalización de contenido. Es un resultado de tesis.

    :return: dict con métricas de ambos escenarios + importancias de variables.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    X = dataset[NOMBRES_CARACTERISTICAS].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    le = LabelEncoder()
    y = le.fit_transform(dataset["veredicto"])
    clases = list(le.classes_)

    resultado: dict = {"n_total": len(dataset), "clases": clases, "dominios_usados": 0}

    # --- 1) Partición aleatoria estratificada ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    clf_r = _nuevo_clf()
    clf_r.fit(X_tr, y_tr)
    pred_r = clf_r.predict(X_te)
    resultado["aleatorio"] = {
        "accuracy": float(accuracy_score(y_te, pred_r)),
        "n_test": int(len(y_te)),
        "reporte": classification_report(y_te, pred_r, target_names=clases,
                                         zero_division=0, output_dict=True),
        "matriz": confusion_matrix(y_te, pred_r).tolist(),
    }

    # --- 2) Partición agrupada por dominio (medios no vistos en test) ---
    tiene_dominio = columna_dominio in dataset.columns
    if tiene_dominio and dataset[columna_dominio].nunique() > 2:
        grupos = dataset[columna_dominio].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=grupos))
        clf_g = _nuevo_clf()
        clf_g.fit(X.iloc[tr_idx], y[tr_idx])
        pred_g = clf_g.predict(X.iloc[te_idx])
        resultado["dominios_usados"] = int(len(set(grupos)))
        resultado["agrupado"] = {
            "accuracy": float(accuracy_score(y[te_idx], pred_g)),
            "n_test": int(len(te_idx)),
            "dominios_train": int(len(set(grupos[tr_idx]))),
            "dominios_test": int(len(set(grupos[te_idx]))),
            "reporte": classification_report(y[te_idx], pred_g, target_names=clases,
                                             zero_division=0, output_dict=True),
            "matriz": confusion_matrix(y[te_idx], pred_g).tolist(),
        }
        resultado["importancias"] = dict(
            sorted(zip(NOMBRES_CARACTERISTICAS, clf_g.feature_importances_),
                   key=lambda kv: kv[1], reverse=True)
        )
    else:
        resultado["agrupado"] = None
        resultado["importancias"] = dict(
            sorted(zip(NOMBRES_CARACTERISTICAS, clf_r.feature_importances_),
                   key=lambda kv: kv[1], reverse=True)
        )

    return resultado


def _nuevo_clf() -> RandomForestClassifier:
    """Instancia un RF con los hiperparámetros del proyecto."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def guardar(clf, le, ruta: str | Path) -> None:
    """Persiste el modelo (joblib) para no reentrenar en cada predicción."""
    import joblib
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "le": le, "caracteristicas": NOMBRES_CARACTERISTICAS}, ruta)
    print(f"Modelo guardado en: {ruta}")


def cargar(ruta: str | Path):
    """Carga el modelo persistido con guardar()."""
    import joblib
    return joblib.load(Path(ruta))


def _imprimir_evaluacion(m: dict) -> None:
    """Presenta en consola el resultado de `evaluar()`."""
    ale = m["aleatorio"]
    print(f"--- Partición ALEATORIA (mismo medio puede estar en train y test) ---")
    print(f"  exactitud: {ale['accuracy']:.3f}   (n_test={ale['n_test']})")
    for clase in m["clases"]:
        r = ale["reporte"].get(clase, {})
        if r:
            print(f"    {clase:<12} precision={r['precision']:.2f} recall={r['recall']:.2f} f1={r['f1-score']:.2f} n={int(r['support'])}")

    agr = m.get("agrupado")
    if agr:
        print(f"\n--- Partición AGRUPADA por dominio (medios NO vistos en test) ---")
        print(f"  exactitud: {agr['accuracy']:.3f}   (n_test={agr['n_test']}, "
              f"dominios train={agr['dominios_train']} / test={agr['dominios_test']})")
        for clase in m["clases"]:
            r = agr["reporte"].get(clase, {})
            if r:
                print(f"    {clase:<12} precision={r['precision']:.2f} recall={r['recall']:.2f} f1={r['f1-score']:.2f} n={int(r['support'])}")
        print(f"\n  Matriz de confusión (filas=real, columnas=predicho): {m['clases']}")
        for nombre, fila in zip(m["clases"], agr["matriz"]):
            print(f"    {nombre:<12} {fila}")

        diferencia = ale["accuracy"] - agr["accuracy"]
        print(f"\n  Brecha aleatorio - agrupado: {diferencia:+.3f}")
        print("  (brecha grande = el modelo se apoya en el medio; brecha pequeña = generaliza)")
    else:
        print("\n  (sin columna de dominio: no se pudo evaluar con partición agrupada)")

    print("\n--- Importancia de variables ---")
    for nombre, valor in list(m["importancias"].items())[:10]:
        barra = "#" * int(valor * 100)
        print(f"  {nombre:<26} {valor:.3f}  {barra}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Random Forest para el veredicto del IVR.")
    ap.add_argument("--entrenar", metavar="CSV", help="CSV etiquetado (características + veredicto) para entrenar.")
    ap.add_argument("--salida", metavar="PATH", default="data/modelo_rf.joblib", help="Dónde guardar el modelo.")
    ap.add_argument("--predecir", metavar="PREMISA", help="Evalúa una premisa: IVR -> vector -> RF -> veredicto.")
    args = ap.parse_args()

    if args.entrenar:
        df = pd.read_csv(args.entrenar)

        # --- Evaluación honesta (aleatoria vs agrupada por dominio) ---
        print(f"\n=== EVALUACION: {len(df)} filas ===")
        print(f"clases: {dict(df['veredicto'].value_counts())}")
        if COLUMNA_DOMINIO in df.columns:
            print(f"dominios: {df[COLUMNA_DOMINIO].nunique()}")
        print()

        metricas = evaluar(df)
        _imprimir_evaluacion(metricas)

        # --- Entrenamiento final con TODOS los datos ---
        print("\n=== Entrenamiento final (todos los datos) ===")
        clf, le = entrena(df)
        guardar(clf, le, args.salida)
        return 0

    if args.predecir:
        from src.vectorizacion import ivr
        informe = ivr.calcular_ivr(args.predecir)
        if not informe["ok"]:
            print(f"Sin evidencias: {informe['motivo']}")
            return 1
        modelo = cargar(args.salida)
        clf, le = modelo["clf"], modelo["le"]
        resultado = predecir(clf, le, _X_de_informe(informe))
        print(f"\nPremisa: {args.predecir}")
        print(f"IVR max = {informe['ivr_max']:.3f}")
        print(f"Veredicto: {resultado['clase']} (prob {resultado['probabilidad']:.3f})")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())