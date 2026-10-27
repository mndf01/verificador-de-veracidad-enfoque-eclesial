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

# Clases del veredicto (traducir al español cuando llegue a la interfaz).
CLASES_VEREDICTO = ["consistente", "inconsistente", "no_verificable"]


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Random Forest para el veredicto del IVR.")
    ap.add_argument("--entrenar", metavar="CSV", help="CSV etiquetado (características + veredicto) para entrenar.")
    ap.add_argument("--salida", metavar="PATH", default="data/modelo_rf.joblib", help="Dónde guardar el modelo.")
    ap.add_argument("--predecir", metavar="PREMISA", help="Evalúa una premisa: IVR -> vector -> RF -> veredicto.")
    args = ap.parse_args()

    if args.entrenar:
        df = pd.read_csv(args.entrenar)
        clf, le = entrena(df)
        cv = cross_val_score(clf, df[NOMBRES_CARACTERISTICAS], le.transform(df["veredicto"]) if le else df["veredicto"], cv=5)
        print(f"Cross-val (5 folds) exactitud media: {cv.mean():.3f} ± {cv.std():.3f}")
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