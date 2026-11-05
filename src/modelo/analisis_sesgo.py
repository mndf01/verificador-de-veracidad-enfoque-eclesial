"""Análisis de sesgo por dominio del verificador.

Responde a la pregunta: ¿el sistema se apoya en la POSTURA EDITORIAL del medio
(afín / crítica / neutral) o en el CONTENIDO de la noticia?

Si dependiera de la postura, un medio afín obtendría siempre tasas de `proximo`
altas y uno crítico siempre bajas. Los datos muestran lo contrario.

Uso:
    .venv/bin/python -m src.modelo.analisis_sesgo --dataset data/etiquetado.csv
    .venv/bin/python -m src.modelo.analisis_sesgo --dataset data/etiquetado.csv --min-noticias 10

Resultado de referencia (1620 noticias, 98 dominios, 2026-11-03):
    - afin    20,2 % de `proximo`  |  critica 20,0 %   -> sin sesgo editorial
    - Varianza del IVR explicada por POSTURA: 10,6 %
    - Varianza del IVR explicada por DOMINIO: 43,0 %
    - vaticannews.va (afín) 54,5 %  vs  catholicregister.org (afín) 5,0 %
    - religiondigital.org (crítica) 52,5 %  vs  secular.org (crítica) 0,0 %
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

POSTURAS = ("afin", "critica", "neutral")


def tabla_por_dominio(df: pd.DataFrame, min_noticias: int = 10) -> pd.DataFrame:
    """Tasa de `proximo` y IVR medio por (dominio, postura)."""
    g = (
        df.groupby(["dominio", "categoria_medio"])
        .agg(
            n=("veredicto", "size"),
            proximo=("veredicto", lambda s: int((s == "proximo").sum())),
            ivr_max=("ivr_max", "mean"),
        )
        .reset_index()
    )
    g["pct_proximo"] = 100 * g["proximo"] / g["n"]
    return g[g["n"] >= min_noticias].sort_values("pct_proximo", ascending=False)


def varianza_explicada(df: pd.DataFrame, columna: str) -> float:
    """Fracción (%) de la varianza de `ivr_max` explicada por agrupar según `columna`.

    Es una eta-cuadrada: varianza de las medias de grupo / varianza total.
    """
    total = df["ivr_max"].var()
    if not total or pd.isna(total):
        return 0.0
    medias = df.groupby(columna)["ivr_max"].transform("mean")
    return float(medias.var() / total * 100)


def analizar(ruta: str | Path, min_noticias: int = 10) -> dict:
    """Ejecuta el análisis completo y devuelve los números clave."""
    df = pd.read_csv(ruta)
    eclesiales = df[df["categoria_medio"].isin(POSTURAS)]

    tabla = tabla_por_dominio(df, min_noticias)
    por_postura = {}
    for postura in POSTURAS:
        sub = tabla[tabla["categoria_medio"] == postura]
        if sub.empty:
            continue
        por_postura[postura] = {
            "dominios": int(len(sub)),
            "noticias": int(sub["n"].sum()),
            "pct_medio": float(sub["pct_proximo"].mean()),
            "pct_mediana": float(sub["pct_proximo"].median()),
            "pct_min": float(sub["pct_proximo"].min()),
            "pct_max": float(sub["pct_proximo"].max()),
            "ivr_medio": float(sub["ivr_max"].mean()),
        }

    return {
        "n_total": int(len(df)),
        "n_dominios": int(df["dominio"].nunique()),
        "tabla": tabla,
        "por_postura": por_postura,
        "varianza_postura": varianza_explicada(eclesiales, "categoria_medio"),
        "varianza_dominio": varianza_explicada(eclesiales, "dominio"),
        "n_eclesiales": int(len(eclesiales)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analiza el sesgo por dominio del verificador.")
    ap.add_argument("--dataset", default="data/etiquetado.csv")
    ap.add_argument("--min-noticias", type=int, default=10,
                    help="Ignora dominios con menos noticias que este umbral.")
    args = ap.parse_args(argv)

    r = analizar(args.dataset, args.min_noticias)

    print(f"\n=== SESGO POR DOMINIO ===")
    print(f"noticias: {r['n_total']} | dominios: {r['n_dominios']}")
    print(f"(dominios con >= {args.min_noticias} noticias en el desglose)\n")

    print("--- Tasa de 'proximo' por postura ---")
    print(f"{'postura':<10} {'dominios':>8} {'noticias':>8} {'% medio':>8} {'mediana':>8} {'rango':>14} {'IVR':>7}")
    print("-" * 70)
    for postura, d in r["por_postura"].items():
        rango = f"{d['pct_min']:.0f}-{d['pct_max']:.0f}%"
        print(f"{postura:<10} {d['dominios']:>8} {d['noticias']:>8} "
              f"{d['pct_medio']:>7.1f}% {d['pct_mediana']:>7.1f}% {rango:>14} {d['ivr_medio']:>7.3f}")

    print(f"\n--- Varianza del IVR explicada ({r['n_eclesiales']} noticias eclesiales) ---")
    print(f"  por POSTURA (3 grupos)   : {r['varianza_postura']:5.1f} %")
    print(f"  por DOMINIO ({r['n_dominios']} grupos): {r['varianza_dominio']:5.1f} %")

    afin = r["por_postura"].get("afin", {}).get("pct_medio", 0)
    crit = r["por_postura"].get("critica", {}).get("pct_medio", 0)
    print(f"\n--- Veredicto ---")
    print(f"  diferencia afin - critica: {afin - crit:+.1f} puntos porcentuales")
    if abs(afin - crit) < 5:
        print("  -> SIN sesgo editorial: el sistema no favorece a los medios afines.")
    else:
        print("  -> ATENCION: posible sesgo editorial, revisar.")

    print(f"\n--- Desglose por dominio (top 8 y bottom 8) ---")
    t = r["tabla"]
    print(f"{'dominio':<34} {'postura':<9} {'n':>3} {'% prox':>7} {'IVR':>7}")
    print("-" * 64)
    for _, fila in pd.concat([t.head(8), t.tail(8)]).iterrows():
        print(f"{fila.dominio:<34} {fila.categoria_medio:<9} {fila.n:>3} "
              f"{fila.pct_proximo:>6.1f}% {fila.ivr_max:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
