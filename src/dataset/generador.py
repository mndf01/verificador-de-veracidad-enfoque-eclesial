"""Generación del dataset de entrenamiento a partir de una base de noticias.

Flujo (usado por `verificador dataset-generar`):

    iglesia.db (SQLite: urls + categoria)
        └─► por URL: web_extractor (texto completo + metadatos)
             └─► ventana deslizante (~175 palabras, solape 50)
                  └─► IVR por ventana  →  features por ventana
                       └─► agregación (max / media / std)  →  16 features por noticia
                            └─► CSV para el Random Forest

Por qué ventana deslizante en vez de la noticia entera: las señales L (literal) y
SO (ontológica) del IVR dividen por el tamaño de la premisa. Con una noticia de 2000
palabras esa división colapsa a ~0.01 y las señales se anulan. Cada ventana, en
cambio, tiene un tamaño comparable al chunk de norma, así que las señales miden algo.

La agregación por `max` garantiza que si UNA ventana encuentra respaldo fuerte, se
refleje en el vector, en vez de diluirse en el promedio de 16 ventanas.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Parámetros por defecto de la ventana deslizante (palabras).
VENTANA_PALABRAS = 175
SOLAPE_PALABRAS = 50


# ----------------------------------------------------------------------------------------
# Lectura de la base de URLs
# ----------------------------------------------------------------------------------------

@dataclass
class UrlNoticia:
    id: int
    url: str
    categoria: str          # afin | critica | neutral (postura del MEDIO, no veracidad)
    titulo: str
    relevancia: int
    dominio: str


def leer_urls(ruta_db: str | Path, categorias: Iterable[str] | None = None) -> list[UrlNoticia]:
    """Lee las URLs de la base `iglesia.db` (tabla `urls`).

    :param ruta_db: ruta al archivo .db (SQLite).
    :param categorias: si se indica, filtra por esas categorías (afin/critica/neutral).
    :return: lista de UrlNoticia ordenada por relevancia descendente.
    """
    ruta = Path(ruta_db).expanduser().resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la base de datos: {ruta}")

    con = sqlite3.connect(str(ruta))
    try:
        sql = "SELECT id, url, categoria, COALESCE(titulo,''), COALESCE(relevancia,0), COALESCE(dominio,'') FROM urls"
        params: list = []
        if categorias:
            cat_list = list(categorias)
            sql += " WHERE categoria IN (" + ",".join("?" * len(cat_list)) + ")"
            params = cat_list
        sql += " ORDER BY relevancia DESC, id ASC"
        filas = con.execute(sql, params).fetchall()
    finally:
        con.close()

    return [UrlNoticia(*fila) for fila in filas]


def muestra_balanceada(urls: list[UrlNoticia], n_total: int = 30) -> list[UrlNoticia]:
    """Toma una muestra repartida a partes iguales entre las categorías presentes.

    Con 3 categorías y n_total=30 → 10 de cada una. Si alguna no llega, redistribuye.
    Se respeta el orden por relevancia dentro de cada categoría.
    """
    por_cat: dict[str, list[UrlNoticia]] = {}
    for u in urls:
        por_cat.setdefault(u.categoria, []).append(u)

    if not por_cat:
        return []

    n_cat = len(por_cat)
    cuota = max(1, n_total // n_cat)
    muestra: list[UrlNoticia] = []
    for cat, lista in por_cat.items():
        muestra.extend(lista[:cuota])

    # Si sobró o faltó (categorías pequeñas), se completa/recorta hasta n_total.
    if len(muestra) < n_total:
        vistos = {(u.id) for u in muestra}
        for u in urls:
            if u.id not in vistos:
                muestra.append(u)
                vistos.add(u.id)
                if len(muestra) >= n_total:
                    break
    return muestra[:n_total]


# ----------------------------------------------------------------------------------------
# Ventana deslizante
# ----------------------------------------------------------------------------------------

def ventanas(texto: str, tam: int = VENTANA_PALABRAS, solape: int = SOLAPE_PALABRAS) -> list[str]:
    """Trocea un texto en ventanas solapadas de `tam` palabras.

    El solape evita perder el contexto de las frases que caen en un corte: cada
    fragmento aparece en dos ventanas consecutivas junto a su contexto vecino.

    :return: lista de ventanas (str). Si el texto es más corto que `tam`, una sola.
    """
    if not texto or not texto.strip():
        return []
    palabras = texto.split()
    if not palabras:
        return []
    if len(palabras) <= tam:
        return [" ".join(palabras)]

    paso = max(1, tam - solape)
    return [" ".join(palabras[i:i + tam]) for i in range(0, len(palabras), paso)]


# ----------------------------------------------------------------------------------------
# Extracción de una noticia
# ----------------------------------------------------------------------------------------

@dataclass
class NoticiaProcesada:
    """Resultado del paso de una noticia por el IVR con ventanas."""
    url: str
    categoria: str
    titulo: str
    ok: bool = False
    motivo: str = ""
    n_ventanas: int = 0
    n_ventanas_con_senal: int = 0
    # métricas de diagnóstico (para el análisis de la muestra)
    ivr_max: float = 0.0
    ivr_top3: float = 0.0
    ivr_medio_ventanas: float = 0.0
    ventanas_detalle: list[dict] = field(default_factory=list)


def procesar_noticia(
    item: UrlNoticia,
    *,
    extraer_contenido: Callable[[str], str | None] | None = None,
    calcular_ivr: Callable[[str], dict] | None = None,
    tam_ventana: int = VENTANA_PALABRAS,
    solape: int = SOLAPE_PALABRAS,
    umbral_senal: float = 0.15,
) -> NoticiaProcesada:
    """Descarga la noticia, la trocea en ventanas y corre el IVR en cada una.

    No construye el vector de características final (eso lo hace `agregar_features`);
    solo recoge el detalle por ventana, que sirve tanto para diagnóstico como para
    la agregación posterior.
    """
    res = NoticiaProcesada(url=item.url, categoria=item.categoria, titulo=item.titulo)

    if extraer_contenido is None:
        from src.web_extractor import extraer_contenido_url as extraer_contenido
    if calcular_ivr is None:
        from src.vectorizacion.ivr import calcular_ivr as calcular_ivr

    texto = extraer_contenido(item.url)
    if not texto:
        res.motivo = "no se pudo extraer contenido"
        return res

    vs = ventanas(texto, tam_ventana, solape)
    if not vs:
        res.motivo = "texto sin palabras"
        return res
    res.n_ventanas = len(vs)

    ivrs: list[float] = []
    for i, v in enumerate(vs):
        informe = calcular_ivr(v)
        if not informe or not informe.get("ok"):
            res.ventanas_detalle.append({"i": i, "ivr": 0.0, "ok": False})
            ivrs.append(0.0)
            continue
        val = float(informe.get("ivr_max", 0.0))
        ivrs.append(val)
        if val >= umbral_senal:
            res.n_ventanas_con_senal += 1
        res.ventanas_detalle.append({
            "i": i,
            "ivr": val,
            "ok": True,
            "n_evidencias": informe.get("n_evidencias", 0),
        })

    if not ivrs:
        res.motivo = "ninguna ventana produjo informe"
        return res

    ordenados = sorted(ivrs, reverse=True)
    res.ok = True
    res.ivr_max = ordenados[0]
    res.ivr_top3 = sum(ordenados[:3]) / min(3, len(ordenados))
    res.ivr_medio_ventanas = sum(ivrs) / len(ivrs)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Diagnóstico rápido del dataset de noticias.")
    ap.add_argument("--db", required=True, help="Ruta a iglesia.db")
    ap.add_argument("--n", type=int, default=30, help="Tamaño de la muestra")
    args = ap.parse_args()

    urls = leer_urls(args.db)
    print(f"URLs en la base: {len(urls)}")
    for u in muestra_balanceada(urls, args.n)[:5]:
        print(f"  [{u.categoria}] {u.url[:80]}")
