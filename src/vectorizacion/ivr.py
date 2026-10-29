"""IVR — Índice de Veracidad Relativa (capa lógica del TFG).

Heurística propia que puntúa cuánto respalda (o contradice) cada norma
recuperada por el RAG a una premisa periodística. Combina cuatro señales,
todas normalizadas a [0, 1]:

    SB = similitud semántica     -> coseno SBERT que ya devuelve `consultar()`
    L  = coincidencia literal    -> n-gramas de contenido compartidos
    SO = coincidencia ontológica -> categoría del tesauro compartida
    J  = peso de autoridad       -> nivel_autoridad del chunk (0.33 .. 1.0)

    IVR = (w_SB·SB + w_L·L + w_SO·SO + w_J·J) / (w_SB + w_L + w_SO + w_J)

SO es la ÚNICA señal opcional: si la premisa no activa ningún concepto del
tesauro no hay nada que comparar, así que se excluye de la suma y el divisor
se renormaliza (por eso el denominador es la suma de los pesos realmente
usados, y no 1). Todos los pesos viven en `config.toml` → [ivr].

El IVR se calcula por evidencia y también agregado sobre el conjunto
recuperado (máximo, media, mínimo, brechas...). Ese agregado es el vector de
características que consumirá el Random Forest de `src/modelo/` (Mes 3): ver
`vector_caracteristicas()`.

Este módulo NO toca ChromaDB directamente (no abre colecciones ni carga
SBERT): consume el contrato de `src.base_datos.chromaDB.consultar()` (o de
`SesionExpansion.consultar_combinado()` cuando se activa la expansión dinámica).

Uso:
    .venv/bin/python -m src.vectorizacion.ivr "El bautismo esta prohibido"
"""
from __future__ import annotations

import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import snowballstemmer

# Permite ejecutarlo como módulo (-m src.vectorizacion.ivr) o como script suelto.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import CONFIG_PATH, obtener_configuraciones
from src.base_datos.chromaDB import consultar
from src.procesamiento_nlp.tesauro_tools import (
    PALABRAS_VACIAS,
    RUTA_TESAURO,
    cargar_tesauro,
    normalizar,
)

# RUTA_TESAURO es relativa al cwd; se ancla a la raíz del proyecto (donde vive
# config.py) para que el IVR funcione aunque se lance desde otra carpeta.
RUTA_TESAURO_ABSOLUTA = CONFIG_PATH.parent / RUTA_TESAURO

# Métricas disponibles para la señal literal (ver `similitud_literal`).
METRICAS_LITERALES = ("contencion", "dice")

_TOKENS = re.compile(r"[a-z0-9]+")

_STEMMER = snowballstemmer.stemmer("spanish")


def _stem_palabra(palabra: str) -> str:
    """Raíz morfológica en español usando Snowball Stemmer."""
    p = palabra.lower()
    if p == "papado":
        return "papado"
    return _STEMMER.stemWord(p)


def _stem_tokens(tokens: list[str]) -> list[str]:
    """Aplica stemming a una lista de tokens."""
    return [_stem_palabra(t) for t in tokens]


def _stem_frase(texto: str) -> str:
    """Normaliza un texto y convierte cada palabra a su raíz morfológica (stem)."""
    return " ".join(_stem_tokens(_TOKENS.findall(normalizar(texto))))


# ----------------------------------------------------------------------------------------
# Índice del tesauro (ontología) — señal SO
# ----------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _indice_tesauro(ruta: Path = RUTA_TESAURO_ABSOLUTA) -> tuple[dict[str, str], dict[str, str]]:
    """Aplana el tesauro en (forma con stemming -> id_concepto, id_concepto -> categoría).

    Aplica el mismo stemmer que la señal literal (Opción 2A de auditoría) para
    detectar plurales, conjugaciones verbales y variaciones de género sin tocar
    el JSON. Se ordena de frases más largas a más cortas para dar prioridad a
    términos específicos (ej. 'obispo de roma' antes que 'obispo').

    :return: (forma_stem_a_concepto, categoria_de). NO mutar: están cacheados.
    """
    conceptos = cargar_tesauro(ruta).get("conceptos", {})
    forma_a_concepto: dict[str, str] = {}
    categoria_de: dict[str, str] = {}

    for id_concepto, concepto in conceptos.items():
        categoria_de[id_concepto] = concepto.get("categoria", "")
        formas = [concepto.get("termino_canonico", "")] + list(concepto.get("variaciones_lexicas", []))
        for forma in formas:
            clave = _stem_frase(str(forma))
            if clave:
                forma_a_concepto.setdefault(clave, id_concepto)

    forma_ordenada = dict(sorted(forma_a_concepto.items(), key=lambda kv: len(kv[0].split()), reverse=True))
    return forma_ordenada, categoria_de


@lru_cache(maxsize=None)
def _patron(forma_stem: str) -> re.Pattern:
    """Forma con stemming -> regex con límites de palabra."""
    return re.compile(r"(?<!\w)" + re.escape(forma_stem) + r"(?!\w)")


def detectar_conceptos(texto: str) -> dict[str, str]:
    """Conceptos del tesauro presentes en el texto (con stemming morfológico).

    :param texto: texto libre (se normaliza y aplica stemming internamente).
    :return: {id_concepto: categoría}. Vacío si el texto no activa ninguno.
    """
    if not texto:
        return {}

    norm_stem = _stem_frase(texto)
    if not norm_stem:
        return {}

    forma_a_concepto, categoria_de = _indice_tesauro()
    encontrados: dict[str, str] = {}

    for forma_stem, id_concepto in forma_a_concepto.items():
        if id_concepto in encontrados or forma_stem not in norm_stem:
            continue
        if _patron(forma_stem).search(norm_stem):
            encontrados[id_concepto] = categoria_de[id_concepto]

    return encontrados


def coincidencia_ontologica(
    conceptos_premisa: dict[str, str],
    categoria: str,
    conceptos_norma: dict[str, str] | None = None,
) -> float | None:
    """Señal SO simétrica (Opción 3A de auditoría).

    Combina dos dimensiones ontológicas:
      1. Fracción de conceptos de la premisa respaldados por la categoría de la
         norma (o presentes directamente en los conceptos de la norma). Cuando la
         norma proviene de expansión dinámica con etiqueta genérica (p. ej.
         "Normativa Externa (carga dinámica)"), se infieren además las categorías
         de los conceptos detectados en el propio fragmento.
      2. Fracción de conceptos de la norma que coinciden con los de la premisa
         (|P ∩ N| / |N|). Si la norma no activa ningún concepto en su texto
         (conceptos_norma == {}), este segundo término vale 0.0, evitando que
         normas genéricas sin conceptos saturen a 1.0.

    Si `conceptos_norma is None` (llamada simple sin texto de norma), devuelve
    únicamente la primera fracción por compatibilidad.
    Devuelve None si la premisa no activó ningún concepto (señal no disponible).
    """
    if not conceptos_premisa:
        return None

    categorias_efectivas = {categoria}
    if conceptos_norma:
        _, categoria_de_tesauro = _indice_tesauro()
        if categoria not in set(categoria_de_tesauro.values()):
            categorias_efectivas |= set(conceptos_norma.values())

    coinciden_premisa = sum(
        1
        for cid, cat in conceptos_premisa.items()
        if cat in categorias_efectivas or (conceptos_norma is not None and cid in conceptos_norma)
    )
    frac_premisa = coinciden_premisa / len(conceptos_premisa)

    if conceptos_norma is None:
        return frac_premisa

    if not conceptos_norma:
        return 0.5 * frac_premisa

    compartidos = len(set(conceptos_premisa) & set(conceptos_norma))
    frac_norma = compartidos / len(conceptos_norma)
    return 0.5 * frac_premisa + 0.5 * frac_norma


# ----------------------------------------------------------------------------------------
# Coincidencia literal — señal L
# ----------------------------------------------------------------------------------------

def _contenido(texto: str) -> list[str]:
    """Tokens normalizados del texto con stemming morfológico, sin palabras vacías.

    Usa Snowball Stemmer en español para que variaciones de género y número
    (necesario/necesaria, obispo/obispos) coincidan en los n-gramas léxicos.
    """
    tokens = [t for t in _TOKENS.findall(normalizar(texto)) if t not in PALABRAS_VACIAS]
    return _stem_tokens(tokens)


def _ngramas(tokens: list[str], n: int) -> set[str]:
    """n-gramas de tokens consecutivos; si hay menos de n, devuelve palabras sueltas."""
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def similitud_literal(premisa: str, texto: str, n: int = 2, metrica: str = "contencion") -> float:
    """Señal L: coincidencia literal (léxica) entre la premisa y el texto de la norma.

    Se comparan n-gramas de palabras de CONTENIDO (sin vacías): en español los
    n-gramas con "de la", "que el" aparecen en cualquier norma y solo meten ruido.

    - "contencion": fracción de los n-gramas de la premisa que están en la norma
      (mide cuánto de lo que dice la noticia aparece textualmente en la norma).
    - "dice": similitud simétrica 2·|A∩B| / (|A|+|B|), que penaliza normas largas.

    :param n: tamaño del n-grama (config.toml → [ivr] n_gramas).
    :param metrica: "contencion" | "dice".
    :return: valor en [0, 1]; 0.0 si alguno de los textos no aporta contenido.
    :raises ValueError: si `metrica` no es conocida o `n` < 1.
    """
    if metrica not in METRICAS_LITERALES:
        raise ValueError(f"Métrica literal desconocida: {metrica!r}. Usar una de {METRICAS_LITERALES}.")
    if n < 1:
        raise ValueError(f"El tamaño del n-grama debe ser >= 1 (recibido: {n}).")

    tokens_premisa = _contenido(premisa)
    tokens_texto = _contenido(texto)
    if not tokens_premisa or not tokens_texto:
        return 0.0

    def _calcular_n(k: int) -> float:
        a, b = _ngramas(tokens_premisa, k), _ngramas(tokens_texto, k)
        if not a or not b:
            return 0.0
        comunes = len(a & b)
        if metrica == "contencion":
            return comunes / len(a)
        return 2 * comunes / (len(a) + len(b))

    if len(tokens_premisa) < n:
        # Premisa demasiado corta (p. ej. "los divorciados"): se compara por palabras.
        return _calcular_n(1)

    if n > 1 and len(tokens_premisa) < 12:
        # Opción 1C de auditoría: en premisas cortas (< 12 tokens de contenido),
        # los bigramas puros se rompen ante incisos normativos ("El bautismo, puerta
        # de los sacramentos... es necesaria para la salvación"). Promediar unigramas
        # y n-gramas con stemming evita el colapso a 0.0 y premia el orden exacto.
        return 0.5 * _calcular_n(1) + 0.5 * _calcular_n(n)

    return _calcular_n(n)


def _terminos_compartidos(conceptos_premisa: dict[str, str], conceptos_norma: dict[str, str]) -> float:
    """Extra (no entra en la fórmula): conceptos del tesauro que la norma también menciona.

    Aporta explicabilidad ("la noticia y la norma hablan de X") y queda como
    característica adicional para el Random Forest.
    """
    if not conceptos_premisa:
        return 0.0
    return len(set(conceptos_premisa) & set(conceptos_norma)) / len(conceptos_premisa)


# ----------------------------------------------------------------------------------------
# Fórmula IVR
# ----------------------------------------------------------------------------------------

def _recortar(valor) -> float:
    """Fuerza la señal al rango [0, 1], saneando explícitamente NaN e infinitos."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return min(max(v, 0.0), 1.0)


def _maximo(valores: list):
    """Máximo ignorando los None y NaN; None si no queda ninguno."""
    presentes = [float(v) for v in valores if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    return max(presentes) if presentes else None


def _medio(valores: list):
    """Media ignorando los None y NaN; None si no queda ninguno."""
    presentes = [float(v) for v in valores if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    return sum(presentes) / len(presentes) if presentes else None


def _minimo(valores: list):
    """Mínimo ignorando los None y NaN; None si no queda ninguno."""
    presentes = []
    for v in valores:
        if v is None:
            continue
        try:
            fv = float(v)
            if not (math.isnan(fv) or math.isinf(fv)):
                presentes.append(fv)
        except (TypeError, ValueError):
            continue
    return min(presentes) if presentes else None


def pesos_ivr(config=None) -> dict[str, float]:
    """Pesos de la fórmula leídos de config.toml → [ivr]."""
    conf = (config or obtener_configuraciones()).ivr
    return {
        "sb": conf.peso_sb,
        "literal": conf.peso_literal,
        "ontologico": conf.peso_ontologico,
        "autoridad": conf.peso_autoridad,
    }


def calcular_ivr_evidencia(senales: dict, pesos: dict) -> float:
    """Aplica la fórmula IVR a las señales de UNA evidencia.

    IVR = Σ wᵢ·sᵢ / Σ wᵢ, sumando solo las señales disponibles (valor no None).
    Así, cuando SO no aplica, los pesos se renormalizan solos.

    :param senales: {"sb": ..., "literal": ..., "ontologico": ..., "autoridad": ...}
    :return: IVR en [0, 1]; 0.0 si no hay ninguna señal disponible o todos los pesos son 0.
    """
    numerador = 0.0
    denominador = 0.0

    for nombre, peso in pesos.items():
        valor = senales.get(nombre)
        if valor is None or peso <= 0:
            continue
        numerador += peso * _recortar(valor)
        denominador += peso

    return numerador / denominador if denominador else 0.0


def calcular_ivr(
    premisa: str,
    n_resultados: int | None = None,
    where: dict | None = None,
    pesos: dict | None = None,
    consultar_fn: Callable[..., list[dict]] | None = None,
) -> dict:
    """Recupera evidencias con el RAG y las puntúa con el IVR.

    :param premisa: texto de la noticia (o afirmación) a evaluar.
    :param n_resultados: evidencias a recuperar; por defecto config.toml → [ivr] n_resultados.
    :param where: filtro de metadata opcional, igual que en `consultar()`.
    :param pesos: sobreescribe los pesos de config (útil para la ablación del TFG).
    :param consultar_fn: función opcional de recuperación (p. ej. `SesionExpansion.consultar_combinado`).
    :return: informe con las evidencias reordenadas por IVR. Si no hay evidencias
        devuelve {"ok": False, "motivo": ..., "evidencias": []}.

    Claves estables del informe (las consumen el RF y el testbed):
        ok, premisa, n_evidencias, pesos, conceptos_premisa, categorias_premisa,
        ivr (= ivr_max), ivr_max, ivr_medio, ivr_min, mejor,
        senales_max {sb, literal, ontologico, autoridad, terminos_compartidos},
        evidencias [contrato de consultar() + posicion_rag + senales + ivr]
    """
    settings = obtener_configuraciones()
    conf = settings.ivr

    if pesos is None:
        pesos = pesos_ivr(settings)

    n_res = conf.n_resultados if n_resultados is None else n_resultados
    if consultar_fn is None:
        evidencias_crudas = consultar(
            premisa,
            n_resultados=n_res,
            where=where,
        )
    else:
        evidencias_crudas = (
            consultar_fn(premisa, n_res, where)
            if where is not None
            else consultar_fn(premisa, n_res)
        )
    if not evidencias_crudas:
        return {
            "ok": False,
            "premisa": premisa,
            "motivo": "sin evidencias recuperadas (premisa vacía, filtro imposible o colección vacía)",
            "n_evidencias": 0,
            "ivr": 0.0,
            "evidencias": [],
        }

    conceptos_premisa = detectar_conceptos(premisa)

    evidencias = []
    for posicion, evidencia in enumerate(evidencias_crudas, start=1):
        conceptos_norma = detectar_conceptos(evidencia["texto"])
        senales = {
            "sb": _recortar(evidencia["similitud"]),
            "literal": similitud_literal(premisa, evidencia["texto"], conf.n_gramas, conf.metrica_literal),
            "ontologico": coincidencia_ontologica(conceptos_premisa, evidencia["categoria"], conceptos_norma),
            "autoridad": _recortar(evidencia["nivel_autoridad"]),
            "terminos_compartidos": _terminos_compartidos(conceptos_premisa, conceptos_norma),
        }
        evidencias.append({
            **evidencia,
            "posicion_rag": posicion,          # puesto que le dio el coseno, antes del IVR
            "senales": senales,
            "ivr": calcular_ivr_evidencia(senales, pesos),
        })

    # El IVR reordena: una norma puede no ser la más parecida en coseno y sí la más citable.
    evidencias.sort(key=lambda e: e["ivr"], reverse=True)

    valores_ivr = [e["ivr"] for e in evidencias]

    return {
        "ok": True,
        "premisa": premisa,
        "n_evidencias": len(evidencias),
        "pesos": dict(pesos),
        "conceptos_premisa": conceptos_premisa,
        "categorias_premisa": sorted(set(conceptos_premisa.values())),
        "ivr": max(valores_ivr),
        "ivr_max": max(valores_ivr),
        "ivr_medio": sum(valores_ivr) / len(valores_ivr),
        "ivr_min": min(valores_ivr),
        "mejor": evidencias[0],
        "senales_max": {
            nombre: _maximo([e["senales"][nombre] for e in evidencias])
            for nombre in ("sb", "literal", "ontologico", "autoridad", "terminos_compartidos")
        },
        "evidencias": evidencias,
    }


def vector_caracteristicas(informe: dict) -> dict[str, float]:
    """Vector plano de características para el Random Forest (Mes 3).

    Convierte el informe del IVR en números, sin textos ni dicts anidados.
    Las señales ausentes (SO cuando no aplica) van a 0.0 y se marcan con
    `so_disponible`, para que el modelo pueda distinguir "0 real" de "no aplica".

    :param informe: salida de `calcular_ivr()`.
    :return: dict de nombre -> float. Vacío si el informe es ok=False.
    """
    if not informe.get("ok"):
        return {}

    senales_max = informe["senales_max"]
    evidencias = informe["evidencias"]
    ivr_max, ivr_medio, ivr_min = informe["ivr_max"], informe["ivr_medio"], informe["ivr_min"]
    sb_max = senales_max["sb"] or 0.0
    sb_medio = _medio([e["senales"]["sb"] for e in evidencias]) or 0.0
    sb_min = _minimo([e["senales"]["sb"] for e in evidencias]) or 0.0
    ontologico_max = senales_max["ontologico"]

    return {
        "n_evidencias": float(informe["n_evidencias"]),
        "n_conceptos_premisa": float(len(informe["conceptos_premisa"])),
        "ivr_max": ivr_max,
        "ivr_medio": ivr_medio,
        "ivr_min": ivr_min,
        "ivr_brecha": ivr_max - ivr_medio,
        "sb_max": sb_max,
        "sb_medio": sb_medio,
        "sb_min": sb_min,
        "sb_brecha": sb_max - sb_min,
        "literal_max": senales_max["literal"] or 0.0,
        "literal_medio": _medio([e["senales"]["literal"] for e in evidencias]) or 0.0,
        "ontologico_max": ontologico_max or 0.0,
        "so_disponible": 1.0 if ontologico_max is not None else 0.0,
        "terminos_compartidos_max": senales_max["terminos_compartidos"] or 0.0,
        "autoridad_max": senales_max["autoridad"] or 0.0,
    }


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Calcula el IVR de una premisa contra el corpus canónico.")
    ap.add_argument("premisa", nargs="*", help="Premisa a evaluar.")
    ap.add_argument("--pdf", default=None, help="Ruta opcional a un PDF externo (expansión dinámica).")
    args = ap.parse_args()

    premisa = " ".join(args.premisa).strip() or "El bautismo esta prohibido"

    if args.pdf:
        from src.expansion_dinamica import SesionExpansion
        with SesionExpansion() as sesion:
            res_pdf = sesion.cargar_pdf(args.pdf)
            print(f"\n[Expansión dinámica] PDF cargado: {res_pdf.nombre} ({res_pdf.fragmentos} fragmentos, J={res_pdf.nivel_autoridad})")
            informe = calcular_ivr(premisa, consultar_fn=sesion.consultar_combinado)
    else:
        informe = calcular_ivr(premisa)

    print(f"\nPremisa: {premisa}")
    if not informe["ok"]:
        print(f"Sin evidencias: {informe['motivo']}")
        raise SystemExit(1)

    print(f"Conceptos del tesauro detectados: {informe['conceptos_premisa'] or 'ninguno'}")
    print(f"Pesos: {informe['pesos']}")
    print(f"\nIVR max = {informe['ivr_max']:.3f} | medio = {informe['ivr_medio']:.3f} | min = {informe['ivr_min']:.3f}")

    for evidencia in informe["evidencias"]:
        senales = evidencia["senales"]
        so = "n/a" if senales["ontologico"] is None else f"{senales['ontologico']:.3f}"
        origen = f" [{evidencia['origen']}]" if "origen" in evidencia else ""
        print(f"\n[{evidencia['posicion_rag']}→{evidencia['ivr']:.3f}]{origen} {evidencia['cita']}")
        print(f"  SB={senales['sb']:.3f}  L={senales['literal']:.3f}  SO={so}  J={senales['autoridad']:.3f}")
        print(f"  {evidencia['texto'][:180].strip()}...")
