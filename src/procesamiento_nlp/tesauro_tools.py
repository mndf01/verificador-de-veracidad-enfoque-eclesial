"""Herramientas para mantener src/procesamiento_nlp/tesauro.json.

Se ejecutan desde la raíz del proyecto:

    python -m src.procesamiento_nlp.tesauro_tools validar
    python -m src.procesamiento_nlp.tesauro_tools candidatos --top 40 --salida candidatos.csv

validar      Revisa el formato del tesauro y avisa de problemas que afectan a la recuperación:
             términos repetidos entre conceptos, términos canónicos que el corpus no usa (y qué
             dice el corpus en su lugar), palabras sueltas muy comunes que pueden ser ambiguas
             y categorías que el clasificador del corpus no conoce.
candidatos   Lista frases frecuentes del corpus que todavía no están en el tesauro, para
             revisarlas a mano y decidir cuáles agregar.

SOLO LEEN: nunca modifican tesauro.json. El tesauro sigue siendo un archivo que se edita a mano;
esto es la red de seguridad.

Peso de autoridad: el equipo acordó que el peso es SOLO el del documento (lo asigna el clasificador
del corpus), así que los conceptos del tesauro no necesitan `nivel_autoridad`. Si el campo existe
se acepta (y se valida que sea un número entre 0 y 1), pero no es obligatorio.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RUTA_TESAURO = Path("src/procesamiento_nlp/tesauro.json")
DIR_CORPUS = Path("src/corp_extractor/extractores/data")
RUTA_CLASIFICADOR = Path("src/corp_extractor/clasificador_corpus/clasificador_metadatos_v1.py")

# Una palabra suelta que aparece en tantos fragmentos del corpus puede significar otra cosa
# ("civil", "iglesia"...): se avisa para que se revise.
PALABRA_COMUN_MIN_FRAGMENTOS = 15

CLAVES_CONOCIDAS = {"termino_canonico", "variaciones_lexicas", "categoria", "nivel_autoridad"}

_STOP = set("""
de la el los las un una unos unas y o u e a al del en por para con sin sobre entre se su sus lo le les
que como no si es son ser sea sean han ha hay este esta estos estas ese esa esos esas aquel aquella
tambien mas muy pero sino ni cuando donde segun hasta desde contra ante bajo tras mismo misma mismos
mismas cada todo todos toda todas otro otra otros otras cualquier cual cuales quien quienes cuyo cuya
puede pueden debe deben tiene tienen hacer hace nada algo asi solo ya fue sido estar esta estan seran
sera podra podran haya hayan deba deban pueda puedan
""".split())

# Palabras de relleno que no forman términos del dominio ("teniendo en cuenta", "primer lugar"...).
_RELLENO = set("""
cuenta lugar tiempo modo manera parte caso casos forma efecto fin tanto cual nuestro nuestra nuestros
nuestras presente presentes siguiente siguientes primer primero primera segundo segunda tercero
teniendo quedando salvo debido conforme respecto
""".split())


class TesauroInvalido(Exception):
    """El archivo no se pudo leer como JSON (el mensaje indica la línea)."""


@dataclass
class Hallazgo:
    nivel: str          # "ERROR" | "AVISO" | "INFO"
    concepto: str       # id del concepto, o "" si es general
    mensaje: str


# ----------------------------------------------------------------------------------------
# Utilidades de texto
# ----------------------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y con espacios simples: para comparar sin que importen tildes."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", sin_tildes).strip()


def _tokens(texto: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", texto.lower())


def _patron_frase(frase_normalizada: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(frase_normalizada) + r"(?!\w)")


# ----------------------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------------------

def cargar_tesauro(ruta: Path) -> dict:
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise TesauroInvalido(f"No se encontró el tesauro en {ruta}.") from None
    except json.JSONDecodeError as e:
        raise TesauroInvalido(
            f"El tesauro tiene un error de formato en la línea {e.lineno}, columna {e.colno}: {e.msg}. "
            "Suele ser una coma de más o de menos, o unas comillas sin cerrar."
        ) from None


def cargar_corpus(directorio: Path) -> list[str]:
    """Textos de todos los fragmentos de los corpus crudos (corpus_*_v1.json). [] si no hay."""
    textos: list[str] = []
    for archivo in sorted(Path(directorio).glob("corpus_*_v1.json")):
        datos = json.loads(archivo.read_text(encoding="utf-8-sig")).get("datos", [])
        textos.extend(item["texto"] for item in datos if item.get("texto"))
    return textos


def categorias_del_clasificador(ruta: Path = RUTA_CLASIFICADOR) -> set[str] | None:
    """Categorías que el clasificador del corpus puede asignar, leídas de sus propias reglas.

    Devuelve None si no se puede leer (así la herramienta sigue funcionando sin esa comprobación).
    """
    try:
        spec = importlib.util.spec_from_file_location("_clasificador", ruta)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        reglas = modulo.REGLAS_ETL
    except Exception:
        return None

    categorias: set[str] = set()

    def recorrer(obj) -> None:
        if isinstance(obj, dict):
            for clave, valor in obj.items():
                if clave in ("categoria", "categoria_default") and isinstance(valor, str):
                    categorias.add(valor)
                elif clave == "mapeo_interno" and isinstance(valor, dict):
                    categorias.update(v for v in valor.values() if isinstance(v, str))
                recorrer(valor)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                recorrer(x)

    recorrer(reglas)
    return categorias or None


# ----------------------------------------------------------------------------------------
# Sugerencias a partir del corpus
# ----------------------------------------------------------------------------------------

def sugerir_formas(termino: str, textos: list[str], k: int = 3) -> list[tuple[str, int]]:
    """Cómo dice el corpus lo que el tesauro llama `termino`: [(frase, nº de fragmentos), ...].

    Busca frases de 1 a 4 palabras que compartan la raíz de las palabras importantes del término
    ("Dimisión del estado clerical" → "estado clerical", "expulsión del estado clerical").
    Las raíces raras en el corpus pesan más que las comunes: para "Derecho de huelga" interesa
    "huelga", no "derecho", que aparece en cientos de fragmentos.
    """
    palabras = [normalizar(w) for w in _tokens(termino)]
    palabras = [p for p in palabras if p not in _STOP and len(p) >= 4]
    if not palabras:
        return []
    raices = [p[: max(5, len(p) - 2)] for p in palabras]

    frases: dict[str, list] = {}                           # frase normalizada → [raíces cubiertas, fragmentos]
    mostrada: dict[str, Counter] = {}                      # frase normalizada → cómo se escribe (con tildes)
    df_raiz = [0] * len(raices)
    termino_norm = normalizar(termino)
    for texto in textos:
        toks = _tokens(texto)
        norm = [normalizar(t) for t in toks]
        for j, r in enumerate(raices):
            if any(t.startswith(r) for t in norm):
                df_raiz[j] += 1
        vistas: set[str] = set()
        for i, t in enumerate(norm):
            if not any(t.startswith(r) for r in raices):
                continue
            for n in range(1, 5):
                for inicio in range(max(0, i - n + 1), i + 1):
                    fin = inicio + n
                    if fin > len(toks) or norm[inicio] in _STOP or norm[fin - 1] in _STOP:
                        continue
                    frase = " ".join(norm[inicio:fin])
                    if frase in vistas or frase == termino_norm:
                        continue
                    vistas.add(frase)
                    cubiertas = frozenset(j for j, r in enumerate(raices) if any(w.startswith(r) for w in norm[inicio:fin]))
                    registro = frases.setdefault(frase, [cubiertas, 0])
                    registro[1] += 1
                    mostrada.setdefault(frase, Counter())[" ".join(toks[inicio:fin])] += 1

    total = max(len(textos), 1)
    rareza = [math.log((total + 1) / (d + 1)) for d in df_raiz]
    ordenadas = sorted(
        frases.items(),
        key=lambda kv: (-sum(rareza[j] for j in kv[1][0]), -kv[1][1], len(kv[0])),
    )
    return [(mostrada[f].most_common(1)[0][0], datos[1]) for f, datos in ordenadas[:k]]


def buscar_candidatos(textos: list[str], conocidos: set[str], minimo: int = 12, top: int = 40) -> list[dict]:
    """Frases de 2-3 palabras frecuentes en el corpus que aún no están en el tesauro."""
    frecuencia: Counter = Counter()
    ejemplo: dict[str, str] = {}
    escrita: dict[str, str] = {}                           # cómo se escribe en el corpus (con tildes)
    for texto in textos:
        toks = _tokens(texto)
        norm = [normalizar(t) for t in toks]
        vistas: set[str] = set()
        for n in (2, 3):
            for i in range(len(toks) - n + 1):
                ventana = norm[i:i + n]
                if ventana[0] in _STOP or ventana[-1] in _STOP:
                    continue
                if any(len(w) < 3 for w in (ventana[0], ventana[-1])):
                    continue
                if any(w in _RELLENO for w in ventana):
                    continue
                if not any(w not in _STOP and len(w) >= 5 for w in ventana):
                    continue
                frase = " ".join(ventana)
                if frase in vistas:
                    continue
                vistas.add(frase)
                frecuencia[frase] += 1
                escrita.setdefault(frase, " ".join(toks[i:i + n]))
                ejemplo.setdefault(frase, " ".join(toks[max(0, i - 4):i + n + 4]))

    conocidos_norm = [normalizar(c) for c in conocidos]
    resultado = []
    for frase, n in frecuencia.most_common():
        if n < minimo or len(resultado) >= top:
            break
        if any(frase == c or _patron_frase(frase).search(c) for c in conocidos_norm):
            continue
        resultado.append({"frase": escrita[frase], "fragmentos": n, "ejemplo": ejemplo[frase]})
    return resultado


# ----------------------------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------------------------

def validar(raw: dict, textos: list[str], categorias: set[str] | None) -> list[Hallazgo]:
    h: list[Hallazgo] = []
    conceptos = raw.get("conceptos") if isinstance(raw, dict) else None
    if not isinstance(conceptos, dict) or not conceptos:
        return [Hallazgo("ERROR", "", "El archivo debe tener una clave \"conceptos\" con al menos un concepto.")]

    textos_norm = [normalizar(t) for t in textos]
    usos: dict[str, set[str]] = {}           # término normalizado → conceptos que lo usan
    con_nivel: list[str] = []
    cache_conteo: dict[str, int] = {}

    def conteo(frase_norm: str) -> int:
        if frase_norm not in cache_conteo:
            patron = _patron_frase(frase_norm)
            cache_conteo[frase_norm] = sum(1 for t in textos_norm if patron.search(t))
        return cache_conteo[frase_norm]

    for cid, c in conceptos.items():
        if not re.fullmatch(r"ID_[A-Z0-9_]+", str(cid)):
            h.append(Hallazgo("ERROR", cid, "El id debe empezar con ID_ y usar solo mayúsculas, números y guion bajo (ej.: ID_PAPA)."))
        if not isinstance(c, dict):
            h.append(Hallazgo("ERROR", cid, "El concepto debe ser un objeto con termino_canonico, variaciones_lexicas y categoria."))
            continue

        canon = c.get("termino_canonico")
        if not isinstance(canon, str) or not canon.strip():
            h.append(Hallazgo("ERROR", cid, "Falta \"termino_canonico\" (el término técnico que usan los documentos oficiales)."))
            canon = ""
        variantes = c.get("variaciones_lexicas")
        if not isinstance(variantes, list) or not variantes:
            h.append(Hallazgo("ERROR", cid, "Falta \"variaciones_lexicas\" (una lista con las formas en que lo dice la prensa)."))
            variantes = []
        elif any(not isinstance(v, str) or not v.strip() for v in variantes):
            h.append(Hallazgo("ERROR", cid, "\"variaciones_lexicas\" tiene un elemento vacío o que no es texto."))
            variantes = [v for v in variantes if isinstance(v, str) and v.strip()]
        categoria = c.get("categoria")
        if not isinstance(categoria, str) or not categoria.strip():
            h.append(Hallazgo("ERROR", cid, "Falta \"categoria\"."))
            categoria = ""

        extras = set(c) - CLAVES_CONOCIDAS
        if extras:
            h.append(Hallazgo("AVISO", cid, f"Campos desconocidos: {sorted(extras)} (se ignorarán)."))
        if "nivel_autoridad" in c:
            con_nivel.append(cid)
            n = c["nivel_autoridad"]
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not (0 < n <= 1):
                h.append(Hallazgo("ERROR", cid, f"nivel_autoridad debe ser un número entre 0 y 1 (tiene {n!r})."))

        # términos repetidos
        propios: Counter = Counter()
        for termino in ([canon] if canon else []) + variantes:
            propios[normalizar(termino)] += 1
            usos.setdefault(normalizar(termino), set()).add(cid)
        for t, veces in propios.items():
            if veces > 1:
                h.append(Hallazgo("AVISO", cid, f"El término \"{t}\" está repetido dentro del mismo concepto."))

        # categoría que el clasificador del corpus no conoce
        if categorias is not None and categoria and categoria not in categorias:
            cercana = difflib.get_close_matches(categoria, sorted(categorias), n=1, cutoff=0.6)
            ayuda = f" ¿Quizá \"{cercana[0]}\"?" if cercana else ""
            h.append(Hallazgo("AVISO", cid, f"La categoría \"{categoria}\" no existe en el clasificador del corpus.{ayuda}"))

        if textos_norm:
            # el término canónico debería ser como lo escriben los documentos oficiales
            if canon and conteo(normalizar(canon)) == 0:
                formas = sugerir_formas(canon, textos)
                sugerencia = ""
                if formas:
                    sugerencia = " El corpus dice, por ejemplo: " + "; ".join(f"\"{f}\" ({n} fragmentos)" for f, n in formas) + "."
                h.append(Hallazgo("AVISO", cid, f"El término canónico \"{canon}\" no aparece en ningún fragmento del corpus.{sugerencia}"))
            # palabras sueltas muy comunes: posible ambigüedad (salvo que sean del propio concepto)
            raices_canon = [normalizar(w)[:5] for w in _tokens(canon) if len(w) >= 5]
            for v in variantes:
                vn = normalizar(v)
                if len(vn) >= 5 and vn[:5] in raices_canon:        # "Matrimonio" dentro de "Vínculo Matrimonial"
                    continue
                if " " not in vn and conteo(vn) >= PALABRA_COMUN_MIN_FRAGMENTOS:
                    h.append(Hallazgo("AVISO", cid, f"La variante \"{v}\" es una sola palabra y aparece en {conteo(vn)} fragmentos del corpus: "
                                                     "revisa si en una noticia podría significar otra cosa (posible ambigüedad)."))

    for termino, ids in sorted(usos.items()):
        if len(ids) > 1:
            h.append(Hallazgo("ERROR", "", f"El término \"{termino}\" está en varios conceptos ({', '.join(sorted(ids))}): la traducción sería ambigua."))

    if con_nivel:
        h.append(Hallazgo("INFO", "", f"{len(con_nivel)} concepto(s) tienen nivel_autoridad. El equipo acordó que el peso es solo el del documento, "
                                      "así que ese campo no se usa (se puede dejar o quitar)."))
    if not textos_norm:
        h.append(Hallazgo("INFO", "", "No se encontró el corpus en esta máquina: se omitieron las comprobaciones contra los textos oficiales."))
    if categorias is None:
        h.append(Hallazgo("INFO", "", "No se pudieron leer las categorías del clasificador: se omitió esa comprobación."))
    return h


def formatear(hallazgos: list[Hallazgo], n_conceptos: int) -> str:
    orden = {"ERROR": 0, "AVISO": 1, "INFO": 2}
    lineas = []
    for hz in sorted(hallazgos, key=lambda x: (orden[x.nivel], x.concepto)):
        etiqueta = f"[{hz.concepto}] " if hz.concepto else ""
        lineas.append(f"{hz.nivel:<6} {etiqueta}{hz.mensaje}")
    cuenta = Counter(hz.nivel for hz in hallazgos)
    lineas.append("")
    lineas.append(f"{n_conceptos} conceptos revisados: {cuenta['ERROR']} error(es), {cuenta['AVISO']} aviso(s).")
    if not cuenta["ERROR"] and not cuenta["AVISO"]:
        lineas.append("Todo en orden.")
    return "\n".join(lineas)


# ----------------------------------------------------------------------------------------
# Línea de comandos
# ----------------------------------------------------------------------------------------

def _conocidos(raw: dict) -> set[str]:
    conocidos: set[str] = set()
    for c in (raw.get("conceptos") or {}).values():
        if isinstance(c, dict):
            conocidos.update(v for v in [c.get("termino_canonico", ""), *c.get("variaciones_lexicas", [])] if isinstance(v, str))
    return conocidos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tesauro_tools", description="Herramientas para mantener tesauro.json (solo lectura).")
    ap.add_argument("--tesauro", type=Path, default=RUTA_TESAURO)
    ap.add_argument("--corpus", type=Path, default=DIR_CORPUS, help="Carpeta con los corpus_*_v1.json")
    sub = ap.add_subparsers(dest="orden", required=True)
    sub.add_parser("validar", help="Revisa formato y coherencia con el corpus.")
    c = sub.add_parser("candidatos", help="Frases frecuentes del corpus que aún no están en el tesauro.")
    c.add_argument("--minimo", type=int, default=12, help="Mínimo de fragmentos en que debe aparecer (12).")
    c.add_argument("--top", type=int, default=40, help="Cuántas mostrar (40).")
    c.add_argument("--salida", type=Path, help="Guardar en un CSV (se abre bien en Excel).")
    args = ap.parse_args(argv)

    try:
        raw = cargar_tesauro(args.tesauro)
    except TesauroInvalido as e:
        print(f"ERROR  {e}", file=sys.stderr)
        return 1
    textos = cargar_corpus(args.corpus)

    if args.orden == "validar":
        hallazgos = validar(raw, textos, categorias_del_clasificador())
        print(formatear(hallazgos, len(raw.get("conceptos") or {})))
        return 1 if any(h.nivel == "ERROR" for h in hallazgos) else 0

    if not textos:
        print(f"ERROR  No se encontró el corpus en {args.corpus}. Genéralo primero con el ETL.", file=sys.stderr)
        return 1
    filas = buscar_candidatos(textos, _conocidos(raw), args.minimo, args.top)
    print(f"{len(filas)} candidatos (aparecen en ≥ {args.minimo} fragmentos y no están en el tesauro). "
          "Revisa cuáles son términos del dominio; habrá ruido.\n")
    for f in filas:
        print(f"{f['fragmentos']:>5}  {f['frase']:<36} …{f['ejemplo']}…")
    if args.salida:
        with open(args.salida, "w", newline="", encoding="utf-8-sig") as fh:       # utf-8-sig: Excel respeta las tildes
            w = csv.DictWriter(fh, fieldnames=["frase", "fragmentos", "ejemplo", "agregar_a_concepto"], delimiter=";")
            w.writeheader()
            for f in filas:
                w.writerow({**f, "agregar_a_concepto": ""})
        print(f"\nGuardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
