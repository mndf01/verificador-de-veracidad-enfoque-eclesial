"""Segmentación heurística de un documento normativo externo (etapa pura).

Estrategia (tesis, sección 2.2.2 y 2.2.8): analizadores heurísticos y expresiones
regulares que detectan "palabras desencadenantes" (trigger words) que marcan el inicio
de las unidades lógicas, aprovechando que los textos normativos tienen una estructura
fuertemente estandarizada.

Se prueban tres esquemas, del más al menos estructurado:

  1. "articulado"  → marcadores "Artículo N", "Art. N", "Canon N", "Can. N"...
                     más encabezados LIBRO / TÍTULO / CAPÍTULO / SECCIÓN que alimentan
                     el contexto jerárquico (igual que el corpus base).
  2. "numerado"    → párrafos numerados "12. Texto..." con numeración creciente
                     (estilo numerales del Compendio o de documentos conciliares).
  3. "ventanas"    → sin estructura detectable: ventanas de ~palabras_max palabras
                     cortadas en límites de oración.

Una unidad más larga que `palabras_max` se parte en ventanas homogéneas para no exceder
la capacidad útil del modelo de embeddings.

Limitación conocida: las heurísticas son deterministas y auditables, pero un documento
con un formato muy atípico puede caer al esquema "ventanas". Siempre se devuelven
advertencias para que el usuario/desarrollador lo note.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from bisect import bisect_right
from dataclasses import dataclass, field

from .tipos import Fragmento, ResultadoSegmentacion

# ----------------------------------------------------------------------------------------
# Expresiones regulares
# ----------------------------------------------------------------------------------------

_ORDINALES = {
    "primero": 1, "primera": 1, "segundo": 2, "segunda": 2, "tercero": 3, "tercera": 3,
    "cuarto": 4, "cuarta": 4, "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6,
    "septimo": 7, "septima": 7, "octavo": 8, "octava": 8, "noveno": 9, "novena": 9,
    "decimo": 10, "decima": 10, "unico": 1, "unica": 1,
}

# "Artículo 12.-", "Art. 3º", "ARTÍCULO PRIMERO", "Canon 5", "Can. 12 § 1", "Cann. 1-6"
_RE_UNIDAD = re.compile(
    r"^(?P<tipo>Art[íi]culo|Art\.?|Cann?\.|Canon|Cn\.)\s*"
    r"(?P<num>\d{1,4}|[A-Za-zÁÉÍÓÚáéíóúñ]{3,10})"
    r"(?P<suf>\s+(?:bis|ter|qu[áa]ter))?"
    r"\s*(?P<sep>[.\-–—:º°ª)]*)\s*(?P<resto>.*)$",
    re.IGNORECASE,
)

_ORD_ENC = (
    r"PRIMER[OA]|SEGUND[OA]|TERCER[OA]|CUART[OA]|QUINT[OA]|SEXT[OA]|S[ÉE]PTIM[OA]|"
    r"OCTAV[OA]|NOVEN[OA]|D[ÉE]CIM[OA]|[ÚU]NIC[OA]|PRELIMINAR"
)
# El número romano debe ir en mayúsculas para no confundir "mi", "di", "civil"...
_RE_ENCABEZADO = re.compile(
    rf"^(?P<tipo>LIBRO|PARTE|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N)\s+(?:[-–—:.]\s*)?"
    rf"(?P<num>(?-i:[IVXLCDM]{{1,8}})|\d{{1,3}}|{_ORD_ENC})\b(?P<resto>.*)$",
    re.IGNORECASE,
)
# No hay un orden universal entre estos niveles: en el Código de Derecho Canónico la SECCIÓN
# va encima del TÍTULO, y en una ley civil española va debajo del CAPÍTULO. Por eso la
# jerarquía NO se fija de antemano: se aprende del orden en que aparecen en cada documento.

# "12. La Iglesia...", "7) Los fieles..."
_RE_NUMERADO = re.compile(r'^(?P<n>\d{1,4})\s*[.)]\s+(?P<resto>\S.*)$')

_SUBTITULO = "SUBTITULO"     # tipo interno para subtítulos sin palabra clave
_FIN_ORACION = (".", ":", ";", "?", "!", "»", "”", '"', ")", "…")
_INICIO_MARCA_FUERTE = ("«", "“", '"', "¿", "¡", "(", "—", "–", "-")
_LIMITE_ENCABEZADO = 160            # encabezado en mayúsculas/minúsculas mezcladas
_LIMITE_ENCABEZADO_MAYUS = 240      # encabezado mayormente en MAYÚSCULAS (p. ej. "TÍTULO IV – ... (Cann. 35–93)")
# En columnas estrechas un título largo se parte en varias líneas; estos topes evitan que
# el resto del título termine pegado al texto de una unidad.
_MAX_LINEAS_CONTINUACION = 4        # líneas extra que puede ocupar un encabezado con palabra clave
_MAX_LINEAS_SUBTITULO = 5           # líneas que puede ocupar un subtítulo sin palabra clave

_CORTE_ORACION = re.compile(r'(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡"«“(\d])')


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _mayormente_mayusculas(texto: str, umbral: float = 0.75) -> bool:
    """True si casi todas las letras son mayúsculas, ignorando lo que va entre paréntesis.

    Los títulos llevan referencias como "(Cann. 1370–1378)" que son mayormente minúsculas;
    contarlas rebajaría el porcentaje de una línea que a todas luces es un título.
    """
    texto = re.sub(r"\([^)]*\)?", "", texto)           # también quita un "(Cann." sin cerrar
    letras = [c for c in texto if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) / len(letras) >= umbral


_RE_REFERENCIA_CANONES = re.compile(r"^\(?\s*Cann?\.\s*[\d\s,–\-y]+\)?$", re.IGNORECASE)


def _es_referencia_canones(linea: str) -> bool:
    """'(Cann. 3-4)' suelto: segunda línea típica de un subtítulo que no cupo en una sola."""
    return bool(_RE_REFERENCIA_CANONES.match(linea.strip()))


def _termina_en_punto(linea: str) -> bool:
    """¿La línea termina en punto? Un punto dentro de un paréntesis ("(Cann.") no cuenta."""
    return re.sub(r"\([^)]*\)?", "", linea).rstrip().endswith(".")


def _es_resto_numerico(linea: str) -> bool:
    """'368–430)': lo que queda de un '(Cann. 368–430)' partido por el ancho de la página."""
    linea = linea.strip()
    return len(linea) <= 30 and any(c.isdigit() for c in linea) and not any(c.isalpha() for c in linea)


def _es_continuacion_titulo(linea: str) -> bool:
    """¿Esta línea puede ser la continuación de un encabezado/subtítulo que no cupo en una línea?"""
    linea = linea.strip()
    return (
        len(linea) <= 100
        and not _termina_en_punto(linea)
        and (_mayormente_mayusculas(linea) or _es_referencia_canones(linea) or _es_resto_numerico(linea))
    )


def _es_subtitulo(linea: str) -> bool:
    """Línea corta, mayormente en mayúsculas y sin punto final: candidata a subtítulo."""
    return (
        sum(c.isalpha() for c in linea) >= 3
        and len(linea) <= _LIMITE_ENCABEZADO_MAYUS
        and not _termina_en_punto(linea)
        and _mayormente_mayusculas(linea)
    )


def _cierra_oracion(linea: str | None) -> bool:
    return linea is None or linea.rstrip().endswith(_FIN_ORACION)


# ----------------------------------------------------------------------------------------
# Clasificación de líneas
# ----------------------------------------------------------------------------------------

@dataclass
class _Marca:
    tipo: str                       # "enc" | "unidad" | "texto"
    tipo_enc: str = ""              # solo "enc": LIBRO | PARTE | TITULO | CAPITULO | SECCION
    id_norma: int = 0               # solo "unidad"
    familia: str = ""               # solo "unidad": "Canon" | "Artículo"
    etiqueta: str = ""              # solo "unidad"
    resto: str = ""                 # texto que sigue a la marca en la misma línea


def _parse_encabezado(linea: str, prev_cierra: bool) -> str | None:
    """Devuelve el tipo de encabezado ("LIBRO", "PARTE", "TITULO", "CAPITULO", "SECCION") o None."""
    mayus = _mayormente_mayusculas(linea)
    if not linea[:1].isupper() or len(linea) > (_LIMITE_ENCABEZADO_MAYUS if mayus else _LIMITE_ENCABEZADO):
        return None
    m = _RE_ENCABEZADO.match(linea)
    if not m:
        return None
    # Una línea "Título 3 de la ley establece..." que continúa una oración NO es encabezado.
    if not (prev_cierra or mayus):
        return None
    return _sin_acentos(m["tipo"]).upper()


def _parse_unidad(linea: str, prev_cierra: bool) -> _Marca | None:
    # Una cita en medio de una oración ("...según el artículo 2 de este decreto") casi
    # siempre empieza en minúscula tras un salto de línea: se descarta de entrada.
    if not linea[:1].isupper():
        return None
    m = _RE_UNIDAD.match(linea)
    if not m:
        return None

    num = m["num"]
    if num.isdigit():
        n = int(num)
    else:
        n = _ORDINALES.get(_sin_acentos(num.lower()))
        if n is None:                       # "Artículos", "Artesanal", "Cantidad"...
            return None

    sep, resto = m["sep"] or "", m["resto"].strip()
    # Marca fuerte: puntuación de enumeración, o la línea termina ahí, o lo que sigue
    # empieza como una oración nueva. Marca débil ("Artículo 5 de la ley establece"):
    # solo se acepta si la línea anterior cerró una oración.
    fuerte = bool(sep) or not resto or resto[0].isupper() or resto[0] in _INICIO_MARCA_FUERTE
    if not fuerte:
        return None
    if not (prev_cierra or sep or not resto):
        return None

    tipo = "Canon" if _sin_acentos(m["tipo"]).lower().startswith(("can", "cn")) else "Artículo"
    suf = (m["suf"] or "").strip()
    etiqueta = f"{tipo} {num}" + (f" {suf}" if suf else "")
    return _Marca("unidad", id_norma=n, familia=tipo, etiqueta=etiqueta, resto=resto)


def _siguiente_numero(lineas: list[tuple[int, str]], desde: int, ventana: int = 250) -> int | None:
    """Primer número de párrafo numerado que aparece a partir de `desde` (o None)."""
    for j in range(desde, min(len(lineas), desde + ventana)):
        m = _RE_NUMERADO.match(lineas[j][1])
        if m:
            return int(m["n"])
    return None


def _clasificar(lineas: list[tuple[int, str]], modo: str) -> list[_Marca]:
    marcas: list[_Marca] = []
    prev_cierra = True
    ultimo_n: int | None = None

    for idx, (_, txt) in enumerate(lineas):
        tipo_enc = _parse_encabezado(txt, prev_cierra)
        if tipo_enc:
            marcas.append(_Marca("enc", tipo_enc=tipo_enc))
            prev_cierra = True
            continue

        marca: _Marca | None = None
        if modo == "articulado":
            marca = _parse_unidad(txt, prev_cierra)
        elif modo == "numerado":
            m = _RE_NUMERADO.match(txt)
            if m:
                n = int(m["n"])
                # Numeración creciente: descarta sublistas internas ("1.", "2." dentro del 12).
                if ultimo_n is None:
                    # Primer numeral: normalmente ≤ 10, pero un capítulo suelto puede empezar en el 20;
                    # entonces vale si el siguiente número que aparece es justo el que le sigue.
                    acepta = n <= 10 or _siguiente_numero(lineas, idx + 1) == n + 1
                else:
                    acepta = ultimo_n < n <= ultimo_n + 20
                # El número que sigue exactamente al anterior (28 tras 27) es una señal fuerte y
                # se acepta aunque la línea previa no cierre oración (p. ej. un numeral que termina
                # en un título o una cita sin punto). Un salto exige además que la oración haya cerrado.
                esperado = ultimo_n is not None and n == ultimo_n + 1
                # Un salto (o el primer numeral) exige además que el texto empiece como una oración;
                # el numeral esperado se acepta con cualquier inicio ("§1. ...", "a) ...").
                inicia_oracion = m["resto"][0].isupper() or m["resto"][0] in '¿¡"«“'
                if esperado or (acepta and prev_cierra and inicia_oracion):
                    ultimo_n = n
                    marca = _Marca("unidad", id_norma=n, etiqueta=f"Numeral {n}", resto=m["resto"].strip())

        if marca:
            marcas.append(marca)
        else:
            marcas.append(_Marca("texto"))
        # Un título en mayúsculas ("NOSTRA AETATE") no lleva punto, pero tras él puede empezar
        # una unidad: cuenta como si hubiera cerrado una oración.
        prev_cierra = _cierra_oracion(txt) or _es_subtitulo(txt)

    return marcas


def _resolver_familias(marcas: list[_Marca]) -> list[_Marca]:
    """Si el documento mezcla "Can." y "Art.", la familia dominante son las unidades.

    En el Código de Derecho Canónico las unidades son los cánones, y "Art. 1 – DE LA LIBRE
    COLACIÓN" es un encabezado de subdivisión que usa la misma palabra. Los marcadores de la
    familia minoritaria pasan a ser subtítulos (se conservan en el contexto, no como unidades).
    Si ambas familias tienen peso parecido no se toca nada: es ambiguo y mejor no adivinar.
    """
    conteo = Counter(m.familia for m in marcas if m.tipo == "unidad")
    if len(conteo) < 2:
        return marcas
    (dominante, n_dom), (_, n_min) = conteo.most_common(2)[0], conteo.most_common(2)[1]
    if n_min > 0.5 * n_dom:
        return marcas
    for m in marcas:
        if m.tipo == "unidad" and m.familia != dominante:
            m.tipo, m.tipo_enc = "enc", _SUBTITULO
    return marcas


# ----------------------------------------------------------------------------------------
# Armado de unidades
# ----------------------------------------------------------------------------------------

@dataclass
class _Unidad:
    etiqueta: str
    id_norma: int
    contexto: list[str]
    pagina_marca: int
    lineas: list[tuple[int, str]] = field(default_factory=list)


def _armar_unidades(lineas: list[tuple[int, str]], marcas: list[_Marca]) -> list[_Unidad]:
    # Pila de (tipo, título). Un tipo NUEVO se anida bajo el actual; un tipo que ya está en la
    # pila la recorta hasta ese punto y lo reemplaza (un "TÍTULO II" cierra al "TÍTULO I" y a
    # los CAPÍTULOS que colgaban de él). Así LIBRO > PARTE > TÍTULO > CAPÍTULO se conserva.
    pila: list[tuple[str, str]] = []
    unidades: list[_Unidad] = []
    actual: _Unidad | None = None

    def contexto() -> list[str]:
        return [titulo for _, titulo in pila]

    def cerrar() -> None:
        nonlocal actual
        if actual is not None:
            unidades.append(actual)
            actual = None

    i = 0
    while i < len(lineas):
        pagina, txt = lineas[i]
        marca = marcas[i]

        if marca.tipo == "enc":
            cerrar()
            if pila and pila[-1][0] == _SUBTITULO:      # un encabezado con palabra clave cierra el subtítulo
                pila.pop()
            titulo = txt
            # "CAPÍTULO I" / "DISPOSICIONES GENERALES": el título suele ir en la línea siguiente
            # (y a veces se parte en varias). Se aceptan hasta _MAX_LINEAS_CONTINUACION.
            for k in range(_MAX_LINEAS_CONTINUACION):
                if i + 1 < len(lineas) and marcas[i + 1].tipo == "texto":
                    sig = lineas[i + 1][1]
                    if _es_continuacion_titulo(sig):
                        # "CAPÍTULO I" + "DE LOS FIELES" → "CAPÍTULO I – DE LOS FIELES";
                        # si el encabezado ya traía su título, la continuación se pega con un espacio.
                        sep = " – " if (k == 0 and len(txt.split()) <= 3) else " "
                        titulo = f"{titulo}{sep}{sig}"
                        i += 1
                        continue
                break
            tipos = [t for t, _ in pila]
            if marca.tipo_enc in tipos:
                del pila[tipos.index(marca.tipo_enc):]
            pila.append((marca.tipo_enc, titulo[:_LIMITE_ENCABEZADO_MAYUS]))

        elif marca.tipo == "unidad":
            cerrar()
            actual = _Unidad(marca.etiqueta, marca.id_norma, contexto(), pagina)
            if marca.resto:
                actual.lineas.append((pagina, marca.resto))

        else:
            # Subtítulo sin palabra clave ("DE LA LIBRE COLACIÓN (Can. 157)"): línea(s) en mayúsculas
            # INMEDIATAMENTE antes de un artículo/canon. Sin esa condición no se toca el texto.
            if _es_subtitulo(txt):
                j, partes = i, []
                while (
                    j < len(lineas) and len(partes) < _MAX_LINEAS_SUBTITULO and marcas[j].tipo == "texto"
                    and (_es_subtitulo(lineas[j][1]) or (partes and _es_continuacion_titulo(lineas[j][1])))
                ):
                    partes.append(lineas[j][1])
                    j += 1
                if j < len(lineas) and marcas[j].tipo == "unidad":
                    cerrar()
                    tipos = [t for t, _ in pila]
                    if _SUBTITULO in tipos:
                        del pila[tipos.index(_SUBTITULO):]
                    pila.append((_SUBTITULO, " ".join(partes)[:_LIMITE_ENCABEZADO_MAYUS]))
                    i = j
                    continue
            if actual is None:
                actual = _Unidad("Preámbulo" if not unidades else "Texto sin numerar", 0, contexto(), pagina)
            actual.lineas.append((pagina, txt))
        i += 1

    cerrar()
    return unidades


# ----------------------------------------------------------------------------------------
# Construcción de fragmentos (partición de unidades largas)
# ----------------------------------------------------------------------------------------

def _unir(lineas: list[tuple[int, str]]) -> tuple[str, list[int], list[int]]:
    """Une las líneas en un texto y guarda, para cada línea, dónde empieza y en qué página está."""
    partes, inicios, paginas, pos = [], [], [], 0
    for pagina, txt in lineas:
        inicios.append(pos)
        paginas.append(pagina)
        partes.append(txt)
        pos += len(txt) + 1
    return " ".join(partes), inicios, paginas


def _pagina_en(pos: int, inicios: list[int], paginas: list[int]) -> int:
    return paginas[max(bisect_right(inicios, pos) - 1, 0)]


def _oraciones(texto: str) -> list[tuple[int, int]]:
    cortes = list(_CORTE_ORACION.finditer(texto))
    inicios = [0] + [m.end() for m in cortes]
    finales = [m.start() for m in cortes] + [len(texto)]
    return list(zip(inicios, finales))


def _ventanas(texto: str, palabras_max: int) -> list[tuple[int, int]]:
    """Parte `texto` en ventanas homogéneas de ≤ palabras_max palabras, cortando en oraciones.

    Devuelve spans (inicio, fin) sobre `texto`.
    """
    total = len(texto.split())
    k = max(1, math.ceil(total / palabras_max))
    objetivo = math.ceil(total / k)

    ventanas: list[tuple[int, int]] = []
    ini: int | None = None
    fin = 0
    acum = 0

    def volcar() -> None:
        nonlocal ini, acum
        if ini is not None:
            ventanas.append((ini, fin))
        ini, acum = None, 0

    for s, e in _oraciones(texto):
        palabras = len(texto[s:e].split())
        if palabras > palabras_max:
            # Una "oración" gigante (sin puntuación): se corta por palabras.
            volcar()
            tokens = list(re.finditer(r"\S+", texto[s:e]))
            for j in range(0, len(tokens), palabras_max):
                grupo = tokens[j:j + palabras_max]
                ventanas.append((s + grupo[0].start(), s + grupo[-1].end()))
            continue
        if acum and (acum + palabras > palabras_max or acum >= objetivo):
            volcar()
        if ini is None:
            ini = s
        fin = e
        acum += palabras
    volcar()
    return ventanas


def _fragmentos_de(u: _Unidad, palabras_max: int, numerar: bool = False) -> list[Fragmento]:
    texto, inicios, paginas = _unir(u.lineas)
    if not texto.strip():
        return []

    if len(texto.split()) <= palabras_max:
        spans = [(0, len(texto))]
    else:
        spans = _ventanas(texto, palabras_max)

    fragmentos = []
    for idx, (s, e) in enumerate(spans, start=1):
        cuerpo = texto[s:e].strip()
        if not cuerpo:
            continue
        p_ini = u.pagina_marca if s == 0 else _pagina_en(s, inicios, paginas)
        p_fin = _pagina_en(max(e - 1, 0), inicios, paginas)
        fragmentos.append(Fragmento(
            texto=cuerpo,
            id_norma=0 if numerar else u.id_norma,
            etiqueta=f"Fragmento {idx}" if numerar else u.etiqueta,
            contexto=u.contexto,
            pagina_inicio=min(p_ini, p_fin),
            pagina_fin=max(p_ini, p_fin),
            parte=1 if numerar else idx,
            partes_total=1 if numerar else len(spans),
        ))
    return fragmentos


# ----------------------------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------------------------

def _advertencias_numeracion(ids: list[int]) -> str | None:
    ids = [i for i in ids if i > 0]
    if len(ids) < 2:
        return None
    esperados = set(range(min(ids), max(ids) + 1))
    faltan = sorted(esperados - set(ids))
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if not faltan and not repetidos:
        return None
    partes = []
    if faltan:
        partes.append(f"faltan {faltan[:8]}{'…' if len(faltan) > 8 else ''}")
    if repetidos:
        partes.append(f"repetidos {repetidos[:8]}{'…' if len(repetidos) > 8 else ''}")
    return "Numeración no consecutiva (" + "; ".join(partes) + "). Puede ser normal si hay normas derogadas, pero conviene revisarla."


def segmentar(
    lineas: list[tuple[int, str]],
    *,
    palabras_max: int = 500,
    palabras_min: int = 20,
) -> ResultadoSegmentacion:
    """Segmenta las líneas limpias [(pagina, texto)] en fragmentos indexables."""
    advertencias: list[str] = []
    if not lineas:
        return ResultadoSegmentacion([], "ventanas", ["No hay texto para segmentar."])

    # 1) ¿Articulado?  2) ¿Numerado?  3) Ventanas.
    esquema = "articulado"
    marcas = _resolver_familias(_clasificar(lineas, "articulado"))
    n_unidades = sum(m.tipo == "unidad" for m in marcas)
    if n_unidades < 3:
        esquema = "numerado"
        marcas = _clasificar(lineas, "numerado")
        n_unidades = sum(m.tipo == "unidad" for m in marcas)
        if n_unidades < 5:
            esquema = "ventanas"

    if esquema == "ventanas":
        unidad = _Unidad("Documento", 0, [], lineas[0][0], list(lineas))
        fragmentos = _fragmentos_de(unidad, palabras_max, numerar=True)
        advertencias.append(
            "No se detectó estructura normativa (artículos, cánones o numerales). "
            f"Se segmentó por ventanas de ~{palabras_max} palabras; las citas apuntarán a páginas, no a artículos."
        )
    else:
        unidades = _armar_unidades(lineas, marcas)
        fragmentos = []
        for u in unidades:
            # Un "Preámbulo" diminuto suele ser solo el título del documento: se descarta.
            if u.id_norma == 0 and u.etiqueta == "Preámbulo" and len(" ".join(t for _, t in u.lineas).split()) < 10:
                continue
            fragmentos.extend(_fragmentos_de(u, palabras_max))

        aviso = _advertencias_numeracion([u.id_norma for u in unidades])
        if aviso:
            advertencias.append(aviso)

    largas = sum(1 for f in fragmentos if f.partes_total > 1 and f.parte == 1)
    if largas:
        advertencias.append(f"{largas} unidad(es) superaban {palabras_max} palabras y se dividieron en partes.")
    cortos = sum(1 for f in fragmentos if len(f.texto.split()) < palabras_min)
    if cortos:
        advertencias.append(
            f"{cortos} fragmento(s) tienen menos de {palabras_min} palabras; "
            "los textos muy breves producen embeddings menos fiables."
        )

    return ResultadoSegmentacion(fragmentos=fragmentos, esquema=esquema, advertencias=advertencias)
