"""Pruebas del módulo de expansión dinámica.

Se ejecutan desde la raíz del proyecto, sin red y sin descargar el modelo SBERT:

    python -m unittest discover -s tests -v        (o:  pytest tests -v)

Los PDF de prueba se generan con reportlab (dependencia de desarrollo). La parte de
Chroma se prueba con un doble de prueba en memoria que imita la API usada.
"""
from __future__ import annotations

import io
import math
import re
import textwrap
import unittest
import zlib

import json
import sys
from pathlib import Path
from unittest import mock

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas
    from PIL import Image
    HAY_REPORTLAB = True
except ImportError:                      # pragma: no cover
    HAY_REPORTLAB = False

try:
    import pdfplumber  # noqa: F401
    HAY_PDFPLUMBER = True
except ImportError:                      # pragma: no cover
    HAY_PDFPLUMBER = False

from src.expansion_dinamica import (
    DocumentoSinContenidoError,
    NivelAutoridadInvalidoError,
    ParametrosExpansion,
    PdfExcedeLimiteError,
    PdfInvalidoError,
    PdfSinTextoNativoError,
    SesionExpansion,
    procesar_pdf,
)
from src.expansion_dinamica.columnas import Palabra, canal_dos_columnas, detectar_dos_columnas, ordenar_lectura
from src.expansion_dinamica.extractor_pdf import extraer_paginas
from src.expansion_dinamica.limpieza import (
    normalizar_unicode,
    quitar_encabezados_y_pies,
    unir_guiones,
)
from src.expansion_dinamica.segmentador import segmentar


# ======================================================================================
# Utilidades de prueba
# ======================================================================================

def pdf_bytes(paginas: list[list[str]], encabezado: str | None = None, pie: bool = True) -> bytes:
    """Genera un PDF con texto nativo; cada página es una lista de líneas."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    ancho, alto = A4
    for n, lineas in enumerate(paginas, start=1):
        if encabezado:
            c.setFont("Helvetica", 9)
            c.drawString(60, alto - 40, encabezado)
        if pie:
            c.setFont("Helvetica", 9)
            c.drawString(ancho / 2 - 35, 30, f"Página {n} de {len(paginas)}")
        c.setFont("Helvetica", 11)
        y = alto - 80
        for linea in lineas:
            c.drawString(60, y, linea)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def pdf_escaneado(n_paginas: int = 3) -> bytes:
    """PDF formado solo por imágenes (sin capa de texto)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    imagen = Image.new("RGB", (600, 800), (235, 235, 235))
    for _ in range(n_paginas):
        c.drawInlineImage(imagen, 0, 0, *A4)
        c.showPage()
    c.save()
    return buf.getvalue()


def envolver(parrafo: str, ancho: int = 80) -> list[str]:
    return textwrap.wrap(parrafo, ancho)


RELLENO = (
    "Los fieles y los pastores observarán lo dispuesto con diligencia, de modo que se cumpla "
    "la disciplina de la Iglesia y se promueva el bien de las almas en cada comunidad."
)

DECRETO = [
    # --- página 1 ---
    ["DECRETO DIOCESANO N.º 12/2024",
     "Por el cual se regula la preparación al sacramento del matrimonio en la diócesis de",
     "Ejemplo, en uso de las facultades que confiere el derecho a este obispo diocesano.",
     "TÍTULO I",
     "DISPOSICIONES GENERALES",
     "Artículo 1.- Objeto. El presente decreto regula la preparación de los fieles que desean",
     "contraer matri-",
     "monio canónico en el territorio de la diócesis, conforme al Código de Derecho Canónico.",
     *envolver("Artículo 2.- Ámbito. Las normas de este decreto se aplican a todas las parroquias "
               "de la diócesis y a los cursos prematrimoniales que en ellas se impartan. " + RELLENO),
     *envolver("Artículo 3.- Responsables. Corresponde al párroco del lugar acompañar a los novios "
               "durante todo el proceso de preparación. " + RELLENO)],
    # --- página 2 ---
    ["Artículo 4.- Documentación. Las parejas deberán presentar la documentación prevista en el",
     "artículo 2 de este decreto antes de la fecha fijada para el curso prematrimonial. El",
     "párroco verificará el cumplimiento de los requisitos exigidos a los contrayentes.",
     "CAPÍTULO I",
     "DE LOS CURSOS PREMATRIMONIALES",
     *envolver("Artículo 5.- Duración. El curso prematrimonial tendrá una duración mínima de ocho "
               "sesiones semanales de dos horas cada una. " + RELLENO),
     *envolver("Artículo 6.- Contenidos. Se abordarán la doctrina sobre el matrimonio, la "
               "espiritualidad conyugal y la educación de los hijos. " + RELLENO)],
    # --- página 3 ---
    ["TÍTULO II",
     "DISPOSICIONES FINALES",
     *envolver("Artículo 7.- Vigencia. El presente decreto entra en vigor el día de su promulgación. "
               + RELLENO),
     *envolver("Artículo 8.- Derogación. Quedan derogadas las disposiciones anteriores que se "
               "opongan a este decreto. " + RELLENO)],
]


# ======================================================================================
# Dobles de prueba de Chroma (imitan solo la API que usa SesionExpansion)
# ======================================================================================

class EmbeddingFalso:
    """Bolsa de palabras hasheada y normalizada: determinista y sin descargas."""
    DIM = 512

    def vector(self, texto: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"\w+", texto.lower()):
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def __call__(self, input):
        return [self.vector(t) for t in input]


class ColeccionFalsa:
    def __init__(self, funcion, falla_en_lote: int | None = None):
        self.funcion = funcion
        self.items: dict[str, tuple[str, dict]] = {}
        self.llamadas_upsert = 0
        self.falla_en_lote = falla_en_lote

    def upsert(self, ids, documents, metadatas):
        self.llamadas_upsert += 1
        if self.falla_en_lote == self.llamadas_upsert:
            raise RuntimeError("fallo simulado")
        for i, d, m in zip(ids, documents, metadatas):
            assert all(v is not None and isinstance(v, (str, int, float, bool)) for v in m.values()), \
                "Chroma solo admite metadatos escalares y sin None"
            self.items[i] = (d, m)

    def count(self):
        return len(self.items)

    def delete(self, ids=None, where=None):
        for i in [i for i, (_, m) in self.items.items()
                  if (ids and i in ids) or (where and all(m.get(k) == v for k, v in where.items()))]:
            del self.items[i]

    def query(self, query_texts, n_results, where=None, include=None):
        q = self.funcion([query_texts[0]])[0]
        candidatos = []
        for i, (d, m) in self.items.items():
            if where and not all(m.get(k) == v for k, v in where.items()):
                continue
            v = self.funcion([d])[0]
            candidatos.append((1 - sum(a * b for a, b in zip(q, v)), i, d, m))
        candidatos.sort(key=lambda t: t[0])
        top = candidatos[:n_results]
        return {
            "ids": [[t[1] for t in top]],
            "documents": [[t[2] for t in top]],
            "metadatas": [[t[3] for t in top]],
            "distances": [[t[0] for t in top]],
        }


class ClienteFalso:
    def __init__(self, falla_en_lote: int | None = None):
        self.colecciones: dict[str, ColeccionFalsa] = {}
        self.falla_en_lote = falla_en_lote

    def get_or_create_collection(self, name, embedding_function, metadata=None):
        assert metadata == {"hnsw:space": "cosine"}
        return self.colecciones.setdefault(name, ColeccionFalsa(embedding_function, self.falla_en_lote))

    def delete_collection(self, name):
        self.colecciones.pop(name, None)


# ======================================================================================
# Limpieza
# ======================================================================================

class TestLimpieza(unittest.TestCase):
    def test_ligaturas_y_guion_blando(self):
        self.assertEqual(normalizar_unicode("oﬁcina\u00adl"), "oficinal")

    def test_conserva_ordinal(self):
        self.assertIn("1º", normalizar_unicode("Art. 1º"))       # NFKC lo habría convertido en "1o"

    def test_une_palabras_partidas_incluso_entre_paginas(self):
        r = unir_guiones([(1, "contraer matri-"), (2, "monio canónico"), (2, "Fin.")])
        self.assertEqual(r, [(1, "contraer matrimonio canónico"), (2, "Fin.")])

    def test_conserva_el_guion_de_un_compuesto_que_el_documento_escribe_completo(self):
        lineas = [
            (1, "La situación económico-social del país es compleja."),
            (1, "Se requiere un análisis de la vida económico-"), (1, "social de las familias y de su contexto."),
            (1, "Todo ello para el matri-"), (1, "monio y la familia."),
        ]
        r = [l for _, l in unir_guiones(lineas)]
        self.assertIn("la vida económico-social de las familias", r[1])        # compuesto: se conserva el guion
        self.assertIn("el matrimonio y la familia.", r[2])                     # partición silábica: se une

    def test_compuesto_desconocido_se_une_como_palabra_partida(self):
        r = [l for _, l in unir_guiones([(1, "una teoría auto-"), (1, "organización sin otros casos.")])]
        self.assertEqual(r, ["una teoría autoorganización sin otros casos."])

    def test_no_une_si_sigue_mayuscula_o_hay_guion_de_rango(self):
        self.assertEqual(unir_guiones([(1, "Cann. 1-"), (1, "6 sobre")]), [(1, "Cann. 1-"), (1, "6 sobre")])
        self.assertEqual(len(unir_guiones([(1, "norma -"), (1, "Siguiente")])), 2)

    def test_quita_encabezado_repetido_y_numeros_de_pagina(self):
        def cuerpo(i):
            # Frases distintas en cada página, como en un documento real.
            return [f"Sentencia {chr(97 + k)}{chr(97 + i)}{chr(110 + k)} sobre una materia distinta." for k in range(6)]

        paginas = [
            ["Diócesis de Ejemplo - Decreto 12/2024", *cuerpo(i), f"Página {i} de 4"]
            for i in range(1, 5)
        ]
        limpias = quitar_encabezados_y_pies(paginas)
        for i, lineas in enumerate(limpias, start=1):
            self.assertEqual(lineas, cuerpo(i))

    def test_documento_de_dos_paginas_completas_tambien_pierde_su_encabezado(self):
        def cuerpo(i):
            return [f"Sentencia {chr(97 + k)}{chr(97 + i)}{chr(110 + k)} sobre una materia distinta." for k in range(8)]

        paginas = [["Diócesis de Ejemplo - Decreto 12/2024", *cuerpo(i), f"Página {i} de 2"] for i in (1, 2)]
        limpias = quitar_encabezados_y_pies(paginas)
        self.assertEqual(limpias, [cuerpo(1), cuerpo(2)])
        # Con una única página completa no hay evidencia de repetición: solo se quita el número.
        una = quitar_encabezados_y_pies([["Encabezado", *cuerpo(1), "7"]])
        self.assertEqual(una, [["Encabezado", *cuerpo(1)]])

    def test_paginas_muy_cortas_no_pierden_texto_por_parecerse(self):
        # Con 3 líneas por página todas serían "borde": no se debe borrar cuerpo repetido.
        paginas = [["Cuerpo de la página.", "Texto único A.", "Texto único B."] for _ in range(5)]
        self.assertEqual(quitar_encabezados_y_pies(paginas), paginas)

    def test_la_ultima_pagina_corta_tambien_pierde_encabezado_y_pie(self):
        def cuerpo(i, n=8):
            return [f"Sentencia {chr(97 + k)}{chr(97 + i)}{chr(110 + k)} sobre una materia distinta." for k in range(n)]

        paginas = [["Encabezado fijo", *cuerpo(i), f"Página {i} de 4"] for i in range(1, 4)]
        paginas.append(["Encabezado fijo", "Cierre del documento.", "Página 4 de 4"])      # página final casi vacía
        limpias = quitar_encabezados_y_pies(paginas)
        self.assertEqual(limpias[-1], ["Cierre del documento."])

    def test_documento_corto_solo_pierde_numeros_de_pagina(self):
        r = quitar_encabezados_y_pies([["Título repetido", "Texto A.", "3"], ["Título repetido", "Texto B."]])
        self.assertEqual(r[0], ["Título repetido", "Texto A."])
        self.assertEqual(r[1], ["Título repetido", "Texto B."])


# ======================================================================================
# Segmentación (sin PDF)
# ======================================================================================

def lineas_de(*textos, pagina=1):
    return [(pagina, t) for t in textos]


class TestSegmentador(unittest.TestCase):
    def test_articulado_con_contexto_jerarquico(self):
        lineas = lineas_de(
            "TÍTULO I", "DE LAS NORMAS GENERALES",
            "CAPÍTULO I", "Artículo 1.- Los fieles deben obediencia a su pastor.",
            "Artículo 2.- El obispo gobierna la diócesis.",
            "CAPÍTULO II", "Artículo 3.- Los párrocos colaboran con el obispo.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.esquema, "articulado")
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3])
        self.assertEqual([f.etiqueta for f in r.fragmentos], ["Artículo 1", "Artículo 2", "Artículo 3"])
        self.assertEqual(r.fragmentos[0].contexto, ["TÍTULO I – DE LAS NORMAS GENERALES", "CAPÍTULO I"])
        self.assertEqual(r.fragmentos[2].contexto, ["TÍTULO I – DE LAS NORMAS GENERALES", "CAPÍTULO II"])
        # El texto no incluye la etiqueta (igual que el corpus base).
        self.assertTrue(r.fragmentos[0].texto.startswith("Los fieles"))

    def test_un_titulo_nuevo_reinicia_los_niveles_inferiores(self):
        lineas = lineas_de(
            "TÍTULO I", "CAPÍTULO I", "Artículo 1.- Uno uno uno.", "Artículo 2.- Dos dos dos.",
            "TÍTULO II", "Artículo 3.- Tres tres tres.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.fragmentos[2].contexto, ["TÍTULO II"])

    def test_titulo_partido_cuya_segunda_linea_lleva_referencia_en_minusculas(self):
        lineas = lineas_de(
            "TÍTULO II – DE LOS DELITOS CONTRA LAS AUTORIDADES ECLESIÁSTICAS Y EL EJERCICIO DE LOS",
            "CARGOS (Cann. 1370–1378)",
            "Can. 1370.- Quien atenta contra la persona del Romano Pontífice incurre en pena.",
            "Can. 1371.- Debe ser castigado con pena justa quien enseña una doctrina condenada.",
            "Can. 1372.- Quien recurre contra un acto del Romano Pontífice a un concilio.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1370, 1371, 1372])
        self.assertEqual(r.fragmentos[0].contexto, [
            "TÍTULO II – DE LOS DELITOS CONTRA LAS AUTORIDADES ECLESIÁSTICAS Y EL EJERCICIO DE LOS CARGOS (Cann. 1370–1378)"])

    def test_titulo_en_columna_estrecha_partido_en_tres_lineas_con_cann_al_final(self):
        # "(Cann." termina en punto, pero es una abreviatura: no cierra el título.
        lineas = lineas_de(
            "TÍTULO IV – DE LOS ACTOS", "ADMINISTRATIVOS SINGULARES (Cann.", "35–93)",
            "CAPÍTULO I – NORMAS COMUNES (Cann.", "35–47)",
            "Can. 35.- Un acto administrativo singular puede darse por decreto o precepto.",
            "Can. 36.- El acto administrativo se entiende según el significado de las palabras.",
            "Can. 37.- El acto administrativo que afecta al fuero externo debe consignarse por escrito.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [35, 36, 37])
        self.assertEqual(r.fragmentos[0].contexto, [
            "TÍTULO IV – DE LOS ACTOS ADMINISTRATIVOS SINGULARES (Cann. 35–93)",
            "CAPÍTULO I – NORMAS COMUNES (Cann. 35–47)"])

    def test_referencia_de_canones_partida_en_dos_lineas(self):
        lineas = lineas_de(
            "TÍTULO I – DE LAS IGLESIAS PARTICULARES Y DE LA AUTORIDAD CONSTITUIDA EN ELLAS (Cann.",
            "368–430)",
            "CAPÍTULO I – DE LAS IGLESIAS PARTICULARES (Cann. 368-374)",
            "Can. 368.- Las Iglesias particulares son sobre todo las diócesis.",
            "Can. 369.- La diócesis es una porción del pueblo de Dios.",
            "Can. 370.- La prelatura territorial es una porción determinada.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [368, 369, 370])     # sin unidad fantasma "368–430)"
        self.assertEqual(r.fragmentos[0].contexto[0],
                         "TÍTULO I – DE LAS IGLESIAS PARTICULARES Y DE LA AUTORIDAD CONSTITUIDA EN ELLAS (Cann. 368–430)")

    def test_art_como_subdivision_dentro_de_un_codigo_de_canones(self):
        # En el CIC "Art. 1 – ..." es un encabezado; las unidades son los cánones ("Can.").
        lineas = lineas_de(
            "CAPÍTULO I – DE LA PROVISIÓN DE UN OFICIO ECLESIÁSTICO (Cann. 146-183)",
            "Art. 1 – DE LA LIBRE COLACIÓN (Can. 157)", "Can. 157.- Si no se establece otra cosa, compete al Obispo.",
            "Art. 2 – DE LA PRESENTACIÓN (Cann. 158-163)", "Can. 158.- La presentación se hace dentro de tres meses.",
            "Can. 159.- Nadie puede ser presentado sin ser idóneo.", "Can. 160.- La presentación puede ser revocada.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [157, 158, 159, 160])
        self.assertEqual(r.fragmentos[0].contexto[-1], "Art. 1 – DE LA LIBRE COLACIÓN (Can. 157)")
        self.assertEqual(r.fragmentos[2].contexto[-1], "Art. 2 – DE LA PRESENTACIÓN (Cann. 158-163)")
        self.assertTrue(all(f.etiqueta.startswith("Canon") for f in r.fragmentos))

    def test_dos_familias_con_peso_parecido_no_se_adivinan(self):
        lineas = lineas_de("Can. 1.- Uno uno.", "Can. 2.- Dos dos.", "Artículo 3.- Tres tres.", "Artículo 4.- Cuatro cuatro.")
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4])

    def test_subtitulo_sin_palabra_clave_antes_de_un_articulo_va_al_contexto(self):
        lineas = lineas_de(
            "CAPÍTULO I", "Can. 1.- Primera norma que se establece.",
            "DE LA LIBRE COLACIÓN (Can. 2)", "Can. 2.- Segunda norma que se establece.",
            "DE LA PRESENTACIÓN", "(Cann. 3-4)", "Can. 3.- Tercera norma que se establece.",
            "CAPÍTULO II", "Can. 4.- Cuarta norma que se establece.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4])
        self.assertEqual(r.fragmentos[0].texto, "Primera norma que se establece.")        # no se le pega el subtítulo
        self.assertEqual(r.fragmentos[1].contexto, ["CAPÍTULO I", "DE LA LIBRE COLACIÓN (Can. 2)"])
        self.assertEqual(r.fragmentos[2].contexto, ["CAPÍTULO I", "DE LA PRESENTACIÓN (Cann. 3-4)"])
        self.assertEqual(r.fragmentos[3].contexto, ["CAPÍTULO II"])                         # un CAPÍTULO cierra el subtítulo

    def test_linea_en_mayusculas_dentro_del_texto_no_es_subtitulo(self):
        lineas = lineas_de(
            "Can. 1.- Primera norma. El texto continúa con una expresión como",
            "SANTA SEDE", "y sigue la frase hasta cerrarse con su punto final.",
            "Can. 2.- Segunda norma.", "Can. 3.- Tercera norma.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertIn("SANTA SEDE", r.fragmentos[0].texto)
        self.assertEqual(r.fragmentos[0].contexto, [])

    def test_encabezado_con_guion_antes_del_numero(self):
        lineas = lineas_de("TÍTULO – V DE LAS ASOCIACIONES DE FIELES (Cann. 298–329)",
                           "Can. 298.- Los fieles pueden asociarse.", "Can. 299.- Tienen derecho a crear asociaciones.",
                           "Can. 300.- Ninguna asociación asumirá el nombre de católica.")
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.fragmentos[0].contexto, ["TÍTULO – V DE LAS ASOCIACIONES DE FIELES (Cann. 298–329)"])

    def test_libro_y_parte_conviven_en_la_jerarquia(self):
        lineas = lineas_de(
            "LIBRO II: DEL PUEBLO DE DIOS (Cann. 204 – 746)", "PARTE I: DE LOS FIELES CRISTIANOS (Cann. 204–329)",
            "TÍTULO I – DE LAS OBLIGACIONES Y DERECHOS", "Can. 204.- Los fieles son quienes han sido incorporados a Cristo.",
            "Can. 205.- Se encuentran en plena comunión los bautizados.",
            "PARTE II: DE LA CONSTITUCIÓN JERÁRQUICA", "Can. 330.- Así como permanecen el oficio de Pedro.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.fragmentos[0].contexto[:3], [
            "LIBRO II: DEL PUEBLO DE DIOS (Cann. 204 – 746)", "PARTE I: DE LOS FIELES CRISTIANOS (Cann. 204–329)",
            "TÍTULO I – DE LAS OBLIGACIONES Y DERECHOS"])
        # Al cambiar de PARTE se cierra el TÍTULO de la anterior, pero se conserva el LIBRO.
        self.assertEqual(r.fragmentos[2].contexto, [
            "LIBRO II: DEL PUEBLO DE DIOS (Cann. 204 – 746)", "PARTE II: DE LA CONSTITUCIÓN JERÁRQUICA"])

    def test_la_jerarquia_se_aprende_del_documento(self):
        # Estilo Código de Derecho Canónico: SECCIÓN encima del CAPÍTULO.
        canonico = lineas_de("SECCIÓN I", "CAPÍTULO I", "Can. 1.- Uno.", "CAPÍTULO II", "Can. 2.- Dos.",
                             "SECCIÓN II", "Can. 3.- Tres.")
        r = segmentar(canonico, palabras_min=1)
        self.assertEqual([f.contexto for f in r.fragmentos], [["SECCIÓN I", "CAPÍTULO I"], ["SECCIÓN I", "CAPÍTULO II"], ["SECCIÓN II"]])
        # Estilo ley civil: SECCIÓN debajo del CAPÍTULO.
        civil = lineas_de("CAPÍTULO I", "SECCIÓN 1", "Artículo 1.- Uno.", "SECCIÓN 2", "Artículo 2.- Dos.",
                          "CAPÍTULO II", "Artículo 3.- Tres.")
        r = segmentar(civil, palabras_min=1)
        self.assertEqual([f.contexto for f in r.fragmentos], [["CAPÍTULO I", "SECCIÓN 1"], ["CAPÍTULO I", "SECCIÓN 2"], ["CAPÍTULO II"]])

    def test_encabezado_largo_con_cann_en_minuscula_y_partido_en_dos_lineas(self):
        lineas = lineas_de(
            "LIBRO I: DE LAS NORMAS GENERALES (Cann. 1 – 203) > TÍTULO IV – DE LOS ACTOS ADMINISTRATIVOS SINGULARES (Cann. 35–93)",
            "Can. 35.- Un acto administrativo singular puede darse por decreto o por rescripto.",
            "Can. 36.- El acto administrativo debe entenderse según el significado propio de las palabras.",
            "TÍTULO V – DE LOS ESTATUTOS Y REGLAMENTOS",
            "(Cann. 94–95) Y DE OTRAS NORMAS PARTICULARES",
            "Can. 37.- Los estatutos obligan solo a las personas jurídicas que se rigen por ellos.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [35, 36, 37])          # sin "Texto sin numerar"
        self.assertTrue(r.fragmentos[0].contexto[0].startswith("LIBRO I"))
        self.assertIn("(Cann. 94–95) Y DE OTRAS NORMAS PARTICULARES", r.fragmentos[2].contexto[-1])

    def test_cita_en_medio_de_oracion_no_es_un_nuevo_articulo(self):
        lineas = lineas_de(
            "Artículo 1.- Primera norma que se establece.",
            "Artículo 2.- Se presentará la documentación prevista en el",
            "artículo 1 de este decreto antes de la fecha fijada.",
            "Artículo 3.- Tercera norma.",
            "Artículo 4.- Cuarta norma.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4])
        self.assertIn("artículo 1 de este decreto", r.fragmentos[1].texto)

    def test_marca_debil_con_mayuscula_se_rechaza_si_la_oracion_no_habia_cerrado(self):
        lineas = lineas_de(
            "Artículo 1.- Primera.", "Artículo 2.- Segunda y se aplicará según lo previsto en",
            "Artículo 7 de la presente ley, que regula la materia.",
            "Artículo 3.- Tercera.", "Artículo 4.- Cuarta.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4])

    def test_canon_y_ordinales_en_palabras(self):
        lineas = lineas_de("Can. 1.- Norma uno.", "Can. 2.- Norma dos.", "Canon 3.- Norma tres.")
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.etiqueta for f in r.fragmentos], ["Canon 1", "Canon 2", "Canon 3"])

        lineas = lineas_de("ARTÍCULO PRIMERO.- Uno.", "ARTÍCULO SEGUNDO.- Dos.", "ARTÍCULO TERCERO.- Tres.")
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3])

    def test_preambulo_con_contenido_se_conserva_y_titulo_suelto_se_descarta(self):
        con_texto = lineas_de(
            "Considerando que la preparación al matrimonio es una necesidad pastoral urgente, y que "
            "conviene unificar los criterios en toda la diócesis, se decreta lo siguiente.",
            "Artículo 1.- Uno.", "Artículo 2.- Dos.", "Artículo 3.- Tres.")
        r = segmentar(con_texto, palabras_min=1)
        self.assertEqual(r.fragmentos[0].etiqueta, "Preámbulo")
        self.assertEqual(r.fragmentos[0].id_norma, 0)

        solo_titulo = lineas_de("DECRETO 12/2024", "Artículo 1.- Uno.", "Artículo 2.- Dos.", "Artículo 3.- Tres.")
        r = segmentar(solo_titulo, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3])

    def test_unidad_larga_se_parte_en_ventanas_homogeneas_y_con_paginas(self):
        oraciones = [f"Esta es la oración número {i} de un artículo muy extenso que no termina." for i in range(120)]
        # 120 oraciones x 12 palabras ≈ 1440 palabras → 3 ventanas de ≤ 500
        lineas = [(1, "Artículo 1.- Introducción breve.")] + [(2 + i // 40, o) for i, o in enumerate(oraciones)]
        lineas += lineas_de("Artículo 2.- Dos.", "Artículo 3.- Tres.", pagina=5)
        r = segmentar(lineas, palabras_max=500, palabras_min=1)
        art1 = [f for f in r.fragmentos if f.id_norma == 1]
        self.assertGreaterEqual(len(art1), 3)
        self.assertTrue(all(len(f.texto.split()) <= 500 for f in art1))
        self.assertEqual([f.parte for f in art1], list(range(1, len(art1) + 1)))
        self.assertTrue(all(f.partes_total == len(art1) for f in art1))
        self.assertEqual(art1[0].pagina_inicio, 1)
        self.assertGreaterEqual(art1[-1].pagina_inicio, 3)       # las ventanas posteriores avanzan de página
        # Homogéneas: la más corta no es una "migaja" respecto de la más larga.
        tamanos = [len(f.texto.split()) for f in art1]
        self.assertGreater(min(tamanos), max(tamanos) * 0.5)
        self.assertTrue(any("se dividieron en partes" in a for a in r.advertencias))

    def test_numerado_creciente_ignora_sublistas(self):
        lineas = lineas_de(
            "1. La Iglesia es una comunidad de fieles.", "2. El Papa es el sucesor de Pedro.",
            "3. Los obispos gobiernan sus diócesis:", "1. primera subdivisión interna.",
            "4. Los presbíteros cooperan con el obispo.", "5. Los diáconos sirven al pueblo de Dios.",
            "6. Los laicos participan de la misión.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.esquema, "numerado")
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4, 5, 6])
        self.assertIn("1. primera subdivisión", r.fragmentos[2].texto)

    def test_numeral_siguiente_se_acepta_aunque_el_anterior_no_cierre_con_punto(self):
        lineas = lineas_de(
            "1. La Iglesia es una comunidad de fieles.", "2. El Papa es el sucesor de Pedro y de los apóstoles",
            "3. Los obispos gobiernan sus diócesis.", "4. Los presbíteros cooperan con el obispo.",
            "5. Los diáconos sirven al pueblo de Dios.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4, 5])

    def test_numeral_esperado_cuyo_texto_empieza_con_parrafo_o_inciso(self):
        lineas = lineas_de(
            "20. Las transmisiones radiofónicas de acciones sagradas son importantes.",
            "21. Para que el pueblo cristiano obtenga con mayor seguridad las gracias abundantes.",
            "22. §1. La reglamentación de la sagrada Liturgia es de competencia de la Iglesia.",
            "23. a) Los Obispos, como sucesores de los Apóstoles, tienen por sí mismos el poder.",
            "24. Para conservar la sana tradición y abrir el camino a un progreso legítimo.",
            "25. Los libros litúrgicos serán revisados lo antes posible.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [20, 21, 22, 23, 24, 25])
        self.assertTrue(r.fragmentos[2].texto.startswith("§1."))

    def test_primer_numeral_alto_solo_si_le_sigue_el_consecutivo(self):
        # Un número suelto al inicio ("2024. ...") no arrastra la detección...
        sueltos = lineas_de("2024. Año de publicación del documento original.", "Texto corrido sin numerales.",
                            "Más texto corrido que no tiene ninguna enumeración.")
        self.assertEqual(segmentar(sueltos, palabras_min=1).esquema, "ventanas")
        # ...pero una serie que empieza en el 20 sí es válida.
        serie = lineas_de(*[f"{n}. Texto del numeral número {n} con contenido suficiente." for n in range(20, 26)])
        r = segmentar(serie, palabras_min=1)
        self.assertEqual((r.esquema, [f.id_norma for f in r.fragmentos]), ("numerado", [20, 21, 22, 23, 24, 25]))

    def test_primer_numeral_tras_un_titulo_en_mayusculas_sin_punto(self):
        lineas = lineas_de(
            "NOSTRA AETATE", "1. En nuestra época el género humano se une cada vez más estrechamente.",
            "2. Ya desde la antigüedad se encuentra en los pueblos una cierta percepción.",
            "3. La Iglesia católica nada rechaza de lo que en estas religiones hay de santo.",
            "4. Al escrutar el misterio de la Iglesia se recuerda el vínculo con Abraham.",
            "5. No podemos invocar a Dios como Padre si nos negamos a tratar como hermanos.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual(r.esquema, "numerado")
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4, 5])
        self.assertEqual(r.fragmentos[0].contexto, ["NOSTRA AETATE"])

    def test_un_numero_que_no_es_el_siguiente_exige_oracion_cerrada(self):
        lineas = lineas_de(
            "1. La Iglesia es una comunidad.", "2. Los obispos gobiernan sus diócesis según lo previsto en el",
            "9. Concilio Vaticano II, que lo estableció así.",           # no es el siguiente (3) y la oración no cerró
            "3. Los presbíteros cooperan.", "4. Los diáconos sirven.", "5. Los laicos participan.",
        )
        r = segmentar(lineas, palabras_min=1)
        self.assertEqual([f.id_norma for f in r.fragmentos], [1, 2, 3, 4, 5])
        self.assertIn("9. Concilio Vaticano II", r.fragmentos[1].texto)

    def test_sin_estructura_cae_a_ventanas_con_advertencia(self):
        parrafo = "Texto corrido sin ninguna marca normativa que permita partirlo. " * 3
        lineas = [(1 + i // 30, parrafo) for i in range(90)]
        r = segmentar(lineas, palabras_max=500, palabras_min=1)
        self.assertEqual(r.esquema, "ventanas")
        self.assertGreater(len(r.fragmentos), 1)
        self.assertTrue(all(len(f.texto.split()) <= 500 for f in r.fragmentos))
        self.assertTrue(all(f.id_norma == 0 for f in r.fragmentos))
        self.assertEqual([f.etiqueta for f in r.fragmentos][:2], ["Fragmento 1", "Fragmento 2"])
        self.assertTrue(any("No se detectó estructura" in a for a in r.advertencias))

    def test_oracion_gigante_sin_puntuacion_tambien_se_parte(self):
        lineas = [(1, " ".join(["palabra"] * 1300))]
        r = segmentar(lineas, palabras_max=500, palabras_min=1)
        self.assertTrue(all(len(f.texto.split()) <= 500 for f in r.fragmentos))
        self.assertEqual(sum(len(f.texto.split()) for f in r.fragmentos), 1300)

    def test_advierte_numeracion_con_huecos(self):
        lineas = lineas_de("Artículo 1.- Uno.", "Artículo 2.- Dos.", "Artículo 5.- Cinco.", "Artículo 6.- Seis.")
        r = segmentar(lineas, palabras_min=1)
        self.assertTrue(any("faltan [3, 4]" in a for a in r.advertencias))

    def test_entrada_vacia(self):
        self.assertEqual(segmentar([]).fragmentos, [])


# ======================================================================================
# De extremo a extremo con PDFs reales
# ======================================================================================

@unittest.skipUnless(HAY_REPORTLAB, "reportlab/Pillow no instalados (dependencias de desarrollo)")
class TestPdfExtremoAExtremo(unittest.TestCase):
    def test_decreto_completo(self):
        datos = pdf_bytes(DECRETO, encabezado="Diócesis de Ejemplo - Decreto 12/2024")
        doc = procesar_pdf(datos, nombre="Decreto 12/2024", params=ParametrosExpansion(palabras_min_chunk=1))
        seg = doc.segmentacion
        self.assertEqual(seg.esquema, "articulado")
        self.assertEqual(doc.total_paginas, 3)

        ids = [f.id_norma for f in seg.fragmentos if f.id_norma]
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6, 7, 8])

        todo = " ".join(f.texto for f in seg.fragmentos)
        self.assertNotIn("Diócesis de Ejemplo - Decreto", todo)        # encabezado fuera
        self.assertNotRegex(todo, r"Página \d de 3")                    # pie fuera
        self.assertIn("matrimonio canónico", todo)                      # "matri-/monio" unido
        self.assertNotIn("matri-", todo)

        art4 = next(f for f in seg.fragmentos if f.id_norma == 4)
        self.assertIn("artículo 2 de este decreto", art4.texto)         # cita interna NO partió el artículo
        self.assertEqual(art4.pagina_inicio, 2)
        art2 = next(f for f in seg.fragmentos if f.id_norma == 2)
        self.assertEqual(art2.contexto, ["TÍTULO I – DISPOSICIONES GENERALES"])
        art5 = next(f for f in seg.fragmentos if f.id_norma == 5)
        self.assertEqual(art5.contexto, ["TÍTULO I – DISPOSICIONES GENERALES", "CAPÍTULO I – DE LOS CURSOS PREMATRIMONIALES"])
        art7 = next(f for f in seg.fragmentos if f.id_norma == 7)
        self.assertEqual(art7.contexto, ["TÍTULO II – DISPOSICIONES FINALES"])
        self.assertEqual(art7.pagina_inicio, 3)

    def test_pdf_sin_estructura(self):
        parrafo = envolver("Texto corrido sin marcas normativas que permita separar unidades lógicas. " * 12)
        paginas = [parrafo * 3 for _ in range(4)]
        doc = procesar_pdf(pdf_bytes(paginas), params=ParametrosExpansion())
        self.assertEqual(doc.segmentacion.esquema, "ventanas")
        self.assertTrue(all(len(f.texto.split()) <= 500 for f in doc.segmentacion.fragmentos))

    def test_nombre_se_toma_del_archivo(self):
        class Subido(io.BytesIO):
            name = "decreto_diocesano.pdf"
        doc = procesar_pdf(Subido(pdf_bytes(DECRETO)))
        self.assertEqual(doc.nombre, "decreto_diocesano")

    def test_escaneado_se_rechaza_sin_ocr(self):
        with self.assertRaises(PdfSinTextoNativoError) as ctx:
            procesar_pdf(pdf_escaneado())
        self.assertIn("no aplica OCR", str(ctx.exception))

    def test_pdf_mixto_con_pocas_paginas_vacias_se_acepta_con_advertencia(self):
        paginas = [DECRETO[0], DECRETO[1], DECRETO[2], []]
        doc = procesar_pdf(pdf_bytes(paginas), params=ParametrosExpansion(palabras_min_chunk=1))
        self.assertTrue(any("poco o ningún texto" in a for a in doc.advertencias))

    def test_archivos_invalidos(self):
        with self.assertRaises(PdfInvalidoError):
            procesar_pdf(b"")
        with self.assertRaises(PdfInvalidoError):
            procesar_pdf(b"esto no es un pdf")
        with self.assertRaises(PdfInvalidoError):
            procesar_pdf(b"%PDF-1.4\nbasura sin estructura valida %%EOF")
        with self.assertRaises(PdfInvalidoError):
            procesar_pdf("/ruta/que/no/existe.pdf")

    def test_limites_de_tamano_y_paginas(self):
        datos = pdf_bytes(DECRETO)
        with self.assertRaises(PdfExcedeLimiteError):
            procesar_pdf(datos, params=ParametrosExpansion(max_paginas=2))
        with self.assertRaises(PdfExcedeLimiteError):
            procesar_pdf(datos, params=ParametrosExpansion(max_mb=0.001))

    def test_documento_sin_contenido_tras_limpiar(self):
        # Texto presente pero que se reduce a solo encabezado/pie → nada utilizable.
        paginas = [["x"] for _ in range(3)]
        with self.assertRaises((DocumentoSinContenidoError, PdfSinTextoNativoError)):
            procesar_pdf(pdf_bytes(paginas, encabezado="Encabezado", pie=True))


# ======================================================================================
# Sesión (Chroma simulado)
# ======================================================================================

CLAVES_CONTRATO_BASE = {
    "id_chroma", "texto", "similitud", "contexto", "fuente", "id_norma",
    "tipo_corpus", "categoria", "nivel_autoridad", "url_origen", "cita",
}


@unittest.skipUnless(HAY_REPORTLAB, "reportlab/Pillow no instalados (dependencias de desarrollo)")
class TestSesion(unittest.TestCase):
    def setUp(self):
        self.cliente = ClienteFalso()
        self.params = ParametrosExpansion(palabras_min_chunk=1)
        self.sesion = SesionExpansion(
            params=self.params, funcion_embedding=EmbeddingFalso(), cliente=self.cliente, sesion_id="prueba01"
        )
        self.pdf = pdf_bytes(DECRETO, encabezado="Diócesis de Ejemplo - Decreto 12/2024")

    def test_carga_y_resumen(self):
        r = self.sesion.cargar_pdf(self.pdf, nombre="Decreto 12/2024")
        self.assertEqual((r.paginas, r.fragmentos, r.esquema), (3, 9, "articulado"))   # 8 artículos + preámbulo
        self.assertEqual(r.nivel_autoridad, 0.33)                        # Nivel 1 por defecto (norma local)
        self.assertEqual(self.sesion.total_fragmentos, 9)
        self.assertFalse(r.ya_cargado)

    def test_aislamiento_solo_coleccion_efimera_propia(self):
        self.sesion.cargar_pdf(self.pdf)
        self.assertEqual(list(self.cliente.colecciones), ["exp_prueba01"])
        self.assertNotEqual(self.sesion.nombre_coleccion, "corpus_canonico")
        metas = [m for _, m in self.cliente.colecciones["exp_prueba01"].items.values()]
        self.assertTrue(all(m["tipo_corpus"] == "expansion_dinamica" for m in metas))

    def test_contrato_compatible_con_el_corpus_base(self):
        self.sesion.cargar_pdf(self.pdf, nombre="Decreto 12/2024", url_origen="https://ejemplo.org/d12.pdf")
        res = self.sesion.consultar("documentación prevista para el curso prematrimonial", n_resultados=3)
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertTrue(CLAVES_CONTRATO_BASE <= set(r), CLAVES_CONTRATO_BASE - set(r))
            self.assertEqual(r["origen"], "expansion_dinamica")
            self.assertEqual(r["url_origen"], "https://ejemplo.org/d12.pdf")
            self.assertTrue(0 <= r["similitud"] <= 1)
        top = res[0]
        self.assertEqual(top["id_norma"], 4)
        self.assertEqual(top["cita"], "Decreto 12/2024, Artículo 4, p. 2")
        self.assertTrue(top["contexto"].startswith("Documento cargado > Decreto 12/2024"))
        self.assertEqual(top["nivel_autoridad"], 0.33)
        self.assertEqual([r["similitud"] for r in res], sorted((r["similitud"] for r in res), reverse=True))

    def test_url_ausente_usa_el_mismo_texto_que_el_corpus_base(self):
        self.sesion.cargar_pdf(self.pdf)
        self.assertEqual(self.sesion.consultar("matrimonio", 1)[0]["url_origen"], "Sin url registrada")

    def test_nivel_de_autoridad_configurable_y_validado(self):
        r = self.sesion.cargar_pdf(self.pdf, nivel_autoridad=0.66)
        self.assertEqual(r.nivel_autoridad, 0.66)
        otra = SesionExpansion(params=self.params, funcion_embedding=EmbeddingFalso(), cliente=ClienteFalso())
        for malo in (0.5, 0, 2, -1):
            with self.assertRaises(NivelAutoridadInvalidoError):
                otra.cargar_pdf(self.pdf, nivel_autoridad=malo)

    def test_mismo_pdf_dos_veces_no_duplica(self):
        self.sesion.cargar_pdf(self.pdf)
        r2 = self.sesion.cargar_pdf(self.pdf)
        self.assertTrue(r2.ya_cargado)
        self.assertEqual(self.cliente.colecciones["exp_prueba01"].count(), 9)

    def test_fallo_a_mitad_de_carga_no_deja_nada_a_medias(self):
        cliente = ClienteFalso(falla_en_lote=2)
        sesion = SesionExpansion(
            params=ParametrosExpansion(palabras_min_chunk=1, batch_size=3),
            funcion_embedding=EmbeddingFalso(), cliente=cliente, sesion_id="falla")
        with self.assertRaises(RuntimeError):
            sesion.cargar_pdf(self.pdf)
        self.assertEqual(cliente.colecciones["exp_falla"].count(), 0)
        self.assertEqual(sesion.documentos(), [])

    def test_pdf_invalido_no_crea_coleccion(self):
        with self.assertRaises(PdfSinTextoNativoError):
            self.sesion.cargar_pdf(pdf_escaneado())
        self.assertEqual(self.cliente.colecciones, {})

    def test_eliminar_documento_y_cerrar(self):
        r = self.sesion.cargar_pdf(self.pdf)
        self.assertTrue(self.sesion.eliminar_documento(r.doc_id))
        self.assertFalse(self.sesion.eliminar_documento(r.doc_id))
        self.assertEqual(self.sesion.consultar("matrimonio"), [])
        self.sesion.cargar_pdf(self.pdf)
        self.sesion.cerrar()
        self.assertEqual(self.cliente.colecciones, {})
        self.sesion.cerrar()                                             # idempotente

    def test_context_manager_limpia_al_salir(self):
        with SesionExpansion(params=self.params, funcion_embedding=EmbeddingFalso(), cliente=self.cliente,
                             sesion_id="ctx") as s:
            s.cargar_pdf(self.pdf)
            self.assertIn("exp_ctx", self.cliente.colecciones)
        self.assertNotIn("exp_ctx", self.cliente.colecciones)

    def test_consultas_invalidas_devuelven_lista_vacia(self):
        self.sesion.cargar_pdf(self.pdf)
        for premisa in ("", "   ", None, 123):
            self.assertEqual(self.sesion.consultar(premisa), [])
        self.assertEqual(self.sesion.consultar("matrimonio", 0), [])
        self.assertEqual(self.sesion.consultar("matrimonio", "abc"), [])

    def test_consulta_combinada_ordena_base_y_expansion_por_similitud(self):
        self.sesion.cargar_pdf(self.pdf, nombre="Decreto 12/2024")

        def base_falsa(premisa, n, where):
            self.assertIsNone(where)
            return [
                {"id_chroma": "corpus_canones:0001", "texto": "Can. 1055", "similitud": 0.99, "cita": "CIC, canon 1055"},
                {"id_chroma": "corpus_canones:0002", "texto": "Can. 1056", "similitud": 0.01, "cita": "CIC, canon 1056"},
            ]

        res = self.sesion.consultar_combinado("documentación curso prematrimonial", n_resultados=4,
                                              consultar_base=base_falsa)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0]["origen"], "corpus_base")
        self.assertEqual(res[0]["cita"], "CIC, canon 1055")
        self.assertIn("expansion_dinamica", [r["origen"] for r in res])
        sims = [r["similitud"] for r in res]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_combinado_sin_documentos_equivale_al_corpus_base(self):
        base = [{"id_chroma": "a", "similitud": 0.5}, {"id_chroma": "b", "similitud": 0.4}]
        res = self.sesion.consultar_combinado("x", 5, consultar_base=lambda p, n, w: [dict(r) for r in base])
        self.assertEqual([r["id_chroma"] for r in res], ["a", "b"])
        self.assertEqual(self.cliente.colecciones, {})                   # ni siquiera crea colección

# ======================================================================================
# Dos columnas: geometría pura (sin PDF)
# ======================================================================================

def _linea(texto, x, y, alto=10):
    """Palabras de una línea que empieza en x (≈5.5 pt por letra y 4 pt entre palabras)."""
    palabras, cx = [], x
    for w in texto.split():
        ancho = 5.5 * len(w)
        palabras.append(Palabra(w, cx, cx + ancho, y, y + alto))
        cx += ancho + 4
    return palabras


def _columna(textos, x, y0, paso=14):
    return [p for i, t in enumerate(textos) for p in _linea(t, x, y0 + i * paso)]


def _textos(prefijo, n, palabras_por_linea=7):
    return [" ".join(f"{prefijo}{i}p{j}" for j in range(palabras_por_linea)) for i in range(n)]


def _dos_columnas(n=30, titulo=None, pie=None):
    ps = _columna(_textos("L", n), 60, 100) + _columna(_textos("R", n), 330, 100)
    if titulo:
        ps += _linea(titulo, 150, 60)
    if pie:
        ps += _linea(pie, 250, 780)
    return ps


def _primeras(lineas):
    return [l.split()[0] for l in lineas]


class TestColumnasGeometria(unittest.TestCase):
    def test_izquierda_completa_y_luego_derecha(self):
        r = ordenar_lectura(_dos_columnas(30))
        self.assertEqual(_primeras(r), [f"L{i}p0" for i in range(30)] + [f"R{i}p0" for i in range(30)])

    def test_titulo_y_pie_a_todo_el_ancho_van_al_principio_y_al_final(self):
        r = ordenar_lectura(_dos_columnas(30, titulo="DECRETO DIOCESANO N.º 12/2024 sobre la preparación", pie="Página 1 de 3"))
        self.assertTrue(r[0].startswith("DECRETO"))
        self.assertTrue(r[1].startswith("L0"))
        self.assertTrue(r[31].startswith("R0"))
        self.assertEqual(r[-1], "Página 1 de 3")

    def test_una_sola_palabra_ancha_centrada_cuenta_como_fila_a_todo_el_ancho(self):
        ps = _dos_columnas(30) + [Palabra("DECRETO", 230, 400, 60, 75)]
        self.assertEqual(ordenar_lectura(ps)[0], "DECRETO")

    def test_encabezado_de_seccion_a_mitad_de_pagina_divide_en_bandas(self):
        ps = _columna(_textos("A", 10), 60, 100) + _columna(_textos("B", 10), 330, 100)
        ps += _linea("CAPÍTULO II DE LOS CURSOS PREMATRIMONIALES Y SU DURACIÓN EN LA DIÓCESIS", 60, 100 + 10 * 14)
        ps += _columna(_textos("C", 10), 60, 100 + 11 * 14) + _columna(_textos("D", 10), 330, 100 + 11 * 14)
        esperado = ([f"A{i}p0" for i in range(10)] + [f"B{i}p0" for i in range(10)] + ["CAPÍTULO"]
                    + [f"C{i}p0" for i in range(10)] + [f"D{i}p0" for i in range(10)])
        self.assertEqual(_primeras(ordenar_lectura(ps)), esperado)

    def test_columnas_desiguales_40_60_y_derecha_mas_corta(self):
        ps = _columna(_textos("L", 30, 5), 50, 100) + _columna(_textos("R", 22, 8), 270, 100)
        self.assertEqual(_primeras(ordenar_lectura(ps)), [f"L{i}p0" for i in range(30)] + [f"R{i}p0" for i in range(22)])

    def test_pagina_dispersa_con_canal_conocido_la_derecha_corta_se_lee_en_orden(self):
        # Última página típica: columna izquierda llena y derecha con solo 3 líneas.
        canal = canal_dos_columnas(_dos_columnas(30))
        self.assertIsNotNone(canal)
        ps = _columna(_textos("L", 30), 60, 100) + _columna(_textos("R", 3), 330, 100)
        self.assertIsNone(ordenar_lectura(ps))                                  # sin canal: criterio estricto, no alcanza
        self.assertEqual(_primeras(ordenar_lectura(ps, canal)),
                         [f"L{i}p0" for i in range(30)] + [f"R{i}p0" for i in range(3)])

    def test_criterio_tolerante_no_acepta_firma_ni_titulo_alineados_a_la_derecha(self):
        canal = canal_dos_columnas(_dos_columnas(30))
        # Firma en la parte baja derecha, debajo del texto de una sola columna: no comparte línea con nada.
        firma = _columna(_textos("U", 30, 13), 60, 100) + _columna(["Dado en Ejemplo", "El Obispo"], 330, 100 + 32 * 14)
        # Título alineado a la derecha, arriba, antes de que empiece el texto.
        titulo = _columna(["Decreto 12/2024", "Año 2024"], 330, 60) + _columna(_textos("U", 30, 13), 60, 100)
        # Página de una sola columna con un número de página suelto a la derecha.
        solo_numero = _columna(_textos("U", 30, 13), 60, 100) + [Palabra("12", 520, 531, 780, 790)]
        for nombre, palabras in [("firma abajo a la derecha", firma), ("título a la derecha arriba", titulo),
                                 ("número suelto", solo_numero)]:
            with self.subTest(nombre):
                self.assertIsNone(ordenar_lectura(palabras, canal))

    def test_pagina_a_todo_el_ancho_no_se_reordena_aunque_se_conozca_el_canal(self):
        canal = canal_dos_columnas(_dos_columnas(30))
        ancha = [p for i in range(30) for p in _linea(" ".join(f"w{i}x{j}" for j in range(13)), 60, 100 + i * 14)]
        self.assertIsNone(ordenar_lectura(ancha, canal))

    def test_casos_que_NO_son_dos_columnas_se_dejan_intactos(self):
        una = _columna(_textos("U", 40, 12), 60, 100)
        indice = (_columna([f"Capítulo {i} de los fieles cristianos y su misión en la Iglesia particular" for i in range(30)], 60, 100)
                  + [Palabra(str(10 + i), 520, 531, 100 + i * 14, 110 + i * 14) for i in range(30)])
        nota_al_margen = _columna(_textos("U", 40, 12), 150, 100) + _columna([f"n{i} x" for i in range(20)], 60, 100)
        tres = (_columna(_textos("A", 30, 4), 50, 100) + _columna(_textos("B", 30, 4), 230, 100)
                + _columna(_textos("C", 30, 4), 410, 100))
        justificado = [p for i in range(40) for p in _linea(" ".join(f"w{i}x{j}" for j in range(13)), 60, 100 + i * 14)]
        sangria = _columna(_textos("U", 30, 13), 60, 100) + _columna(_textos("S", 8, 4), 330, 100 + 22 * 14)
        casi_vacia = _dos_columnas(3)
        for nombre, palabras in [("una columna", una), ("índice con números", indice), ("nota al margen", nota_al_margen),
                                 ("tres columnas", tres), ("párrafo justificado", justificado),
                                 ("sangría inferior", sangria), ("página casi vacía", casi_vacia)]:
            with self.subTest(nombre):
                self.assertIsNone(ordenar_lectura(palabras))
                self.assertFalse(detectar_dos_columnas(palabras))


# ======================================================================================
# Dos columnas: PDFs reales
# ======================================================================================

def envolver_pt(texto: str, ancho_pt: float, fuente: str = "Helvetica", tam: int = 11) -> list[str]:
    """Envuelve por ANCHO REAL en puntos, como un editor (las mayúsculas ocupan más que las minúsculas)."""
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = (actual + " " + palabra).strip()
        if not actual or stringWidth(prueba, fuente, tam) <= ancho_pt:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def pdf_dos_columnas(lineas, *, por_columna=40, orden="filas", titulo=None, encabezado=None, pie=True) -> bytes:
    """Reparte `lineas` (ya en orden de lectura) en dos columnas: llena la izquierda, luego la derecha y pasa de página.

    orden="columnas": el PDF dibuja primero toda la columna izquierda y luego la derecha.
    orden="filas":    dibuja fila a fila alternando izquierda/derecha (el caso que mezcla el texto al extraerlo).
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    ancho, alto = A4
    por_pagina = 2 * por_columna
    paginas = [lineas[i:i + por_pagina] for i in range(0, len(lineas), por_pagina)]
    for n, pag in enumerate(paginas, start=1):
        izq, der = pag[:por_columna], pag[por_columna:]
        c.setFont("Helvetica", 9)
        if encabezado:
            c.drawString(60, alto - 40, encabezado)
        if pie:
            c.drawString(ancho / 2 - 35, 30, f"Página {n} de {len(paginas)}")
        y0 = alto - 80
        if titulo and n == 1:
            c.setFont("Helvetica-Bold", 13)
            c.drawCentredString(ancho / 2, y0, titulo)
            y0 -= 30
        c.setFont("Helvetica", 11)
        if orden == "columnas":
            for i, l in enumerate(izq):
                c.drawString(60, y0 - i * 14, l)
            for i, l in enumerate(der):
                c.drawString(320, y0 - i * 14, l)
        else:
            for i in range(max(len(izq), len(der))):
                if i < len(izq):
                    c.drawString(60, y0 - i * 14, izq[i])
                if i < len(der):
                    c.drawString(320, y0 - i * 14, der[i])
        c.showPage()
    c.save()
    return buf.getvalue()


def pdf_una_columna(lineas, por_pagina=48, encabezado=None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    ancho, alto = A4
    paginas = [lineas[i:i + por_pagina] for i in range(0, len(lineas), por_pagina)]
    for n, pag in enumerate(paginas, start=1):
        c.setFont("Helvetica", 9)
        if encabezado:
            c.drawString(60, alto - 40, encabezado)
        c.drawString(ancho / 2 - 35, 30, f"Página {n} de {len(paginas)}")
        c.setFont("Helvetica", 11)
        for i, l in enumerate(pag):
            c.drawString(60, alto - 80 - i * 14, l)
        c.showPage()
    c.save()
    return buf.getvalue()


def decreto_lineas(ancho_pt: float) -> list[str]:
    """Un decreto de 20 artículos con TÍTULO/CAPÍTULO, envuelto al ancho de columna dado."""
    L = envolver_pt("DECRETO DIOCESANO N.º 12/2024 por el cual se regula la preparación al sacramento "
                    "del matrimonio en la diócesis de Ejemplo.", ancho_pt)
    L += ["TÍTULO I", "DISPOSICIONES GENERALES"]
    for n in range(1, 9):
        L += envolver_pt(f"Artículo {n}.- Tema {n}. El párroco acompañará a los novios en la etapa {n} del proceso. {RELLENO}", ancho_pt)
    L += ["CAPÍTULO I", "DE LOS CURSOS PREMATRIMONIALES"]
    for n in range(9, 17):
        L += envolver_pt(f"Artículo {n}.- Curso {n}. Las sesiones del módulo {n} se impartirán cada semana. {RELLENO}", ancho_pt)
    L += ["TÍTULO II", "DISPOSICIONES FINALES"]
    for n in range(17, 21):
        L += envolver_pt(f"Artículo {n}.- Final {n}. Entrará en vigor al promulgarse. {RELLENO}", ancho_pt)
    return L


def _huella(doc):
    return [(f.id_norma, f.texto, tuple(f.contexto)) for f in doc.segmentacion.fragmentos]


@unittest.skipUnless(HAY_REPORTLAB and HAY_PDFPLUMBER, "requiere reportlab, Pillow y pdfplumber")
class TestPdfDosColumnas(unittest.TestCase):
    ENC = "Diócesis de Ejemplo - Decreto 12/2024"
    P = ParametrosExpansion(palabras_min_chunk=1)

    @classmethod
    def setUpClass(cls):
        cls.referencia = _huella(procesar_pdf(pdf_una_columna(decreto_lineas(440), encabezado=cls.ENC), params=cls.P))

    def test_dos_columnas_dan_lo_mismo_que_una_columna_en_ambos_ordenes_de_dibujo(self):
        for orden in ("columnas", "filas"):
            with self.subTest(orden=orden):
                pdf = pdf_dos_columnas(decreto_lineas(235), orden=orden, encabezado=self.ENC)
                self.assertEqual(_huella(procesar_pdf(pdf, params=self.P)), self.referencia)

    def test_sin_la_deteccion_un_pdf_dibujado_por_filas_sale_mezclado(self):
        pdf = pdf_dos_columnas(decreto_lineas(235), orden="filas", encabezado=self.ENC)
        sin = _huella(procesar_pdf(pdf, params=ParametrosExpansion(palabras_min_chunk=1, detectar_columnas=False)))
        iguales = sum(1 for a, b in zip(sin, self.referencia) if a == b)
        self.assertLess(iguales, len(self.referencia) // 2)

    def test_titulo_a_todo_el_ancho_encabezado_y_pie(self):
        pdf = pdf_dos_columnas(decreto_lineas(235), titulo="PRUEBA DE TÍTULO A TODO EL ANCHO", encabezado=self.ENC)
        doc = procesar_pdf(pdf, params=self.P)
        self.assertTrue(doc.segmentacion.fragmentos[0].texto.startswith("PRUEBA DE TÍTULO A TODO EL ANCHO"))
        todo = " ".join(f.texto for f in doc.segmentacion.fragmentos)
        self.assertNotIn("Diócesis de Ejemplo - Decreto", todo)
        self.assertNotRegex(todo, r"Página \d de \d")
        self.assertEqual([f.id_norma for f in doc.segmentacion.fragmentos if f.id_norma], list(range(1, 21)))

    def test_palabra_cortada_entre_el_final_de_una_columna_y_el_inicio_de_la_siguiente(self):
        L = decreto_lineas(235)
        L[39] = "Artículo 99.- Final. Quienes deseen contraer matri-"        # última línea de la columna izquierda
        L[40] = "monio canónico acudirán a su párroco."                       # primera de la derecha
        doc = procesar_pdf(pdf_dos_columnas(L, orden="filas", encabezado=self.ENC), params=self.P)
        art = next(f for f in doc.segmentacion.fragmentos if f.id_norma == 99)
        self.assertIn("contraer matrimonio canónico acudirán", art.texto)
        self.assertNotIn("matri-", art.texto)

    def test_ultima_pagina_con_la_columna_derecha_casi_vacia(self):
        # El documento termina a mitad de la columna derecha: esa página no cumple el criterio estricto,
        # pero el documento ya se sabe de dos columnas.
        L = decreto_lineas(235)
        n = 2 * 80 + 40 + 4              # dos páginas completas (80 líneas) + una con la izquierda llena y 4 a la derecha
        L = (L * 3)[:n]
        pdf = pdf_dos_columnas(L, orden="filas", encabezado=self.ENC)
        e = extraer_paginas(pdf, self.P)
        ultima = [l for l in e.paginas[-1].texto.split("\n") if not l.startswith(("Página", "Diócesis"))]
        self.assertEqual(e.total_paginas, 3)
        self.assertEqual(ultima, L[160:])                      # orden de lectura exacto, sin mezclar filas
        self.assertEqual(e.paginas_reordenadas, e.total_paginas)

    def test_avisa_cuando_reordena(self):
        doc = procesar_pdf(pdf_dos_columnas(decreto_lineas(235), encabezado=self.ENC), params=self.P)
        self.assertTrue(any("dos columnas" in a for a in doc.advertencias))

    def test_una_columna_no_se_toca_ni_avisa(self):
        e = extraer_paginas(pdf_una_columna(decreto_lineas(440), encabezado=self.ENC), self.P)
        self.assertEqual(e.paginas_reordenadas, 0)
        self.assertFalse(any("columna" in a for a in e.advertencias))

    def test_tres_columnas_es_ambiguo_y_no_se_reordena(self):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        L = decreto_lineas(150)
        for pag in range(3):
            c.setFont("Helvetica", 9)
            for col in range(3):
                for i, l in enumerate(L[pag * 150 + col * 50: pag * 150 + (col + 1) * 50]):
                    c.drawString(40 + col * 180, A4[1] - 60 - i * 14, l)
            c.showPage()
        c.save()
        self.assertEqual(extraer_paginas(buf.getvalue(), self.P).paginas_reordenadas, 0)

    def test_documento_mixto_solo_reordena_las_paginas_de_dos_columnas(self):
        # Portada de una columna + cuerpo en dos columnas: se reordenan solo las páginas del cuerpo.
        cuerpo = pdf_dos_columnas(decreto_lineas(235), encabezado=self.ENC)
        portada = pdf_una_columna(envolver_pt(RELLENO * 6, 440), encabezado=self.ENC)
        from pypdf import PdfWriter, PdfReader
        w = PdfWriter()
        for r in (PdfReader(io.BytesIO(portada)), PdfReader(io.BytesIO(cuerpo))):
            for pagina in r.pages:
                w.add_page(pagina)
        buf = io.BytesIO()
        w.write(buf)
        e = extraer_paginas(buf.getvalue(), self.P)
        self.assertGreaterEqual(e.paginas_reordenadas, 2)
        self.assertLess(e.paginas_reordenadas, e.total_paginas)

    def test_sin_pdfplumber_se_degrada_con_aviso_y_sin_fallar(self):
        pdf = pdf_dos_columnas(decreto_lineas(235), orden="columnas", encabezado=self.ENC)
        with mock.patch.dict(sys.modules, {"pdfplumber": None}):       # simula que no está instalada
            e = extraer_paginas(pdf, self.P)
        self.assertEqual(e.paginas_reordenadas, 0)
        self.assertTrue(any("pdfplumber" in a for a in e.advertencias))

    def test_se_puede_desactivar_por_configuracion(self):
        pdf = pdf_dos_columnas(decreto_lineas(235), encabezado=self.ENC)
        e = extraer_paginas(pdf, ParametrosExpansion(detectar_columnas=False))
        self.assertEqual(e.paginas_reordenadas, 0)


RUTA_CIC = Path("src/corp_extractor/extractores/data/corpus_derecho_canonico_v1.json")


@unittest.skipUnless(HAY_REPORTLAB and HAY_PDFPLUMBER and RUTA_CIC.exists(), "requiere reportlab, pdfplumber y el corpus del proyecto")
class TestCorpusRealEnDosColumnas(unittest.TestCase):
    """Reconstruye un PDF de dos columnas con cánones REALES del corpus y comprueba que el módulo
    recupera cada canon con su texto y su contexto jerárquico idénticos."""

    def test_150_canones_en_dos_columnas(self):
        items = json.loads(RUTA_CIC.read_text(encoding="utf-8"))["datos"][:150]
        seguro = lambda s: s.encode("cp1252", "replace").decode("cp1252")        # Helvetica estándar = WinAnsi
        norm = lambda s: re.sub(r"\s+", " ", seguro(s)).strip()

        lineas, previo = [], None
        for it in items:
            ctx = it["contexto_jerarquico"]
            if ctx != previo:
                nuevos, antes = ctx.split(" > "), (previo.split(" > ") if previo else [])
                k = 0
                while k < min(len(antes), len(nuevos)) and antes[k] == nuevos[k]:
                    k += 1
                for nivel in nuevos[k:]:                    # un PDF real imprime cada nivel una vez, en su línea
                    lineas += envolver_pt(seguro(nivel), 235)
                previo = ctx
            lineas += envolver_pt(seguro(f"Can. {it['canon_id']}. {it['texto']}"), 235)

        pdf = pdf_dos_columnas(lineas, por_columna=42, orden="filas", titulo="CODIGO DE DERECHO CANONICO",
                               encabezado="Encabezado corrido")
        doc = procesar_pdf(pdf, params=ParametrosExpansion())

        grupos = []                                         # une las partes de una unidad larga
        for f in doc.segmentacion.fragmentos:
            if grupos and f.partes_total > 1 and f.parte > 1 and grupos[-1]["id"] == f.id_norma:
                grupos[-1]["t"] += " " + f.texto
            else:
                grupos.append({"id": f.id_norma, "t": f.texto, "c": " > ".join(f.contexto)})

        self.assertEqual(len(grupos), len(items))
        for g, it in zip(grupos, items):
            self.assertEqual(g["id"], it["canon_id"])
            self.assertEqual(norm(g["t"]), norm(it["texto"]))
            self.assertEqual(norm(g["c"]), norm(it["contexto_jerarquico"]))


if __name__ == "__main__":
    unittest.main()
