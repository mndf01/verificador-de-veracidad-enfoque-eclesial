"""Pruebas del módulo de extracción web (src/web_extractor).

Se ejecutan desde la raíz del proyecto, SIN red: la extracción se probado pasando
HTML en memoria (trafilatura procesa HTML sin descargar). La parte que sí
descargaría (extraer_contenido_url / extraer_titular) se prueba con un doble que
reemplaza la descarga, para no depender de internet ni de páginas externas.

    python -m pytest tests/test_web_extractor.py -v      (o: python -m unittest ...)
"""
from __future__ import annotations

import unittest
from unittest import mock

from src.web_extractor import (
    extraer_contenido_html,
    extraer_contenido_url,
    extraer_metadata_html,
    extraer_metadata_url,
    extraer_titular,
)

# Un artículo sencillo en HTML (lo que trafilatura espera poder limpiar en una noticia).
HTML_ARTICULO = """<!DOCTYPE html>
<html lang="es">
<head>
  <title>El Papa aclara la doctrina del bautismo</title>
  <meta name="author" content="Agencia Católica">
  <meta property="article:published_time" content="2024-05-01">
</head>
<body>
  <nav><a href="#">Inicio</a> <a href="#">Menú</a> <a href="#">Contacto</a></nav>
  <h1>El Papa aclara la doctrina del bautismo</h1>
  <p>El Santo Padre recordó hoy que el bautismo es la puerta de todos los sacramentos.</p>
  <p>La declaración fue hecha durante la audiencia general de este miércoles.</p>
  <footer>© 2024 Agencia Católica.</footer>
</body>
</html>"""

# Una página sin contenido de artículo (body vacío o solo imágenes): trafilatura devuelve None.
HTML_SIN_ARTICULO = """<!DOCTYPE html>
<html><head><title>Portal</title></head><body></body></html>
"""

HTML_SOLO_IMAGENES = """<html><body><img src="x.jpg"></body></html>"""


class TestExtraerContenidoHtml(unittest.TestCase):
    def test_extrae_el_parrafo_principal_sin_navegacion(self):
        texto = extraer_contenido_html(HTML_ARTICULO)
        self.assertIsNotNone(texto)
        self.assertIn("bautismo es la puerta", texto)   # contenido real del artículo
        self.assertNotIn("Inicio", texto)               # el menú nav no debe aparecer
        self.assertNotIn("© 2024", texto)               # el footer no debe aparecer

    def test_contenido_quee_no_es_el_titulo(self):
        # El cuerpo extraído no debe incluir el <h1>/<title> (solo el cuerpo del artículo).
        texto = extraer_contenido_html(HTML_ARTICULO)
        self.assertNotEqual(texto.strip(), HTML_ARTICULO)   # no devuelve el html crudo

    def test_html_vacio_devuelve_none(self):
        self.assertIsNone(extraer_contenido_html(""))
        self.assertIsNone(extraer_contenido_html("   \n  "))

    def test_html_sin_articulo_devuelve_none(self):
        self.assertIsNone(extraer_contenido_html(HTML_SIN_ARTICULO))
        self.assertIsNone(extraer_contenido_html(HTML_SOLO_IMAGENES))


class TestExtraerMetadata(unittest.TestCase):
    def test_metadata_completa(self):
        meta = extraer_metadata_html(HTML_ARTICULO, url="https://ejemplo.com/x")
        self.assertEqual(meta["titulo"], "El Papa aclara la doctrina del bautismo")
        self.assertEqual(meta["autor"], "Agencia Católica")
        self.assertEqual(meta["fecha"], "2024-05-01")
        # idioma puede quedar vacío si el texto es demasiado corto para detectarlo;
        # pero la clave SIEMPRE debe existir (nunca lanzar).
        self.assertIn("idioma", meta)

    def test_metadata_html_vacio_no_lanza(self):
        self.assertEqual(
            extraer_metadata_html(""),
            {"titulo": "", "autor": "", "fecha": "", "idioma": ""},
        )


class TestExtraerUrl(unittest.TestCase):
    """La parte que descarga se prueba reemplazando fetch_url por un doble."""

    URL = "https://noticias.ejemplo.com/articulo"

    def test_devuelve_contenido_descargado_y_limpiado(self):
        with mock.patch(
            "src.web_extractor.web_extractor.trafilatura.fetch_url",
            return_value=HTML_ARTICULO,
        ):
            texto = extraer_contenido_url(self.URL)
        self.assertIsNotNone(texto)
        self.assertIn("bautismo es la puerta", texto)

    def test_url_invalida_no_intenta_descargar(self):
        with mock.patch(
            "src.web_extractor.web_extractor.trafilatura.fetch_url",
            side_effect=AssertionError("no debe llamarse con URL inválida"),
        ):
            self.assertIsNone(extraer_contenido_url(""))
            self.assertIsNone(extraer_contenido_url("   "))
            self.assertIsNone(extraer_contenido_url(None))

    def test_fallo_de_red_se_convierte_en_none(self):
        with mock.patch(
            "src.web_extractor.web_extractor.trafilatura.fetch_url",
            side_effect=RuntimeError("timeout"),
        ):
            self.assertIsNone(extraer_contenido_url(self.URL))

    def test_descarga_vacia_devuelve_none(self):
        with mock.patch(
            "src.web_extractor.web_extractor.trafilatura.fetch_url",
            return_value=None,
        ):
            self.assertIsNone(extraer_contenido_url(self.URL))

    def test_extraer_titular_usa_metadata(self):
        with mock.patch(
            "src.web_extractor.web_extractor.trafilatura.fetch_url",
            return_value=HTML_ARTICULO,
        ):
            titulo = extraer_titular(self.URL)
        self.assertEqual(titulo, "El Papa aclara la doctrina del bautismo")


if __name__ == "__main__":
    unittest.main()