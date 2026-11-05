"""Tests de la interfaz de terminal (Rich) y del chat.

Se captura la salida de Rich con `Console(file=StringIO())`, así que no hace falta
una terminal real ni color. El chat se prueba con dobles del orquestador para no
descargar noticias ni cargar el modelo.
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path as _P

_RAIZ = _P(__file__).resolve().parents[1]
if _RAIZ not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.cli import ui  # noqa: E402
from src.orquestador import ResultadoEvaluacion  # noqa: E402


def _consola():
    """Consola que escribe a un buffer en memoria (sin terminal ni color)."""
    buffer = io.StringIO()
    return ui.crear_consola(sin_color=True, archivo=buffer), buffer


def _resultado(**kw) -> ResultadoEvaluacion:
    base = dict(
        premisa="El bautismo es necesario para la salvacion",
        url="https://ejemplo.com/n",
        ok=True,
        veredicto="proximo",
        probabilidad=0.87,
        informe_ivr={
            "ok": True,
            "n_evidencias": 2,
            "ivr_max": 0.854,
            "ivr_medio": 0.727,
            "ivr_min": 0.655,
            "conceptos_premisa": {"ID_BAUTISMO": "Sacramentos y Liturgia"},
            "evidencias": [
                {"ivr": 0.854, "cita": "CIC, canon 849", "nivel_autoridad": 1.0,
                 "origen": "corpus_base", "senales": {}},
                {"ivr": 0.655, "cita": "Compendio, numeral 252", "nivel_autoridad": 0.55,
                 "origen": "corpus_base", "senales": {}},
            ],
        },
    )
    base.update(kw)
    return ResultadoEvaluacion(**base)


class TestRenderizado(unittest.TestCase):
    """`ui.py` dibuja sin romperse y muestra la información esperada."""

    def test_bienvenida_menciona_similitud_no_verdad(self):
        consola, buf = _consola()
        ui.mostrar_bienvenida(consola, {"clases": ["proximo", "distante", "no_definido"],
                                        "n_features": 16})
        salida = buf.getvalue()
        self.assertIn("Verificador de Veracidad", salida)
        # Mensaje clave de honestidad metodológica: mide similitud, no verdad.
        self.assertIn("SIMILITUD", salida)

    def test_resultado_proximo(self):
        consola, buf = _consola()
        ui.mostrar_resultado(consola, _resultado())
        salida = buf.getvalue()
        self.assertIn("PRÓXIMO AL CORPUS", salida)
        self.assertIn("87%", salida)
        self.assertIn("CIC, canon 849", salida)
        self.assertIn("0.854", salida)

    def test_resultado_distante(self):
        consola, buf = _consola()
        ui.mostrar_resultado(consola, _resultado(veredicto="distante", probabilidad=0.65))
        self.assertIn("DISTANTE DEL CORPUS", buf.getvalue())

    def test_resultado_no_definido(self):
        consola, buf = _consola()
        ui.mostrar_resultado(consola, _resultado(veredicto="no_definido", probabilidad=0.71))
        salida = buf.getvalue()
        self.assertIn("NO DEFINIDO", salida)
        self.assertIn("no hay base", salida.lower())

    def test_resultado_con_error_no_lanza(self):
        consola, buf = _consola()
        res = ResultadoEvaluacion(premisa="x", url="y", ok=False,
                                  errores=["No se pudo extraer contenido de la URL."])
        ui.mostrar_resultado(consola, res)
        salida = buf.getvalue()
        self.assertIn("Sin resultado", salida)
        self.assertIn("No se pudo extraer contenido", salida)

    def test_resultado_sin_evidencias_no_lanza(self):
        consola, buf = _consola()
        res = _resultado(informe_ivr={"ok": False, "evidencias": []})
        ui.mostrar_resultado(consola, res)
        self.assertIn("PRÓXIMO AL CORPUS", buf.getvalue())

    def test_muestra_pdf_en_sesion(self):
        consola, buf = _consola()

        class _Resumen:
            nombre = "decreto_ejemplo"
            fragmentos = 9

        res = _resultado()
        res.informe_ivr["expansion_dinamica"] = {
            "ruta_pdf": "x.pdf", "resumen": _Resumen()}
        ui.mostrar_resultado(consola, res)
        salida = buf.getvalue()
        self.assertIn("decreto_ejemplo", salida)
        self.assertIn("9 fragmentos", salida)

    def test_evidencia_de_pdf_se_marca(self):
        consola, buf = _consola()
        res = _resultado()
        res.informe_ivr["evidencias"][0]["origen"] = "expansion_dinamica"
        ui.mostrar_resultado(consola, res)
        self.assertIn("PDF", buf.getvalue())

    def test_historial_vacio(self):
        consola, buf = _consola()
        ui.mostrar_historial(consola, [])
        self.assertIn("Aún no hay evaluaciones", buf.getvalue())

    def test_historial_con_registros(self):
        consola, buf = _consola()
        ui.mostrar_historial(consola, [
            {"consulta": "https://a.com", "veredicto": "proximo", "probabilidad": 0.9},
            {"consulta": "texto libre", "veredicto": "distante", "probabilidad": 0.6},
        ])
        salida = buf.getvalue()
        self.assertIn("https://a.com", salida)
        self.assertIn("proximo", salida)
        self.assertIn("90%", salida)

    def test_barra_acota_fuera_de_rango(self):
        """Valores >1 o <0 no deben romper la barra."""
        self.assertIn("1.000", ui._barra(1.5).plain)
        self.assertIn("0.000", ui._barra(-0.3).plain)

    def test_ayuda_lista_comandos(self):
        consola, buf = _consola()
        ui.mostrar_ayuda(consola)
        salida = buf.getvalue()
        for cmd in ("/pdf", "/historial", "/modelo", "/salir"):
            self.assertIn(cmd, salida)


class TestSesionChat(unittest.TestCase):
    """Estado y comandos de la sesión, sin tocar red ni Chroma."""

    def _sesion(self):
        from src.cli.chat import SesionChat
        return SesionChat()

    def test_registra_historial(self):
        s = self._sesion()
        s.registrar("https://a.com", _resultado())
        self.assertEqual(len(s.historial), 1)
        self.assertEqual(s.historial[0]["veredicto"], "proximo")

    def test_comando_salir_termina(self):
        from src.cli.chat import _procesar_comando
        consola, _ = _consola()
        self.assertFalse(_procesar_comando(consola, self._sesion(), "/salir"))

    def test_comando_ayuda_continua(self):
        from src.cli.chat import _procesar_comando
        consola, buf = _consola()
        self.assertTrue(_procesar_comando(consola, self._sesion(), "/ayuda"))
        self.assertIn("/pdf", buf.getvalue())

    def test_comando_historial(self):
        from src.cli.chat import _procesar_comando
        consola, buf = _consola()
        s = self._sesion()
        s.registrar("https://a.com", _resultado())
        self.assertTrue(_procesar_comando(consola, s, "/historial"))
        self.assertIn("https://a.com", buf.getvalue())

    def test_comando_desconocido_avisa(self):
        from src.cli.chat import _procesar_comando
        consola, buf = _consola()
        self.assertTrue(_procesar_comando(consola, self._sesion(), "/inventado"))
        self.assertIn("Comando desconocido", buf.getvalue())

    def test_pdf_sin_argumento_informa(self):
        from src.cli.chat import _procesar_comando
        consola, buf = _consola()
        self.assertTrue(_procesar_comando(consola, self._sesion(), "/pdf"))
        self.assertIn("No hay PDF cargado", buf.getvalue())

    def test_pdf_off_sin_pdf_no_lanza(self):
        from src.cli.chat import _procesar_comando
        consola, _ = _consola()
        self.assertTrue(_procesar_comando(consola, self._sesion(), "/pdf off"))


class TestEvaluacionDesdeChat(unittest.TestCase):
    """El chat enruta URLs vs texto libre al orquestador correctamente."""

    def test_texto_libre_no_descarga(self):
        from src.cli.chat import SesionChat
        capturado = {}

        def fake_evaluar(url, premisa=None, extraer_contenido=None, **kw):
            capturado.update(url=url, premisa=premisa)
            return _resultado()

        import src.orquestador as O
        original = O.evaluar_noticia
        O.evaluar_noticia = fake_evaluar
        try:
            SesionChat().evaluar("El bautismo es necesario")
        finally:
            O.evaluar_noticia = original

        self.assertEqual(capturado["premisa"], "El bautismo es necesario")
        self.assertEqual(capturado["url"], "")       # no es URL → no descarga

    def test_url_se_pasa_como_url(self):
        from src.cli.chat import SesionChat
        capturado = {}

        def fake_evaluar(url, premisa=None, **kw):
            capturado.update(url=url, premisa=premisa)
            return _resultado()

        import src.orquestador as O
        original = O.evaluar_noticia
        O.evaluar_noticia = fake_evaluar
        try:
            SesionChat().evaluar("https://ejemplo.com/noticia")
        finally:
            O.evaluar_noticia = original

        self.assertEqual(capturado["url"], "https://ejemplo.com/noticia")
        self.assertIsNone(capturado["premisa"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
