"""Pruebas del módulo IVR (src/vectorizacion/ivr.py).

Se ejecutan desde la raíz del proyecto, sin red, sin Chroma real y sin cargar SBERT
(la recuperación se sustituye por un doble de prueba):

    python -m unittest discover -s tests -v        (o:  pytest tests -v)

Los ids de concepto se leen del propio tesauro, así que si el equipo agrega o renombra
términos estas pruebas siguen siendo válidas.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from src.procesamiento_nlp.tesauro_tools import RUTA_TESAURO
from src.vectorizacion import ivr


def conceptos_del_tesauro() -> dict:
    return json.loads(RUTA_TESAURO.read_text(encoding="utf-8-sig"))["conceptos"]


def id_de_forma(forma: str) -> str:
    """Id del concepto que contiene esa forma (término canónico o variación léxica)."""
    for id_concepto, concepto in conceptos_del_tesauro().items():
        formas = [concepto.get("termino_canonico", "")] + list(concepto.get("variaciones_lexicas", []))
        if forma in formas:
            return id_concepto
    raise AssertionError(f"La forma {forma!r} no está en el tesauro")


def evidencia_falsa(texto: str, similitud: float, categoria: str, nivel_autoridad: float) -> dict:
    """Contrato mínimo que devuelve chromaDB.consultar()."""
    return {
        "id_chroma": "corpus_canones:0001",
        "texto": texto,
        "similitud": similitud,
        "contexto": "LIBRO I",
        "fuente": "Código de Derecho Canónico",
        "id_norma": 1,
        "tipo_corpus": "corpus_canones",
        "categoria": categoria,
        "nivel_autoridad": nivel_autoridad,
        "url_origen": "Sin url registrada",
        "cita": "CIC, canon 1",
    }


PESOS_UNO = {"sb": 0.25, "literal": 0.25, "ontologico": 0.25, "autoridad": 0.25}


class TestDeteccionDeConceptos(unittest.TestCase):
    def test_detecta_variacion_lexica_sin_acentos_ni_mayusculas(self):
        detectados = ivr.detectar_conceptos("el PAPÁ viajó a Roma")
        self.assertIn(id_de_forma("Papa"), detectados)

    def test_respeta_limites_de_palabra(self):
        # "papado" contiene "papa" pero NO es el término Papa.
        self.assertNotIn(id_de_forma("Papa"), ivr.detectar_conceptos("El papado medieval"))

    def test_texto_vacio_o_sin_conceptos(self):
        self.assertEqual(ivr.detectar_conceptos(""), {})
        self.assertEqual(ivr.detectar_conceptos("llueve mucho hoy en la ciudad"), {})

    def test_toda_categoria_devuelta_pertenece_al_tesauro(self):
        categorias_esperadas = {c.get("categoria") for c in conceptos_del_tesauro().values()}
        detectados = ivr.detectar_conceptos("El Papa y el bautismo de los fieles")
        self.assertTrue(detectados)
        self.assertTrue(set(detectados.values()) <= categorias_esperadas)


class TestCoincidenciaOntologica(unittest.TestCase):
    def test_sin_conceptos_la_senal_no_aplica(self):
        self.assertIsNone(ivr.coincidencia_ontologica({}, "Sacramentos y Liturgia"))

    def test_todos_los_conceptos_en_la_misma_categoria(self):
        conceptos = {"A": "Sacramentos y Liturgia", "B": "Sacramentos y Liturgia"}
        self.assertEqual(ivr.coincidencia_ontologica(conceptos, "Sacramentos y Liturgia"), 1.0)
        self.assertEqual(ivr.coincidencia_ontologica(conceptos, "Derecho Penal Eclesiástico"), 0.0)

    def test_fraccion_de_conceptos_coincidentes(self):
        conceptos = {"A": "Sacramentos y Liturgia", "B": "Derecho Penal Eclesiástico"}
        self.assertEqual(ivr.coincidencia_ontologica(conceptos, "Sacramentos y Liturgia"), 0.5)


class TestSimilitudLiteral(unittest.TestCase):
    NORMA = ("El parroco debe ser nombrado por el obispo diocesano, oido el consejo presbiteral, "
             "y ha de poseer las cualidades que el derecho requiere para el cargo.")

    def test_textos_identicos(self):
        self.assertEqual(ivr.similitud_literal("bautismo de los ninos", "bautismo de los ninos"), 1.0)

    def test_textos_sin_contenido_compartido(self):
        self.assertEqual(ivr.similitud_literal("el papa nombra obispos", "ayuno y abstinencia"), 0.0)

    def test_solo_palabras_vacias(self):
        self.assertEqual(ivr.similitud_literal("de la que el", self.NORMA), 0.0)
        self.assertEqual(ivr.similitud_literal("el papa nombra obispos", "de la que el"), 0.0)

    def test_contencion_detecta_frase_copiada(self):
        valor = ivr.similitud_literal("nombrado por el obispo diocesano", self.NORMA)
        self.assertGreater(valor, 0.0)

    def test_dice_penaliza_normas_largas(self):
        premisa = "nombrado por el obispo diocesano"
        contencion = ivr.similitud_literal(premisa, self.NORMA, metrica="contencion")
        dice = ivr.similitud_literal(premisa, self.NORMA, metrica="dice")
        self.assertLess(dice, contencion)

    def test_premisa_corta_cae_a_palabras_sueltas(self):
        self.assertEqual(ivr.similitud_literal("divorcio", "el divorcio no es posible"), 1.0)

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            ivr.similitud_literal("a b", "a b", metrica="inventada")
        with self.assertRaises(ValueError):
            ivr.similitud_literal("a b", "a b", n=0)


class TestFormulaIvr(unittest.TestCase):
    def test_promedio_ponderado_conocido(self):
        senales = {"sb": 1.0, "literal": 0.5, "ontologico": 0.0, "autoridad": 1.0}
        esperado = 0.4 * 1.0 + 0.25 * 0.5 + 0.20 * 0.0 + 0.15 * 1.0
        self.assertAlmostEqual(ivr.calcular_ivr_evidencia(senales, ivr.pesos_ivr()), esperado)

    def test_sin_ontologia_renormaliza_los_pesos(self):
        # SB=1, L=0, J=1 y SO no disponible: el divisor pierde el peso de SO.
        senales = {"sb": 1.0, "literal": 0.0, "ontologico": None, "autoridad": 1.0}
        pesos = ivr.pesos_ivr()
        esperado = (pesos["sb"] * 1.0 + pesos["literal"] * 0.0 + pesos["autoridad"] * 1.0) / (
            pesos["sb"] + pesos["literal"] + pesos["autoridad"]
        )
        self.assertAlmostEqual(ivr.calcular_ivr_evidencia(senales, pesos), esperado)
        # Si SO se contara como 0 (en vez de no disponible) el IVR sería menor.
        senales_con_so_cero = {**senales, "ontologico": 0.0}
        self.assertLess(ivr.calcular_ivr_evidencia(senales_con_so_cero, pesos), esperado)

    def test_todos_los_pesos_en_cero(self):
        pesos = {"sb": 0.0, "literal": 0.0, "ontologico": 0.0, "autoridad": 0.0}
        self.assertEqual(ivr.calcular_ivr_evidencia({"sb": 1.0}, pesos), 0.0)

    def test_las_senales_se_recortan_a_uno(self):
        senales = {"sb": 1.5, "literal": 1.5, "ontologico": 1.5, "autoridad": 1.5}
        self.assertEqual(ivr.calcular_ivr_evidencia(senales, PESOS_UNO), 1.0)

    def test_senal_ausente_no_cuenta(self):
        self.assertEqual(ivr.calcular_ivr_evidencia({}, PESOS_UNO), 0.0)

    def test_los_pesos_solo_importan_en_proporcion(self):
        senales = {"sb": 0.8, "literal": 0.4, "ontologico": None, "autoridad": 0.6}
        escala = {nombre: peso * 10 for nombre, peso in PESOS_UNO.items()}
        self.assertAlmostEqual(
            ivr.calcular_ivr_evidencia(senales, PESOS_UNO),
            ivr.calcular_ivr_evidencia(senales, escala),
        )


class TestInformeIvr(unittest.TestCase):
    """Prueba la agregación con un doble de consultar(): no toca Chroma ni SBERT."""

    CATEGORIA = "Jerarquía y Gobierno Eclesiástico"
    PREMISA = "El Papa nombró a un nuevo obispo diocesano"

    def _parchear(self, evidencias):
        return mock.patch.object(ivr, "consultar", return_value=evidencias)

    def test_sin_evidencias_devuelve_ok_falso(self):
        with self._parchear([]):
            informe = ivr.calcular_ivr(self.PREMISA)
        self.assertFalse(informe["ok"])
        self.assertEqual(informe["evidencias"], [])
        self.assertEqual(ivr.vector_caracteristicas(informe), {})

    def test_reordena_por_ivr_y_guarda_la_posicion_del_rag(self):
        evidencias = [
            evidencia_falsa("texto irrelevante sobre el ayuno", 0.90, "Sacramentos y Liturgia", 0.66),
            evidencia_falsa("el obispo diocesano es nombrado y el Papa lo confirma", 0.70, self.CATEGORIA, 1.0),
        ]
        with self._parchear(evidencias):
            informe = ivr.calcular_ivr(self.PREMISA)

        self.assertTrue(informe["ok"])
        self.assertEqual(informe["n_evidencias"], 2)
        # La segunda venía peor rankeada por coseno pero gana con el IVR.
        self.assertEqual(informe["mejor"]["posicion_rag"], 2)
        self.assertEqual(informe["mejor"]["cita"], "CIC, canon 1")
        self.assertEqual(informe["evidencias"][0]["ivr"], informe["ivr_max"])
        self.assertGreaterEqual(informe["ivr_max"], informe["ivr_medio"])
        self.assertGreaterEqual(informe["ivr_medio"], informe["ivr_min"])

    def test_sin_conceptos_en_la_premisa_so_es_no_disponible(self):
        evidencias = [evidencia_falsa("texto cualquiera", 0.8, self.CATEGORIA, 1.0)]
        with self._parchear(evidencias):
            informe = ivr.calcular_ivr("llueve mucho hoy en la ciudad")

        self.assertEqual(informe["conceptos_premisa"], {})
        self.assertIsNone(informe["senales_max"]["ontologico"])
        self.assertIsNone(informe["evidencias"][0]["senales"]["ontologico"])
        caracteristicas = ivr.vector_caracteristicas(informe)
        self.assertEqual(caracteristicas["so_disponible"], 0.0)
        self.assertEqual(caracteristicas["ontologico_max"], 0.0)

    def test_vector_de_caracteristicas_plano(self):
        evidencias = [
            evidencia_falsa("el obispo diocesano y el Papa", 0.75, self.CATEGORIA, 1.0),
            evidencia_falsa("otro texto", 0.60, "Sacramentos y Liturgia", 0.66),
        ]
        with self._parchear(evidencias):
            informe = ivr.calcular_ivr(self.PREMISA)

        caracteristicas = ivr.vector_caracteristicas(informe)
        esperadas = {
            "n_evidencias", "n_conceptos_premisa", "ivr_max", "ivr_medio", "ivr_min", "ivr_brecha",
            "sb_max", "sb_medio", "sb_min", "sb_brecha", "literal_max", "literal_medio",
            "ontologico_max", "so_disponible", "terminos_compartidos_max", "autoridad_max",
        }
        self.assertEqual(set(caracteristicas), esperadas)
        self.assertTrue(all(isinstance(v, float) for v in caracteristicas.values()))
        self.assertEqual(caracteristicas["so_disponible"], 1.0)
        self.assertAlmostEqual(caracteristicas["sb_max"], 0.75)

    def test_usa_los_pesos_pasados_por_parametro(self):
        evidencias = [evidencia_falsa("el obispo y el Papa", 0.5, self.CATEGORIA, 1.0)]
        with self._parchear(evidencias):
            informe = ivr.calcular_ivr(self.PREMISA, pesos={"sb": 1.0, "literal": 0.0,
                                                          "ontologico": 0.0, "autoridad": 0.0})
        self.assertAlmostEqual(informe["ivr_max"], 0.5)


if __name__ == "__main__":
    unittest.main()
