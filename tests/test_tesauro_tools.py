"""Pruebas de las herramientas del tesauro (src/procesamiento_nlp/tesauro_tools.py).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.procesamiento_nlp.tesauro_tools import (
    TesauroInvalido,
    buscar_candidatos,
    cargar_corpus,
    cargar_tesauro,
    categorias_del_clasificador,
    main,
    normalizar,
    sugerir_formas,
    validar,
)


def concepto(canon="Romano Pontífice", variantes=("el Papa",), categoria="Jerarquía y Gobierno Eclesiástico", **extra):
    return {"termino_canonico": canon, "variaciones_lexicas": list(variantes), "categoria": categoria, **extra}


def tesauro(**conceptos):
    return {"conceptos": conceptos}


def mensajes(hallazgos, nivel):
    return [h.mensaje for h in hallazgos if h.nivel == nivel]


class TestNormalizar(unittest.TestCase):
    def test_sin_tildes_ni_mayusculas(self):
        self.assertEqual(normalizar("  Dimisión  del ESTADO clerical "), "dimision del estado clerical")


class TestValidarFormato(unittest.TestCase):
    def test_tesauro_correcto_no_da_errores_ni_avisos(self):
        h = validar(tesauro(ID_PAPA=concepto()), [], None)
        self.assertEqual([x for x in h if x.nivel in ("ERROR", "AVISO")], [])

    def test_sin_clave_conceptos(self):
        self.assertEqual(validar({}, [], None)[0].nivel, "ERROR")
        self.assertEqual(validar({"conceptos": {}}, [], None)[0].nivel, "ERROR")

    def test_faltan_campos_obligatorios_y_id_mal_formado(self):
        h = validar(tesauro(papa={"categoria": ""}), [], None)
        errores = " | ".join(mensajes(h, "ERROR"))
        for esperado in ("El id debe empezar con ID_", "termino_canonico", "variaciones_lexicas", "categoria"):
            self.assertIn(esperado, errores)

    def test_variante_vacia_es_error(self):
        h = validar(tesauro(ID_X=concepto(variantes=["Papa", "  "])), [], None)
        self.assertTrue(any("vacío" in m for m in mensajes(h, "ERROR")))

    def test_termino_en_dos_conceptos_es_error_aunque_cambien_tildes_y_mayusculas(self):
        h = validar(tesauro(ID_A=concepto(variantes=["Pontífice"]), ID_B=concepto("Otro", ["pontifice"])), [], None)
        self.assertTrue(any("pontifice" in m and "ID_A" in m and "ID_B" in m for m in mensajes(h, "ERROR")))

    def test_termino_repetido_dentro_del_mismo_concepto_es_aviso(self):
        h = validar(tesauro(ID_A=concepto(variantes=["Papa", "papa"])), [], None)
        self.assertTrue(any("repetido dentro del mismo concepto" in m for m in mensajes(h, "AVISO")))

    def test_campos_desconocidos_son_aviso(self):
        h = validar(tesauro(ID_A=concepto(color="rojo")), [], None)
        self.assertTrue(any("color" in m for m in mensajes(h, "AVISO")))


class TestNivelAutoridadOpcional(unittest.TestCase):
    def test_no_es_obligatorio(self):
        h = validar(tesauro(ID_A=concepto()), [], None)
        self.assertFalse(any("nivel_autoridad" in x.mensaje for x in h))

    def test_si_existe_y_es_valido_solo_informa_que_no_se_usa(self):
        h = validar(tesauro(ID_A=concepto(nivel_autoridad=0.8)), [], None)
        self.assertEqual(mensajes(h, "ERROR"), [])
        self.assertTrue(any("no se usa" in m for m in mensajes(h, "INFO")))

    def test_valores_invalidos_son_error(self):
        for malo in (1.5, 0, -1, "alto", True, None):
            with self.subTest(valor=malo):
                h = validar(tesauro(ID_A=concepto(nivel_autoridad=malo)), [], None)
                self.assertTrue(any("nivel_autoridad" in m for m in mensajes(h, "ERROR")))


class TestCategorias(unittest.TestCase):
    CATS = {"Doctrina Social y Política", "Derechos y Obligaciones de los Fieles (Laicado)", "Sacramentos y Liturgia"}

    def test_categoria_desconocida_avisa_y_sugiere_la_parecida(self):
        h = validar(tesauro(ID_A=concepto(categoria="Derechos y Obligaciones de los Fieles")), [], self.CATS)
        aviso = mensajes(h, "AVISO")[0]
        self.assertIn("no existe en el clasificador", aviso)
        self.assertIn("(Laicado)", aviso)

    def test_categoria_valida_no_avisa(self):
        h = validar(tesauro(ID_A=concepto(categoria="Sacramentos y Liturgia")), [], self.CATS)
        self.assertEqual(mensajes(h, "AVISO"), [])

    def test_sin_lista_de_categorias_se_omite_con_info(self):
        h = validar(tesauro(ID_A=concepto()), [], None)
        self.assertTrue(any("categorías" in m for m in mensajes(h, "INFO")))

    def test_lee_las_categorias_de_las_reglas_incluido_mapeo_interno(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "clasif.py"
            f.write_text(
                "REGLAS_ETL = {'A': {'categoria_default': 'Cat A'},"
                " 'B': {'mapeo_interno': {'LIBRO I': 'Cat B', 'LIBRO VI': 'Cat Penal'}},"
                " 'C': {'mapeo_documentos': {'x': {'categoria': 'Cat C', 'autoridad': 1.0}}}}\n", encoding="utf-8")
            self.assertEqual(categorias_del_clasificador(f), {"Cat A", "Cat B", "Cat Penal", "Cat C"})
            roto = Path(d) / "roto.py"
            roto.write_text("esto no es python (", encoding="utf-8")
            self.assertIsNone(categorias_del_clasificador(roto))
            self.assertIsNone(categorias_del_clasificador(Path(d) / "no_existe.py"))

    def test_clasificador_real_del_proyecto_incluye_derecho_penal(self):
        real = Path("src/corp_extractor/clasificador_corpus/clasificador_metadatos_v1.py")
        if not real.exists():
            self.skipTest("no está el clasificador del proyecto")
        self.assertIn("Derecho Penal Eclesiástico", categorias_del_clasificador(real))


class TestContraElCorpus(unittest.TestCase):
    CORPUS = (["El obispo diocesano gobierna. Los fieles tienen derecho a la libertad."] * 20
              + ["Quien ha sido sancionado con la expulsión del estado clerical pierde los derechos."] * 5
              + ["La huelga es un recurso extremo; el derecho de los trabajadores es limitado."] * 2)

    def test_termino_canonico_ausente_avisa_y_sugiere_lo_que_dice_el_corpus(self):
        h = validar(tesauro(ID_E=concepto("Dimisión del estado clerical", ["lo echaron de cura"])), self.CORPUS, None)
        aviso = [m for m in mensajes(h, "AVISO") if "no aparece" in m][0]
        self.assertIn("estado clerical", aviso)

    def test_termino_canonico_presente_no_avisa(self):
        h = validar(tesauro(ID_E=concepto("Estado clerical", ["cura"])), self.CORPUS, None)
        self.assertFalse(any("no aparece" in m for m in mensajes(h, "AVISO")))

    def test_la_sugerencia_prefiere_palabras_raras_sobre_las_comunes(self):
        # "derecho" está en muchos fragmentos; "huelga" en solo 2: lo informativo es "huelga".
        formas = sugerir_formas("Derecho de huelga", self.CORPUS)
        self.assertEqual(formas[0][0], "huelga")

    def test_sugerir_conserva_las_tildes_del_corpus(self):
        formas = [f for f, _ in sugerir_formas("Dimisión del estado clerical", self.CORPUS, k=5)]
        self.assertIn("expulsión del estado clerical", formas)

    def test_variante_de_una_palabra_muy_comun_avisa(self):
        corpus = ["La autoridad civil y el derecho civil regulan el estado civil."] * 20
        h = validar(tesauro(ID_L=concepto("Laico", ["Civil"])), corpus, None)
        self.assertTrue(any("Civil" in m and "ambigüedad" in m for m in mensajes(h, "AVISO")))

    def test_variante_de_su_propio_concepto_no_avisa(self):
        corpus = ["El matrimonio es un sacramento; el vínculo matrimonial es indisoluble."] * 20
        h = validar(tesauro(ID_M=concepto("Vínculo Matrimonial", ["Matrimonio"])), corpus, None)
        self.assertFalse(any("ambigüedad" in m for m in mensajes(h, "AVISO")))

    def test_sin_corpus_se_omiten_las_comprobaciones_con_info(self):
        h = validar(tesauro(ID_A=concepto("Algo inexistente")), [], None)
        self.assertTrue(any("corpus" in m for m in mensajes(h, "INFO")))
        self.assertEqual(mensajes(h, "AVISO"), [])


class TestCandidatos(unittest.TestCase):
    CORPUS = (["La sede apostólica y el obispo diocesano actúan, teniendo en cuenta el bien común."] * 15
              + ["El obispo diocesano preside la iglesia particular."] * 5)

    def test_encuentra_frases_frecuentes_con_tildes_y_ordenadas(self):
        frases = [f["frase"] for f in buscar_candidatos(self.CORPUS, set(), minimo=10)]
        self.assertEqual(frases[0], "obispo diocesano")
        self.assertIn("sede apostólica", frases)
        self.assertIn("bien común", frases)

    def test_respeta_el_minimo_y_el_top(self):
        self.assertNotIn("iglesia particular", [f["frase"] for f in buscar_candidatos(self.CORPUS, set(), minimo=10)])
        self.assertEqual(len(buscar_candidatos(self.CORPUS, set(), minimo=10, top=1)), 1)

    def test_excluye_lo_que_ya_esta_en_el_tesauro_aunque_cambien_las_tildes(self):
        frases = [f["frase"] for f in buscar_candidatos(self.CORPUS, {"Sede Apostolica"}, minimo=10)]
        self.assertNotIn("sede apostólica", frases)
        self.assertIn("obispo diocesano", frases)

    def test_descarta_frases_de_relleno(self):
        frases = [f["frase"] for f in buscar_candidatos(self.CORPUS, set(), minimo=10)]
        self.assertFalse(any("teniendo" in f or "cuenta" in f for f in frases))


class TestArchivosYLineaDeComandos(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.base = Path(self.dir.name)
        (self.base / "corpus").mkdir()
        datos = [{"texto": "El obispo diocesano gobierna la diócesis y preside la sede apostólica."}] * 14
        (self.base / "corpus" / "corpus_x_v1.json").write_text(json.dumps({"datos": datos}), encoding="utf-8")
        (self.base / "corpus" / "otro.json").write_text("{}", encoding="utf-8")        # se ignora: no es corpus_*_v1
        self.tes = self.base / "tesauro.json"

    def tearDown(self):
        self.dir.cleanup()

    def correr(self, *args):
        salida, error = io.StringIO(), io.StringIO()
        with redirect_stdout(salida), redirect_stderr(error):
            codigo = main(["--tesauro", str(self.tes), "--corpus", str(self.base / "corpus"), *args])
        return codigo, salida.getvalue(), error.getvalue()

    def test_cargar_corpus_lee_solo_los_corpus_y_tolera_carpeta_vacia(self):
        self.assertEqual(len(cargar_corpus(self.base / "corpus")), 14)
        self.assertEqual(cargar_corpus(self.base / "no_existe"), [])

    def test_json_mal_formado_dice_en_que_linea(self):
        self.tes.write_text('{\n  "conceptos": {\n    "ID_A": {"x": 1,}\n  }\n}', encoding="utf-8")
        with self.assertRaises(TesauroInvalido) as ctx:
            cargar_tesauro(self.tes)
        self.assertIn("línea 3", str(ctx.exception))
        codigo, _, error = self.correr("validar")
        self.assertEqual(codigo, 1)
        self.assertIn("línea 3", error)

    def test_archivo_inexistente(self):
        codigo, _, error = self.correr("validar")
        self.assertEqual(codigo, 1)
        self.assertIn("No se encontró", error)

    def test_validar_devuelve_0_si_no_hay_errores_y_1_si_los_hay(self):
        self.tes.write_text(json.dumps(tesauro(ID_O=concepto("Obispo diocesano", ["el obispo de la zona"]))), encoding="utf-8")
        codigo, salida, _ = self.correr("validar")
        self.assertEqual(codigo, 0)
        self.assertIn("1 conceptos revisados", salida)
        self.tes.write_text(json.dumps(tesauro(ID_O=concepto(variantes=["x"]), ID_P=concepto("Otro", ["X"]))), encoding="utf-8")
        self.assertEqual(self.correr("validar")[0], 1)

    def test_candidatos_imprime_y_guarda_un_csv_que_excel_abre_con_tildes(self):
        self.tes.write_text(json.dumps(tesauro(ID_O=concepto("Obispo", ["el obispo"]))), encoding="utf-8")
        salida_csv = self.base / "cand.csv"
        codigo, salida, _ = self.correr("candidatos", "--minimo", "10", "--salida", str(salida_csv))
        self.assertEqual(codigo, 0)
        self.assertIn("sede apostólica", salida)
        crudo = salida_csv.read_bytes()
        self.assertEqual(crudo[:3], b"\xef\xbb\xbf")                       # BOM para Excel
        self.assertIn("sede apostólica;14;".encode("utf-8"), crudo)

    def test_candidatos_sin_corpus_da_un_error_claro(self):
        self.tes.write_text(json.dumps(tesauro(ID_O=concepto())), encoding="utf-8")
        (self.base / "corpus" / "corpus_x_v1.json").unlink()
        codigo, _, error = self.correr("candidatos")
        self.assertEqual(codigo, 1)
        self.assertIn("corpus", error)


if __name__ == "__main__":
    unittest.main()
