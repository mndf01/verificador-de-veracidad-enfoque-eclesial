"""Pruebas del clasificador de metadatos del corpus (src/corp_extractor/clasificador_corpus).

    python -m unittest discover -s tests -v

Los valores esperados se leen de las propias reglas del clasificador (REGLAS_ETL), así que si
cambias una categoría o un nivel de autoridad en las reglas, estas pruebas siguen siendo válidas.
"""
from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.corp_extractor.clasificador_corpus import clasificador_metadatos_v1 as clf

REGLAS = clf.REGLAS_ETL
MAPEO_LIBROS = REGLAS["CANONICO"]["mapeo_interno"]
RUTA_CIC_CRUDO = Path("src/corp_extractor/extractores/data/corpus_derecho_canonico_v1.json")


def clasificar(fuente: str, items: list[dict]) -> list[dict]:
    """Corre clasificar_archivo sobre un archivo temporal y devuelve los items clasificados."""
    with tempfile.TemporaryDirectory() as d:
        entrada, salida = Path(d) / "entrada.json", Path(d) / "salida.json"
        entrada.write_text(json.dumps({"fuente": fuente, "datos": items}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            clf.clasificar_archivo(str(entrada), str(salida))
        return json.loads(salida.read_text(encoding="utf-8"))["datos"]


def canon(numero: int, contexto: str) -> dict:
    return {"canon_id": numero, "contexto_jerarquico": contexto, "texto": f"Texto del canon {numero}.", "url_origen": "u"}


def numeral(numero: int, contexto: str) -> dict:
    return {"numeral_id": numero, "contexto_jerarquico": contexto, "texto": f"Texto del numeral {numero}.", "url_origen": "u"}


class TestCategoriaDelCodigoPorLibro(unittest.TestCase):
    FUENTE = REGLAS["CANONICO"]["condicion_fuente"]

    def test_cada_libro_recibe_su_propia_categoria(self):
        for libro, categoria in MAPEO_LIBROS.items():
            with self.subTest(libro=libro):
                [item] = clasificar(self.FUENTE, [canon(1, f"{libro}: DE ALGO (Cann. 1 – 2)")])
                self.assertEqual(item["categoria"], categoria)

    def test_libro_ii_no_se_confunde_con_libro_i(self):
        # Error original: "LIBRO I" está contenido en "LIBRO II", "LIBRO III"... y todos caían en el Libro I.
        [item] = clasificar(self.FUENTE, [canon(204, "LIBRO II: DEL PUEBLO DE DIOS (Cann. 204 – 746) > PARTE I: DE LOS FIELES")])
        self.assertEqual(item["categoria"], MAPEO_LIBROS["LIBRO II"])
        self.assertNotEqual(item["categoria"], MAPEO_LIBROS["LIBRO I"])

    def test_libros_con_prefijo_comun_no_se_mezclan(self):
        # V ⊂ VI ⊂ VII  e  I ⊂ II ⊂ III ⊂ IV
        casos = {"LIBRO V": "LIBRO V", "LIBRO VI": "LIBRO VI", "LIBRO VII": "LIBRO VII", "LIBRO III": "LIBRO III", "LIBRO IV": "LIBRO IV"}
        items = [canon(i, f"{libro}: TÍTULO") for i, libro in enumerate(casos, start=1)]
        for item, libro in zip(clasificar(self.FUENTE, items), casos.values()):
            self.assertEqual(item["categoria"], MAPEO_LIBROS[libro], libro)

    def test_el_libro_puede_no_estar_al_inicio_del_contexto(self):
        [item] = clasificar(self.FUENTE, [canon(1055, "CÓDIGO > LIBRO IV: DE LA FUNCIÓN DE SANTIFICAR (Cann. 834 – 1253)")])
        self.assertEqual(item["categoria"], MAPEO_LIBROS["LIBRO IV"])

    def test_la_autoridad_es_la_de_las_reglas_y_no_cambia_con_el_libro(self):
        items = [canon(i, f"{libro}: X") for i, libro in enumerate(MAPEO_LIBROS, start=1)]
        esperada = REGLAS["CANONICO"]["autoridad_default"]
        self.assertEqual({i["nivel_autoridad"] for i in clasificar(self.FUENTE, items)}, {esperada})

    def test_todo_canon_termina_con_una_categoria_de_las_reglas(self):
        validas = set(MAPEO_LIBROS.values())
        items = [canon(i, f"{libro}: X") for i, libro in enumerate(MAPEO_LIBROS, start=1)]
        self.assertTrue({i["categoria"] for i in clasificar(self.FUENTE, items)} <= validas)


class TestOtrasFuentesNoCambian(unittest.TestCase):
    def test_documento_del_concilio_con_regla_propia(self):
        for nombre, regla in REGLAS["VATICANO_II"]["mapeo_documentos"].items():
            with self.subTest(documento=nombre):
                [item] = clasificar("Constituciones - Concilio Vaticano II", [numeral(1, f"Concilio Vaticano II > Constituciones > {nombre}")])
                self.assertEqual((item["categoria"], item["nivel_autoridad"]), (regla["categoria"], regla["autoridad"]))

    def test_documento_del_concilio_sin_regla_usa_el_valor_por_defecto(self):
        reglas = REGLAS["VATICANO_II"]
        [item] = clasificar("Decretos - Concilio Vaticano II", [numeral(1, "Concilio Vaticano II > Decretos > Documento Inexistente")])
        self.assertEqual((item["categoria"], item["nivel_autoridad"]), (reglas["categoria_default"], reglas["autoridad_default"]))

    def test_compendio_usa_su_valor_por_defecto(self):
        reglas = REGLAS["COMPENDIO"]
        [item] = clasificar(reglas["condicion_fuente"], [numeral(1, "Compendio > Parte I")])
        self.assertEqual((item["categoria"], item["nivel_autoridad"]), (reglas["categoria_default"], reglas["autoridad_default"]))


@unittest.skipUnless(RUTA_CIC_CRUDO.exists(), "no está el corpus crudo del Código en esta máquina")
class TestConElCorpusRealDelCodigo(unittest.TestCase):
    def test_los_1752_canones_reciben_la_categoria_de_su_libro(self):
        crudo = json.loads(RUTA_CIC_CRUDO.read_text(encoding="utf-8-sig"))
        resultado = clasificar(crudo["fuente"], crudo["datos"])
        self.assertEqual(len(resultado), len(crudo["datos"]))
        sin_libro = []
        for item in resultado:
            m = re.match(r"\s*(LIBRO [IVX]+)\b", item["contexto_jerarquico"])
            if not m:
                sin_libro.append(item["canon_id"])
                continue
            self.assertEqual(item["categoria"], MAPEO_LIBROS[m.group(1)], f"canon {item['canon_id']} ({m.group(1)})")
        self.assertEqual(sin_libro, [], "cánones cuyo contexto no empieza con LIBRO ...")
        # Con el error original los 1752 caían en una sola categoría.
        self.assertGreater(len({i["categoria"] for i in resultado}), 1)


if __name__ == "__main__":
    unittest.main()
