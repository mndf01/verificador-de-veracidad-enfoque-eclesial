"""Tests de `src.orquestador` (la capa lógica que une web_extractor → IVR → RF).

Siguiendo el patrón de `test_web_extractor` y `test_ivr`, se prueban los flujos
SIN red y SIN Chroma real inyectando dobles: funciones fake de `extraer_contenido`,
`extraer_titular`, `calcular_ivr`, `vector_caracteristicas`, `cargar_modelo` y
`predecir`. Así se valida la orquestación (orden, manejo de errores, contrato del
ResultadoEvaluacion) sin depender de servicios externos.

No se ejecuta ningún pipeline real: solo se verifica la lógica de enlace.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path as _P

# Anclar la raíz del proyecto para importar `src.orquestador` en tests aislados.
_RAIZ = _P(__file__).resolve().parents[1]
if _RAIZ not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src import orquestador  # noqa: E402


# ----------------------------------------------------------------------------------------
# Dobles (fakes) para no tocar red / Chroma / modelo
# ----------------------------------------------------------------------------------------
_EVIDENCIA = {
    "id_chroma": "x",
    "texto": "La Iglesia Católica enseña que el bautismo es necesario para la salvación.",
    "similitud": 0.75,
    "contexto": "CIC > Libro IV",
    "fuente": "CIC",
    "id_norma": 849,
    "tipo_corpus": "corpus_canones",
    "categoria": "sacramentos",
    "nivel_autoridad": 1.0,
    "url_origen": "https://www.vatican.va/",
    "cita": "CIC, canon 849",
}


def _informe_fake(premisa: str) -> dict:
    """Informe IVR sintético con la forma que consume el RF (vector_caracteristicas)."""
    ev = {**_EVIDENCIA, "senales": {"sb": 0.5, "literal": 0.3, "ontologico": 0.6, "autoridad": 1.0,
                                    "terminos_compartidos": 0.5}, "ivr": 0.6}
    return {
        "ok": True,
        "premisa": premisa,
        "n_evidencias": 1,
        "pesos": {"sb": 0.4, "literal": 0.25, "ontologico": 0.2, "autoridad": 0.15},
        "conceptos_premisa": {"bautismo": "sacramentos"},
        "ivr": 0.6, "ivr_max": 0.6, "ivr_medio": 0.6, "ivr_min": 0.6,
        "mejor": ev,
        "senales_max": {"sb": 0.5, "literal": 0.3, "ontologico": 0.6, "autoridad": 1.0,
                        "terminos_compartidos": 0.5},
        "evidencias": [ev],
    }


def _vector_real(informe: dict) -> dict:
    from src.modelo.rf import NOMBRES_CARACTERISTICAS
    return {c: 0.5 for c in NOMBRES_CARACTERISTICAS}


def _modelo_fake():
    return {"clf": object(), "le": None}


def _pred_fake(clf, le, x_row):
    return {"clase": "consistente", "probabilidad": 0.91,
            "probabilidades": {"consistente": 0.91, "inconsistente": 0.05, "no_verificable": 0.04}}


class TestEvaluarNoticia(unittest.TestCase):
    """Flujo feliz y manejo de errores con todas las dependencias inyectadas."""

    def test_flujo_completo_con_premisa_manual(self):
        """Con una premisa explícita no se necesita extraer el titular."""
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com/noticia",
            premisa="El bautismo es necesario",
            extraer_contenido=lambda url: "texto de la noticia",
            calcular_ivr=_informe_fake,
            vector_caracteristicas=_vector_real,
            cargar_modelo=lambda _ruta: _modelo_fake(),
            predecir=_pred_fake,
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.veredicto, "consistente")
        self.assertAlmostEqual(res.probabilidad, 0.91)
        self.assertEqual(res.informe_ivr["n_evidencias"], 1)

    def test_deriva_premisa_del_titular(self):
        """Cuando no hay premisa, se usa el titular extraído (flujo automatizado)."""
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com/noticia",
            premisa=None,
            extraer_contenido=lambda url: "texto",
            extraer_titular=lambda url: "El papa se reúne con obispos",
            calcular_ivr=_informe_fake,
            vector_caracteristicas=_vector_real,
            cargar_modelo=lambda _ruta: _modelo_fake(),
            predecir=_pred_fake,
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.premisa, "El papa se reúne con obispos")
        self.assertEqual(res.titular_sugerido, "El papa se reúne con obispos")

    def test_falla_si_contenido_y_premisa_ausentes(self):
        """Sin texto extraído y sin premisa manual → error, ok=False."""
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com",
            premisa=None,
            extraer_contenido=lambda url: None,
            extraer_titular=lambda url: None,
        )
        self.assertFalse(res.ok)
        self.assertTrue(res.errores)

    def test_falla_sin_titular_desde_url(self):
        """Hay contenido pero no titular y no se pasó premisa → no se puede derivar."""
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com",
            premisa=None,
            extraer_contenido=lambda url: "texto",
            extraer_titular=lambda url: None,
        )
        self.assertFalse(res.ok)
        self.assertIn("titular", (res.errores[0] if res.errores else ""))

    def test_ivr_sin_evidencias_aborta_con_motivo(self):
        """Si el IVR devuelve ok=False, el orquestador informa el motivo."""
        def ivr_vacio(premisa):
            return {"ok": False, "premisa": premisa, "motivo": "sin evidencias recuperadas",
                    "n_evidencias": 0, "ivr": 0.0, "evidencias": []}
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com",
            premisa="premisa",
            extraer_contenido=lambda url: "texto",
            calcular_ivr=ivr_vacio,
        )
        self.assertFalse(res.ok)
        self.assertTrue(any("sin evidencias" in e for e in res.errores))
        self.assertFalse(res.informe_ivr.get("ok"))

    def test_sin_modelo_rf_reporta_error(self):
        """Si falta el modelo RF persistido, se reporta y sugiere entrenar."""
        def cargar_modelo_roto(ruta):
            raise FileNotFoundError("modelo no existe")
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com",
            premisa="premisa",
            extraer_contenido=lambda url: "texto",
            calcular_ivr=_informe_fake,
            vector_caracteristicas=_vector_real,
            cargar_modelo=cargar_modelo_roto,
        )
        self.assertFalse(res.ok)
        self.assertTrue(any("modelo" in e.lower() for e in res.errores))

    def test_errores_en_predecir_se_capturan(self):
        """Si predecir lanza, el orquestador no revienta y deja error."""
        def predecir_roto(clf, le, x):
            raise RuntimeError("fallo de prediccion")
        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com",
            premisa="premisa",
            extraer_contenido=lambda url: "texto",
            calcular_ivr=_informe_fake,
            vector_caracteristicas=_vector_real,
            cargar_modelo=lambda _ruta: _modelo_fake(),
            predecir=predecir_roto,
        )
        self.assertFalse(res.ok)
        self.assertTrue(any("predecir" in e.lower() for e in res.errores))

    def test_flujo_con_pdf_expansion_dinamica(self):
        """Cuando se pasa ruta_pdf, se carga el PDF y se inyecta consultar_combinado al IVR."""
        cargados = []
        usados = []

        def fake_cargar_pdf(ruta):
            cargados.append(str(ruta))
            return {"nombre": "Decreto 12/2024", "fragmentos": 9}

        def fake_consultar_combinado(premisa, n):
            return [_EVIDENCIA]

        def fake_ivr_con_fn(premisa, n_resultados=5, consultar_fn=None):
            usados.append(consultar_fn)
            return _informe_fake(premisa)

        res = orquestador.evaluar_noticia(
            url="https://ejemplo.com/noticia",
            premisa="El bautismo es necesario",
            ruta_pdf="decreto.pdf",
            extraer_contenido=lambda url: "texto",
            calcular_ivr=fake_ivr_con_fn,
            cargar_pdf_dinamico=fake_cargar_pdf,
            consultar_combinado=fake_consultar_combinado,
            vector_caracteristicas=_vector_real,
            cargar_modelo=lambda _ruta: _modelo_fake(),
            predecir=_pred_fake,
        )
        self.assertTrue(res.ok)
        self.assertEqual(len(cargados), 1)
        self.assertEqual(usados, [fake_consultar_combinado])
        self.assertIn("expansion_dinamica", res.informe_ivr)


class TestVeredictoParaInterfaz(unittest.TestCase):
    def test_mapeo_español(self):
        for clase, esperado in [("consistente", "consistente"), ("inconsistente", "inconsistente"),
                                ("no_verificable", "no verificable")]:
            res = orquestador.ResultadoEvaluacion(premisa="x", ok=True, veredicto=clase,
                                                  probabilidad=0.9)
            linea = orquestador.veredicto_para_interfaz(res)
            self.assertIn(esperado, linea.lower())

    def test_sin_veredicto_devuelve_sin_veredicto(self):
        res = orquestador.ResultadoEvaluacion(premisa="x", ok=True, veredicto=None)
        self.assertEqual(orquestador.veredicto_para_interfaz(res), "Sin veredicto.")


class TestNivelLabel(unittest.TestCase):
    def test_etiquetas(self):
        self.assertIn("universal", orquestador._nivel_label(1.0).lower())
        self.assertIn("constitución", orquestador._nivel_label(0.85).lower())
        self.assertIn("decreto", orquestador._nivel_label(0.70).lower())
        self.assertIn("compendio", orquestador._nivel_label(0.55).lower())
        self.assertIn("local", orquestador._nivel_label(0.33).lower())
        self.assertIn("Nivel 0.5", orquestador._nivel_label(0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
