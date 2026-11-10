"""Pruebas unitarias para `src.modelo.baseline`."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modelo.baseline import (
    calibrar_umbrales_baseline,
    clasificar_por_umbrales,
    evaluar_comparativa,
    generar_graficos,
    prueba_mcnemar,
)
from src.modelo.rf import CARACTERISTICAS_SIN_IVR


def _dataset_sintetico() -> pd.DataFrame:
    filas = []
    dominios = ["medio_a.com", "medio_b.com", "medio_c.com", "medio_d.com"]
    for i in range(40):
        dom = dominios[i % len(dominios)]
        if i < 10:
            veredicto = "proximo"
            sb_max = 0.78
            lit = 0.45
            so = 0.60
        elif i < 25:
            veredicto = "distante"
            sb_max = 0.72 if i % 2 == 0 else 0.55  # simula falsos positivos del baseline por umbral fijo
            lit = 0.05
            so = 0.10
        else:
            veredicto = "no_definido"
            sb_max = 0.40
            lit = 0.01
            so = 0.0
        fila = {col: 0.1 for col in CARACTERISTICAS_SIN_IVR}
        fila.update({
            "sb_max": sb_max,
            "sb_medio": sb_max - 0.05,
            "literal_max": lit,
            "literal_medio": lit,
            "ontologico_max": so,
            "so_disponible": 1.0 if so > 0 else 0.0,
            "autoridad_max": 1.0,
            "dominio": dom,
            "veredicto": veredicto,
        })
        filas.append(fila)
    return pd.DataFrame(filas)


class TestBaseline(unittest.TestCase):
    def test_clasificar_por_umbrales(self):
        preds = clasificar_por_umbrales([0.75, 0.70, 0.55, 0.45, 0.30], tau_alta=0.70, tau_baja=0.45)
        self.assertEqual(preds, ["proximo", "proximo", "distante", "distante", "no_definido"])

    def test_calibrar_umbrales_baseline(self):
        df = _dataset_sintetico()
        t_alta, t_baja = calibrar_umbrales_baseline(df)
        self.assertGreaterEqual(t_alta, t_baja)
        self.assertGreater(t_alta, 0.0)

    def test_prueba_mcnemar(self):
        y_true = ["proximo"] * 25
        pred_h = ["proximo"] * 24 + ["distante"]
        pred_b = ["distante"] * 20 + ["proximo"] * 5
        res = prueba_mcnemar(y_true, pred_h, pred_b)
        self.assertGreater(res["b_hibrido_gana"], res["c_baseline_gana"])
        self.assertLess(res["p_value"], 0.001)
        self.assertTrue(res["significativo_p001"])

    def test_evaluar_comparativa_y_graficos(self):
        df = _dataset_sintetico()
        res = evaluar_comparativa(df)
        self.assertIn("baseline_fijo", res)
        self.assertIn("baseline_calibrado", res)
        self.assertIn("modelo_hibrido", res)
        self.assertIn("mcnemar_vs_fijo", res)

        with tempfile.TemporaryDirectory() as tmp:
            rutas = generar_graficos(df, tmp)
            self.assertEqual(len(rutas), 6)
            for ruta in rutas.values():
                self.assertTrue(Path(ruta).exists())
                self.assertGreater(Path(ruta).stat().st_size, 200)


if __name__ == "__main__":
    unittest.main()
