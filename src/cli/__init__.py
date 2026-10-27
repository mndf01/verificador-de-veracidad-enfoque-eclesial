"""Interfaz de terminal (CLI) para el verificador de veracidad.

Capa DELGADA de parseo de argumentos sobre `src.orquestador`: reúne en un solo
comando (`verificador` / `python -m src.cli`) toda la funcionalidad suelta que
antes había que lanzar módulo por módulo.

Comandos:
    verificador evaluar URL [--pdf RUTA] [--premisa TEXTO] [--no-interactivo]
        Evalúa una noticia por URL → premisa → IVR → RF → veredicto.
    verificador setup [--modo estatico|dinamico]
        Ejecuta el ETL + indexa el corpus en Chroma (desbloquea data/ vacío).
    verificador dataset-generar --salida CSV
        Genera un CSV de entrenamiento (16 características + veredicto).
    verificador entrenar [--dataset CSV] [--salida joblib]
        Entrena y persiste el Random Forest.

El CLI es un orquestador de INTENCIÓN: la lógica de negocio vive en
`src/orquestador.py`; aquí solo se parsea, se preguntan confirmaciones y se
imprimen resultados.
"""
from .cli import main

__all__ = ["main"]