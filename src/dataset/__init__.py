"""Generación del dataset de entrenamiento del Random Forest.

API pública:
    leer_urls(ruta_db, categorias)  → URLs de iglesia.db
    muestra_balanceada(urls, n)     → muestra repartida por categoría
    ventanas(texto, tam, solape)    → troceo en ventanas solapadas
    procesar_noticia(item, ...)     → noticia → IVR por ventana (diagnóstico)
"""
from .generador import (
    SOLAPE_PALABRAS,
    VENTANA_PALABRAS,
    NoticiaProcesada,
    UrlNoticia,
    leer_urls,
    muestra_balanceada,
    procesar_noticia,
    ventanas,
)

__all__ = [
    "leer_urls", "muestra_balanceada", "ventanas", "procesar_noticia",
    "UrlNoticia", "NoticiaProcesada", "VENTANA_PALABRAS", "SOLAPE_PALABRAS",
]
