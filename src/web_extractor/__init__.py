"""Extracción de contenido web (noticias/artículos) para el verificador.

API pública:
    extraer_contenido_url(url)   → texto principal de la noticia (str) o None
    extraer_contenido_html(html) → igual, pero a partir de HTML ya descargado
    extraer_metadata_url(url)    → {titulo, autor, fecha, idioma}
    extraer_metadata_html(html)  → igual, a partir de HTML (sin red)
    extraer_titular(url)         → titular de la noticia (premisa candidata)

Basado en trafilatura. No toca Chroma ni el IVR: es puro (URL/HTML → texto/metadatos).
"""
from .web_extractor import (
    extraer_contenido_html,
    extraer_contenido_url,
    extraer_metadata_html,
    extraer_metadata_url,
    extraer_titular,
)

__all__ = [
    "extraer_contenido_url",
    "extraer_contenido_html",
    "extraer_metadata_url",
    "extraer_metadata_html",
    "extraer_titular",
]