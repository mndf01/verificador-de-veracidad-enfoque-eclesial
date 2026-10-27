"""Extracción de contenido web (noticias/artículos) para el verificador.

Este módulo cierra el eslabón que alimenta el pipeline "URL → noticia → premisa":

    URL ──► web_extractor (extrae el texto) ──► premisa ──► IVR ──► RF ──► veredicto

Está construido sobre **trafilatura** (ya es dependencia en `pyproject.toml`), una
librería especializada en extraer texto de artículos/noticias: descarga la página,
identifica el contenido principal y descarta menús, anuncios, pies, etc. No hace OCR
ni navegación con JavaScript: su alcance son páginas HTML con contenido accesible.

La capa NO toca Chroma, el IVR ni el Random Forest. Es pura: recibe una URL (o un
HTML ya descargado) y devuelve texto/matadatos en estructuras planas de Python, igual
que el resto de los módulos desacoplados del proyecto. Así se puede probar sin red
(por ejemplo, con un pequeño HTML en memoria).

Uso desde terminal:

    .venv/bin/python -m src.web_extractor "https://www.vatican.va/content/..."
    .venv/bin/python -m src.web_extractor "https://..." --titulo     # solo el titular

Uso como función:

    from src.web_extractor import extraer_contenido_url, extraer_titular

    texto = extraer_contenido_url("https://...")
    titular = extraer_titular("https://...")
"""
from __future__ import annotations

import logging
import sys

import trafilatura

log = logging.getLogger(__name__)

# Límite defensivo: trafilatura ya trabaja bien con páginas grandes, pero evitamos
# que una URL maliciosa (o un falso positivo de "es HTML") consuma recursos sin límite.
# Si el HTML descargado supera este tamaño se devuelve None (el propio trafilatura
# por defecto ya recorta ~2 MB; este es un corte extra por seguridad/claridad).
_MAX_HTML_BYTES = 5_000_000


# ----------------------------------------------------------------------------------------
# Extracción del texto principal
# ----------------------------------------------------------------------------------------

def extraer_contenido_html(html: str) -> str | None:
    """Extrae el texto principal (en claro) de un documento HTML ya obtenido.

    Diseñado para poder probar la extracción SIN red: pasar el HTML crudo de un
    artículo y obtener solo su contenido útil. Útil también cuando quien descarga
    es otro componente (cache, proxy) y este módulo solo se encarga de limpiar.

    :param html: HTML crudo de la página (string).
    :return: texto del contenido principal, ya limpio y sin saltos de línea
        superfluos; o None si no se pudo extraer nada (página sin artículo,
        HTML inválido, `<body>` vacío, solo imágenes/sliders...).
    """
    if not html or not html.strip():
        log.warning("html vacío: no hay contenido que extraer")
        return None

    # El parsing de HTML (lxml) puede lanzar excepciones muy variadas; las capturamos
    # todas para que la extracción nunca reviente el orquestador.
    try:
        extraido = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,   # prioriza no meter ruido (viene de una url de noticia)
        )
    except Exception as e:
        log.error("Fallo al extraer HTML: %s", e)
        return None

    if not extraido:
        log.info("Trafilatura no encontró texto principal en el HTML")
        return None
    return extraido.strip() or None


def extraer_contenido_url(url: str) -> str | None:
    """Descarga una URL y extrae el texto del contenido principal.

    :param url: URL absoluta de la noticia/artículo (http/https).
    :return: el texto de la noticia ya limpio (str), o None si no se pudo
        descargar o extraer. Nunca lanza: ante cualquier error devuelve None y
        lo registra en el log, para que el orquestador decida cómo reportarlo.
    :raises: no lanza errores de red; los convierte en None con log.warning.
    """
    if not isinstance(url, str) or not url.strip():
        log.warning("URL inválida: %r", url)
        return None

    log.info("Descargando %s ...", url)
    try:
        descargado = trafilatura.fetch_url(url)
    except Exception as e:                 # trafilatura puede lanzar errores de red muy variados
        log.warning("Fallo de red al descargar %s: %s", url, e)
        return None

    if not descargado:
        log.warning("No se pudo descargar la URL (vacía, bloqueada o inaccesible): %s", url)
        return None

    if len(descargado.encode("utf-8")) > _MAX_HTML_BYTES:
        log.warning("HTML demasiado grande para %s; se descarta la extracción", url)
        return None

    contenido = extraer_contenido_html(descargado)
    if contenido:
        log.info("Extraídos %d caracteres de %s", len(contenido), url)
    return contenido


# ----------------------------------------------------------------------------------------
# Metadatos (titular, autor, fecha, idioma)
# ----------------------------------------------------------------------------------------

def extraer_metadata_html(html: str, url: str | None = None) -> dict:
    """Extrae metadatos básicos de un artículo a partir de su HTML.

    :return: dict con claves `titulo`, `autor`, `fecha`, `idioma`. Cualquier
        metadato ausente va como "" (cadena vacía), No None, para que sea fácil
        de volcar a JSON o mostrar en la interfaz sin manejar nulos.
    """
    if not html or not html.strip():
        return {"titulo": "", "autor": "", "fecha": "", "idioma": ""}

    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception as e:                 # metadata es frágil en HTMLs raros
        log.warning("No se pudieron extraer metadatos: %s", e)
        return {"titulo": "", "autor": "", "fecha": "", "idioma": ""}

    def _vacuo(valor: object) -> str:
        return str(valor).strip() if valor else ""

    return {
        "titulo": _vacuo(meta.title),
        "autor": _vacuo(meta.author),
        "fecha": _vacuo(meta.date),
        "idioma": _vacuo(meta.language),
    }


def extraer_metadata_url(url: str) -> dict:
    """Descarga una URL y devuelve sus metadatos (igual contrato que extraer_metadata_html)."""
    if not isinstance(url, str) or not url.strip():
        return {"titulo": "", "autor": "", "fecha": "", "idioma": ""}
    try:
        descargado = trafilatura.fetch_url(url)
    except Exception as e:
        log.warning("Fallo de red en extraer_metadata_url(%s): %s", url, e)
        return {"titulo": "", "autor": "", "fecha": "", "idioma": ""}
    if not descargado:
        return {"titulo": "", "autor": "", "fecha": "", "idioma": ""}
    return extraer_metadata_html(descargado, url=url)


def extraer_titular(url: str) -> str | None:
    """Conveniencia: devuelve el titular de la noticia (premisa candidata).

    Pensado para que la interfaz proponga el titular como premisa al usuario.
    :return: el título si se pudo extraer, o None si no.
    """
    meta = extraer_metadata_url(url)
    return meta["titulo"] or None


# ----------------------------------------------------------------------------------------
# Ejecución como módulo (prueba desde terminal)
# ----------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extrae el contenido de una noticia desde su URL.",
        epilog='Ej: .venv/bin/python -m src.web_extractor "https://..." --titulo',
    )
    ap.add_argument("url", help="URL absoluta de la noticia/artículo.")
    ap.add_argument("--titulo", action="store_true", help="Mostrar solo el titular.")
    ap.add_argument("--metadata", action="store_true", help="Mostrar también los metadatos.")
    args = ap.parse_args(argv)

    if args.titulo:
        titulo = extraer_titular(args.url)
        print(titulo if titulo else "(sin titular)")
        return 0

    print(f"\nURL : {args.url}\n")
    contenido = extraer_contenido_url(args.url)
    if not contenido:
        print("✖ No se pudo extraer contenido de la URL.")
        return 1

    if args.metadata:
        meta = extraer_metadata_url(args.url)
        for clave, valor in meta.items():
            print(f"{clave:<8}: {valor}")
        print("\n" + "-" * 60)

    print(f"Contenido ({len(contenido)} caracteres / {len(contenido.split())} palabras):\n")
    print(contenido)
    return 0


if __name__ == "__main__":
    sys.exit(main())