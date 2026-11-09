"""Caché permanente del TEXTO extraído de cada noticia — HERRAMIENTA DE ENTRENAMIENTO.

**Ámbito: solo `dataset-generar`.** Este caché NO forma parte del programa de uso
(evaluar / chat): el flujo de usuario descarga la noticia en el momento y no lee
ni escribe aquí. Existe únicamente para no repetir descargas mientras se construye
o recalcula el dataset de entrenamiento.

Motivación: el caché del IVR (`cache.py`) guarda los resultados del cálculo
(señales, IVR, conceptos detectados). Pero si cambia algo que afecta al cálculo
—el tesauro, la fórmula del IVR, la ventana deslizante— hay que recalcularlo todo,
y sin el texto original habría que **volver a descargar** cada noticia (~5 s cada
una; 2,5 h para 2000).

Este caché guarda el texto extraído una sola vez y lo reutiliza entre recálculos.

Separación de responsabilidades:

    cache_textos.jsonl   URL → texto + metadatos   (permanente, cambio de tesauro)
    cache_ivr.jsonl      URL → señales del IVR     (se invalida al cambiar el tesauro)

Formato: JSONL, una línea por URL. Permite reanudar de forma incremental.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

RUTA_CACHE_TEXTOS = Path(__file__).resolve().parents[2] / "data" / "cache_textos.jsonl"


class CacheTextos:
    """Caché en disco del texto extraído de noticias, indexado por URL."""

    def __init__(self, ruta: str | Path | None = None):
        self.ruta = Path(ruta) if ruta else RUTA_CACHE_TEXTOS
        self._datos: dict[str, dict] = {}
        self._cargado = False
        self.aciertos = 0          # cuántas veces se sirvió del caché
        self.fallos_memo = 0       # cuántas descargas fallaron (también se cachean)

    # ------------------------------------------------------------------ carga

    def _cargar(self) -> None:
        if self._cargado:
            return
        self._cargado = True
        if not self.ruta.exists():
            return
        with open(self.ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    continue            # línea corrupta: se ignora
                url = registro.get("url")
                if url:
                    self._datos[url] = registro

    def __len__(self) -> int:
        self._cargar()
        return len(self._datos)

    def __contains__(self, url: str) -> bool:
        self._cargar()
        return url in self._datos

    # ------------------------------------------------------------------ API

    def obtener(self, url: str) -> str | None:
        """Texto cacheado de la URL, o None si no está (o si se registró como fallo)."""
        self._cargar()
        registro = self._datos.get(url)
        if not registro or registro.get("texto") is None:
            return None
        self.aciertos += 1
        return registro["texto"]

    def ya_procesada(self, url: str) -> bool:
        """True si la URL ya se intentó (con éxito o con fallo)."""
        self._cargar()
        return url in self._datos

    def guardar(self, url: str, texto: str | None, **extra) -> None:
        """Registra el texto (o el fallo, con texto=None) y lo persiste."""
        self._cargar()
        registro = {"url": url, "texto": texto}
        registro.update(extra)
        self._datos[url] = registro
        if texto is None:
            self.fallos_memo += 1
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    def obtener_o_descargar(
        self,
        url: str,
        descargar: Callable[[str], str | None] | None = None,
    ) -> str | None:
        """Devuelve el texto del caché o lo descarga y lo guarda.

        Los fallos también se cachean: así no se reintenta un dominio bloqueado
        en cada recálculo.
        """
        if self.ya_procesada(url):
            return self.obtener(url)

        if descargar is None:
            from src.web_extractor import extraer_contenido_url as descargar

        texto = descargar(url)
        self.guardar(url, texto)
        return texto

    def compactar(self) -> int:
        """Reescribe el archivo dejando una línea por URL. Devuelve las eliminadas."""
        self._cargar()
        if not self._datos:
            return 0
        lineas = sum(1 for _ in open(self.ruta, encoding="utf-8")) if self.ruta.exists() else 0
        tmp = self.ruta.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for registro in self._datos.values():
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        tmp.replace(self.ruta)
        return max(0, lineas - len(self._datos))

    def urls(self) -> list[str]:
        self._cargar()
        return list(self._datos)

    def estadisticas(self) -> dict:
        """Resumen para la interfaz: cuántas con texto y cuántas fallidas."""
        self._cargar()
        con_texto = sum(1 for r in self._datos.values() if r.get("texto") is not None)
        return {
            "total": len(self._datos),
            "con_texto": con_texto,
            "fallidas": len(self._datos) - con_texto,
            "ruta": str(self.ruta),
        }
