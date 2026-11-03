"""Caché persistente de noticias ya procesadas por el IVR.

Motivación: descargar y calcular el IVR de una noticia cuesta ~5 s (dominado por
la descarga HTTP). Regenerar el dataset desde cero es caro e innecesario cuando
solo cambian los umbrales de etiquetado o la agregación de features.

Qué se cachea por URL:
  * el texto extraído de la noticia (evita volver a descargar), y
  * el detalle por ventana del IVR (evita recalcular las 16 queries a Chroma).

Qué NO se cachea: la etiqueta `veredicto`. Se recalcula en cada ejecución a
partir del detalle cacheado, porque depende de los umbrales, que se ajustan.

Formato: un JSONL (una línea JSON por noticia) para poder reanudar de forma
incremental sin reescribir el archivo entero.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .generador import NoticiaProcesada

RUTA_CACHE_DEFECTO = Path(__file__).resolve().parents[2] / "data" / "cache_ivr.jsonl"


class CacheIVR:
    """Caché en disco de noticias procesadas, indexada por URL."""

    def __init__(self, ruta: str | Path | None = None):
        self.ruta = Path(ruta) if ruta else RUTA_CACHE_DEFECTO
        self._datos: dict[str, dict] = {}
        self._cargado = False
        self._nuevos = 0

    # ------------------------------------------------------------------ carga / guardado

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
                    continue          # línea corrupta: se ignora y se regenera
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

    def obtener(self, url: str) -> NoticiaProcesada | None:
        """Devuelve la noticia cacheada, o None si no está."""
        self._cargar()
        registro = self._datos.get(url)
        if not registro:
            return None
        try:
            return NoticiaProcesada(
                url=registro["url"],
                categoria=registro.get("categoria", ""),
                titulo=registro.get("titulo", ""),
                dominio=registro.get("dominio", ""),
                ok=registro.get("ok", False),
                motivo=registro.get("motivo", ""),
                n_ventanas=registro.get("n_ventanas", 0),
                n_ventanas_con_senal=registro.get("n_ventanas_con_senal", 0),
                ivr_max=registro.get("ivr_max", 0.0),
                ivr_top3=registro.get("ivr_top3", 0.0),
                ivr_medio_ventanas=registro.get("ivr_medio_ventanas", 0.0),
                senales_max=registro.get("senales_max", {}),
                conceptos_premisa=registro.get("conceptos_premisa", {}),
                ventanas_detalle=registro.get("ventanas_detalle", []),
            )
        except (KeyError, TypeError):
            return None

    def guardar(self, res: NoticiaProcesada) -> None:
        """Añade (o actualiza) una noticia al caché y la persiste de inmediato."""
        self._cargar()
        registro = asdict(res)
        self._datos[res.url] = registro
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self._nuevos += 1

    def compactar(self) -> int:
        """Reescribe el archivo dejando una sola línea por URL. Devuelve las eliminadas."""
        self._cargar()
        if not self._datos:
            return 0
        # Cuenta las líneas actuales para informar de cuántas se colapsaron.
        total_lineas = sum(1 for _ in open(self.ruta, encoding="utf-8")) if self.ruta.exists() else 0
        tmp = self.ruta.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for registro in self._datos.values():
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        tmp.replace(self.ruta)
        return max(0, total_lineas - len(self._datos))

    def urls(self) -> list[str]:
        self._cargar()
        return list(self._datos)
