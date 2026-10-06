"""Sesión de expansión dinámica: indexa PDFs externos de forma AISLADA y TRANSITORIA.

Garantías de diseño (tesis, secciones 1.4 y 2.2.8):

  * Aislamiento: los fragmentos viven en una colección propia de un cliente Chroma EN
    MEMORIA (EphemeralClient). Nunca se escribe en `chroma_dir` ni en la colección
    `corpus_canonico`, así que el corpus base no puede contaminarse ni alterarse.
  * Transitoriedad: al cerrar la sesión (o terminar el proceso) todo desaparece.
  * Comparabilidad: se usa la MISMA función de embedding y el MISMO espacio (coseno)
    que el corpus base ("regla de oro" del README), de modo que las similitudes de
    ambas fuentes se pueden ordenar juntas sin sesgo.
  * Mismo contrato: `consultar()` devuelve dicts con las mismas claves que
    `src.base_datos.chromaDB.consultar()`, más `origen` y la página, para que el
    cálculo del IVR trate ambas fuentes por igual.

Esta clase es la única parte del módulo que depende de Chroma/embeddings; los imports
son perezosos para poder probarla con dobles de prueba.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from .errores import NivelAutoridadInvalidoError
from .extractor_pdf import FuentePDF
from .parametros import ParametrosExpansion
from .procesamiento import procesar_pdf

log = logging.getLogger(__name__)

TIPO_CORPUS = "expansion_dinamica"
NIVELES_AUTORIDAD = (0.33, 0.66, 1.0)   # Nivel 1 / 2 / 3 según Rol.pdf
_SEPARADOR_CONTEXTO = " > "             # igual que el corpus base


@dataclass
class ResumenDocumento:
    """Lo que la interfaz muestra tras cargar un PDF."""
    doc_id: str
    nombre: str
    paginas: int
    fragmentos: int
    esquema: str                          # articulado | numerado | ventanas
    nivel_autoridad: float
    categoria: str
    advertencias: list[str] = field(default_factory=list)
    ya_cargado: bool = False


def validar_nivel_autoridad(nivel: float) -> float:
    for valido in NIVELES_AUTORIDAD:
        if abs(float(nivel) - valido) < 1e-9:
            return valido
    raise NivelAutoridadInvalidoError(
        f"nivel_autoridad debe ser uno de {NIVELES_AUTORIDAD} (recibido: {nivel})."
    )


@lru_cache(maxsize=1)
def _cliente_efimero(telemetria: bool):
    """Un único cliente en memoria por proceso; cada sesión usa su propia colección."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=telemetria))


def _etiqueta_paginas(inicio: int, fin: int) -> str:
    return f"p. {inicio}" if inicio == fin else f"pp. {inicio}–{fin}"


class SesionExpansion:
    """Conjunto de documentos externos cargados durante una sesión de validación.

    Uso típico (p. ej. en Streamlit, guardada en st.session_state):

        sesion = SesionExpansion()
        resumen = sesion.cargar_pdf(archivo_subido, nivel_autoridad=0.33)
        evidencias = sesion.consultar_combinado("El obispo puede ...", n_resultados=5)
        sesion.cerrar()
    """

    def __init__(
        self,
        params: ParametrosExpansion | None = None,
        funcion_embedding: Any | None = None,
        cliente: Any | None = None,
        sesion_id: str | None = None,
    ):
        self._params = params
        self._funcion_embedding = funcion_embedding
        self._cliente = cliente
        self.sesion_id = sesion_id or uuid.uuid4().hex[:12]
        self._coleccion_obj = None
        self._documentos: dict[str, ResumenDocumento] = {}

    # ------------------------------------------------------------------ infraestructura

    @property
    def params(self) -> ParametrosExpansion:
        if self._params is None:
            try:
                self._params = ParametrosExpansion.desde_config()
            except ImportError:           # fuera del proyecto (p. ej. pruebas) → valores por defecto
                self._params = ParametrosExpansion()
        return self._params

    @property
    def nombre_coleccion(self) -> str:
        return f"exp_{self.sesion_id}"

    def _coleccion(self):
        if self._coleccion_obj is None:
            if self._funcion_embedding is None:
                from src.base_datos.embeddings import obtener_funcion_embedding
                self._funcion_embedding = obtener_funcion_embedding()
            cliente = self._cliente or _cliente_efimero(self.params.telemetria)
            self._cliente = cliente
            self._coleccion_obj = cliente.get_or_create_collection(
                name=self.nombre_coleccion,
                embedding_function=self._funcion_embedding,
                metadata={"hnsw:space": self.params.espacio_hnsw},
            )
        return self._coleccion_obj

    # ------------------------------------------------------------------ carga

    def cargar_pdf(
        self,
        fuente: FuentePDF,
        nombre: str | None = None,
        *,
        nivel_autoridad: float | None = None,
        categoria: str | None = None,
        url_origen: str | None = None,
    ) -> ResumenDocumento:
        """Procesa e indexa un PDF. Si falla a mitad de camino, no deja nada a medio cargar.

        Args:
            nivel_autoridad: 0.33 (norma local, por defecto), 0.66 o 1.0.
            categoria: etiqueta temática; por defecto "Normativa Externa (carga dinámica)".
            url_origen: opcional, para mostrarla en la cita.

        Raises:
            ExpansionError (y subclases) con un mensaje apto para mostrar al usuario.
        """
        nivel = validar_nivel_autoridad(
            self.params.nivel_autoridad_default if nivel_autoridad is None else nivel_autoridad
        )
        categoria = categoria or self.params.categoria_default

        doc = procesar_pdf(fuente, nombre=nombre, params=self.params)

        if doc.doc_id in self._documentos:
            previo = self._documentos[doc.doc_id]
            return ResumenDocumento(**{**previo.__dict__, "ya_cargado": True})

        nombre_doc = doc.nombre.replace(_SEPARADOR_CONTEXTO, " - ")
        ids, textos, metas = [], [], []
        for i, f in enumerate(doc.segmentacion.fragmentos):
            ids.append(f"{TIPO_CORPUS}:{doc.doc_id}:{i:04d}")
            textos.append(f.texto)
            contexto = _SEPARADOR_CONTEXTO.join(["Documento cargado", nombre_doc, *f.contexto])
            if f.partes_total > 1:
                contexto += f" (parte {f.parte}/{f.partes_total})"
            meta = {
                "fuente": nombre_doc,
                "tipo_corpus": TIPO_CORPUS,
                "id_norma": int(f.id_norma),
                "contexto_jerarquico": contexto,
                "categoria": categoria,
                "nivel_autoridad": nivel,
                "doc_id": doc.doc_id,
                "etiqueta": f.etiqueta,
                "pagina_inicio": int(f.pagina_inicio),
                "pagina_fin": int(f.pagina_fin),
            }
            if url_origen:                      # Chroma no admite None en metadatos
                meta["url_origen"] = url_origen
            metas.append(meta)

        coleccion = self._coleccion()
        lote = max(1, self.params.batch_size)
        try:
            for i in range(0, len(ids), lote):
                coleccion.upsert(ids=ids[i:i + lote], documents=textos[i:i + lote], metadatas=metas[i:i + lote])
        except Exception:
            log.exception("Falló la indexación de %s; se revierte la carga.", nombre_doc)
            try:
                coleccion.delete(where={"doc_id": doc.doc_id})
            except Exception:
                log.exception("No se pudo revertir la carga parcial de %s.", nombre_doc)
            raise

        resumen = ResumenDocumento(
            doc_id=doc.doc_id,
            nombre=nombre_doc,
            paginas=doc.total_paginas,
            fragmentos=len(ids),
            esquema=doc.segmentacion.esquema,
            nivel_autoridad=nivel,
            categoria=categoria,
            advertencias=list(doc.advertencias),
        )
        self._documentos[doc.doc_id] = resumen
        return resumen

    # ------------------------------------------------------------------ gestión

    def documentos(self) -> list[ResumenDocumento]:
        return list(self._documentos.values())

    @property
    def total_fragmentos(self) -> int:
        return sum(d.fragmentos for d in self._documentos.values())

    def eliminar_documento(self, doc_id: str) -> bool:
        if doc_id not in self._documentos:
            return False
        self._coleccion().delete(where={"doc_id": doc_id})
        del self._documentos[doc_id]
        return True

    def cerrar(self) -> None:
        """Descarta todo lo cargado. Es idempotente."""
        if self._coleccion_obj is not None and self._cliente is not None:
            try:
                self._cliente.delete_collection(self.nombre_coleccion)
            except Exception:
                log.exception("No se pudo eliminar la colección temporal %s.", self.nombre_coleccion)
        self._coleccion_obj = None
        self._documentos.clear()

    def __enter__(self) -> "SesionExpansion":
        return self

    def __exit__(self, *exc) -> None:
        self.cerrar()

    # ------------------------------------------------------------------ consulta

    def consultar(self, premisa: str, n_resultados: int = 5, doc_id: str | None = None) -> list[dict]:
        """Busca solo en los documentos cargados. Mismo contrato que `chromaDB.consultar()`."""
        if not isinstance(premisa, str) or not premisa.strip():
            log.warning("No se ingresó una premisa válida para la consulta")
            return []
        try:
            n_resultados = int(n_resultados)
        except (TypeError, ValueError):
            log.error("El número de resultados debe ser un entero")
            return []
        if n_resultados <= 0 or not self._documentos:
            return []

        coleccion = self._coleccion()
        total = coleccion.count()
        if total == 0:
            return []

        resultado = coleccion.query(
            query_texts=[premisa],
            n_results=min(n_resultados, total, self.params.max_results),
            where={"doc_id": doc_id} if doc_id else None,
            include=["metadatas", "documents", "distances"],
        )

        contratos = []
        for id_chroma, meta, texto, distancia in zip(
            resultado["ids"][0], resultado["metadatas"][0], resultado["documents"][0], resultado["distances"][0]
        ):
            documento = meta["fuente"]
            paginas = _etiqueta_paginas(meta["pagina_inicio"], meta["pagina_fin"])
            contratos.append({
                # --- mismo contrato que chromaDB.consultar() ---
                "id_chroma": id_chroma,
                "texto": texto,
                "similitud": max(1 - distancia, 0),
                "contexto": meta["contexto_jerarquico"],
                "fuente": documento,
                "id_norma": meta["id_norma"],
                "tipo_corpus": meta["tipo_corpus"],
                "categoria": meta["categoria"],
                "nivel_autoridad": meta["nivel_autoridad"],
                "url_origen": meta.get("url_origen") or "Sin url registrada",
                "cita": f"{documento}, {meta['etiqueta']}, {paginas}",
                # --- extras de la expansión dinámica ---
                "origen": TIPO_CORPUS,
                "doc_id": meta["doc_id"],
                "pagina_inicio": meta["pagina_inicio"],
                "pagina_fin": meta["pagina_fin"],
            })
        return contratos

    def consultar_combinado(
        self,
        premisa: str,
        n_resultados: int = 5,
        where_base: dict | None = None,
        consultar_base: Callable[..., list[dict]] | None = None,
    ) -> list[dict]:
        """Corpus base + documentos cargados, ordenados juntos por similitud.

        Si no hay documentos cargados devuelve exactamente lo que devolvería el corpus base.
        `consultar_base` permite inyectar otra función (pruebas); por defecto se usa
        `src.base_datos.chromaDB.consultar`.
        """
        if consultar_base is None:
            from src.base_datos.chromaDB import consultar as consultar_base

        base = [{**r, "origen": "corpus_base"} for r in consultar_base(premisa, n_resultados, where_base)]
        extra = self.consultar(premisa, n_resultados)
        combinados = sorted(base + extra, key=lambda r: r["similitud"], reverse=True)
        return combinados[:n_resultados]
