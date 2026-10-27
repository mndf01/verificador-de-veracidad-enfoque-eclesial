"""Orquestador del verificador de veracidad (capa lógica, sin I/O de terminal).

Une en un solo flujo las piezas desacopladas del proyecto:

    URL ──► web_extractor ──► premisa ──► RAG (corpus base + opcional PDF
         de expansión dinámica) ──► IVR ──► Random Forest ──► veredicto

Esta capa es PURA respecto a la terminal: no hace `input()` ni `print()` (salvo
mensajes de estado vía un logger opcional). Así se puede probar con dobles
(igual que `test_web_extractor`) y la interfaz (`src/cli/`) queda como una capa
fina de parseo de argumentos sobre estas funciones.

Toda función acepta inyectar sus dependencias reales (por ejemplo, inyectar una
función `extraer` falsa o un `calcular_ivr` falso) para poder testear el flujo
inclusive sin red y sin Chroma. Esto repite el patrón de dobles que ya usa el
resto del proyecto (`ColeccionFalsa`, `ClienteFalso`, `fetch_url`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# Asignaciones de nivel_autoridad numérico → etiqueta legible (para la presentación).
NIVEL_AUTORIDAD_LABEL = {0.33: "Nivel 1 (local)", 0.66: "Nivel 2 (nacional)", 1.0: "Nivel 3 (universal)"}

# Veredictos posibles del Random Forest (coinciden con rf.CLASES_VEREDICTO).
VEREDICTOS = ("consistente", "inconsistente", "no_verificable")

# Etiquetas en español para la interfaz.
VEREDICTO_ESP = {
    "consistente": "consistente ✓",
    "inconsistente": "inconsistente ✗",
    "no_verificable": "no verificable (sin evidencia suficiente) ?",
}

# Mapeo de sentencia de usuario según el veredicto (mensaje y tono).
_SENTENCIA = {
    "consistente": "La afirmación es compatible con la doctrina/material consultado.",
    "inconsistente": "La afirmación contradice o no se respalda en el material consultado.",
    "no_verificable": "No hay evidencia suficiente en el corpus para pronunciarse.",
}


@dataclass
class ResultadoEvaluacion:
    """Informe completo que devuelve `evaluar_noticia()` para mostrar en la interfaz."""

    premisa: str
    titular_sugerido: str | None = None
    url: str | None = None
    ok: bool = False
    veredicto: str | None = None          # "consistente" | "inconsistente" | "no_verificable"
    probabilidad: float | None = None     # confianza del RF en la clase ganadora
    probabilidades: dict[str, float] = field(default_factory=dict)
    informe_ivr: dict = field(default_factory=dict)
    errores: list[str] = field(default_factory=list)


def _etiqueta_veredicto(clase: str) -> str:
    """Convierte el veredicto crudo del RF en etiqueta en español para la interfaz."""
    if clase not in _SENTENCIA:
        return clase
    return VEREDICTO_ESP[clase]


def _mensaje_veredicto(clase: str) -> str:
    """Sentencia humana derivada del veredicto (para imprimir en grande)."""
    return _SENTENCIA.get(clase, "Veredicto desconocido.")


# ----------------------------------------------------------------------------------------
# Núcleo del flujo
# ----------------------------------------------------------------------------------------

def evaluar_noticia(
    url: str,
    premisa: str | None = None,
    ruta_pdf: str | Path | None = None,
    *,
    extraer_contenido: Callable[[str], str | None] | None = None,
    extraer_titular: Callable[[str], str | None] | None = None,
    calcular_ivr: Callable[[str], dict] | None = None,
    cargar_modelo: Callable[[str | Path], dict] | None = None,
    predecir: Callable[[object, object, list[float]], dict] | None = None,
    vector_caracteristicas: Callable[[dict], dict] | None = None,
    cargar_pdf_dinamico: Callable[[Path], object] | None = None,
    consultar_combinado: Callable[[str, int], list[dict]] | None = None,
    n_resultados: int = 5,
) -> ResultadoEvaluacion:
    """Evalúa una noticia completa: URL → texto → premisa → IVR → RF → veredicto.

    Es la función maestra que la interfaz (`src/cli/`) invoca. Acepta la premisa
    directamente (el que llama ya la ha confirmado/editado) o la deriva del
    titular extraído de la URL.

    :param url: URL absoluta de la noticia.
    :param premisa: texto de la afirmación a evaluar. Si es None, se propone el
        titular extraído de la URL como premisa candidata.
    :param ruta_pdf: opcional; PDF a cargar en sesión de expansión dinámica para
        ampliar el corpus de consulta (consultar_combinado).
    :param ...: funciones inyectables (para pruebas y para personalizar I/O).
    :return: ResultadoEvaluacion con todo lo necesario para presentar el informe.
    """
    resultado = ResultadoEvaluacion(url=url, premisa=premisa or "")

    # 0. Función por defecto: extraer texto de la URL.
    if extraer_contenido is None:
        from src.web_extractor import extraer_contenido_url as extraer_contenido

    # 1. Obtener la premisa (confirmada o derivada del titular).
    texto = extraer_contenido(url)
    if not texto and not premisa:
        resultado.errores.append(
            "No se pudo extraer contenido de la URL, y no se proporcionó una premisa manual."
        )
        return resultado

    if premisa is None:
        # Deriva la premisa del titular (la interfaz la confirma/edita ANTES de llamar aquí).
        if extraer_titular is None:
            from src.web_extractor import extraer_titular as extraer_titular
        titular = extraer_titular(url)
        resultado.titular_sugerido = titular
        if not titular:
            resultado.errores.append(
                "No se pudo extraer un titular de la URL para usar como premisa."
            )
            return resultado
        premisa = titular
        resultado.premisa = premisa

    # 2. Calcular el IVR sobre la premisa (recupera evidencias del corpus base).
    if calcular_ivr is None:
        from src.vectorizacion.ivr import calcular_ivr as calcular_ivr
    informe = calcular_ivr(premisa) if isinstance(premisa, str) else None
    if not informe or not informe.get("ok"):
        motivo = (informe or {}).get("motivo", "informe inválido")
        resultado.errores.append(f"El IVR no encontró evidencias para evaluar la premisa ({motivo}).")
        resultado.informe_ivr = informe or {}
        return resultado

    # 2b. Expansión dinámica con PDF (pendiente de integración total).
    #     Actualmente dejamos constancia de que se solicitó un PDF, pero el IVR
    #     se calcula sobre el corpus base. La integración real del PDF en el IVR
    #     (mediante consultar_combinado + recalcular) se abordará en una fase
    #     posterior; el CLI ya expone el hook `--pdf` para no romper la firma.
    if ruta_pdf is not None:
        from pathlib import Path as _P
        resultado.informe_ivr["pdf_pendiente"] = str(_P(ruta_pdf))
        log.info("PDF pendiente de combinar en el IVR (fase posterior): %s", ruta_pdf)

    resultado.informe_ivr = dict(informe)
    resultado.informe_ivr["premisa"] = premisa

    # 3. Vector de características para el Random Forest.
    if vector_caracteristicas is None:
        from src.vectorizacion.ivr import vector_caracteristicas as vector_caracteristicas
    caracteristicas = vector_caracteristicas(informe)
    if not caracteristicas:
        resultado.errores.append("No se pudo construir el vector de características del IVR.")
        return resultado

    # 4. Cargar el modelo RF y predecir el veredicto.
    if cargar_modelo is None:
        from src.modelo.rf import cargar as cargar_modelo
    modelo_fallido = False
    try:
        modelo = cargar_modelo(modelo_ruta_por_defecto())
    except Exception as e:
        modelo_fallido = True
        resultado.errores.append(
            f"No se pudo cargar el modelo Random Forest ({type(e).__name__}: {e}). "
            "Ejecuta `verificador entrenar` primero."
        )
    if modelo_fallido:
        return resultado

    # 5. Construir el vector fila (orden estable de NOMBRES_CARACTERISTICAS).
    from src.modelo.rf import NOMBRES_CARACTERISTICAS
    x_row = [float(caracteristicas.get(nombre, 0.0)) for nombre in NOMBRES_CARACTERISTICAS]

    if predecir is None:
        from src.modelo.rf import predecir as predecir
    try:
        pred = predecir(modelo["clf"], modelo["le"], x_row)
    except Exception as e:
        resultado.errores.append(f"Error al predecir el veredicto ({type(e).__name__}: {e}).")
        return resultado

    resultado.ok = True
    resultado.veredicto = pred.get("clase")
    resultado.probabilidad = pred.get("probabilidad")
    resultado.probabilidades = pred.get("probabilidades", {})
    return resultado


def modelo_ruta_por_defecto() -> str | Path:
    """Ruta por defecto del modelo RF persistido, anclada a la raíz del proyecto."""
    from config import CONFIG_PATH
    return CONFIG_PATH.parent / "data" / "modelo_rf.joblib"


# ----------------------------------------------------------------------------------------
# Diagnóstico y helpers para la interfaz
# ----------------------------------------------------------------------------------------

def _nivel_label(nivel) -> str:
    try:
        return NIVEL_AUTORIDAD_LABEL.get(float(nivel), f"Nivel {nivel}")
    except (TypeError, ValueError):
        return "Nivel desconocido"


def veredicto_para_interfaz(resultado: ResultadoEvaluacion) -> str:
    """Devuelve la línea de veredicto formateada que mostrará la interfaz."""
    if not resultado.ok or not resultado.veredicto:
        return "Sin veredicto."
    return f"{_etiqueta_veredicto(resultado.veredicto)} — {_mensaje_veredicto(resultado.veredicto)}"