"""Modo chat interactivo del verificador (Rich + prompt_toolkit).

Sesión de terminal para probar el modelo de forma conversacional:

    verificador chat                  # sesión normal
    verificador chat --pdf RUTA       # sesión con un PDF cargado (expansión dinámica)

Cada entrada del usuario se interpreta así:

  * una URL          → evalúa esa noticia (descarga y analiza el cuerpo)
  * `/comando`       → comandos de sesión (`/pdf`, `/historial`, `/ayuda`, `/salir`)
  * cualquier texto  → lo evalúa como afirmación a contrastar

La presentación vive en `ui.py` (Rich); aquí solo está el bucle, el parseo de
comandos y el estado de la sesión. La lógica de evaluación está en
`src/orquestador.py` — este módulo no calcula nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .ui import (
    crear_consola,
    mostrar_ayuda,
    mostrar_bienvenida,
    mostrar_historial,
    mostrar_info_modelo,
    mostrar_resultado,
)

# Prefijos que identifican una URL frente a una afirmación.
PREFIJOS_URL = ("http://", "https://", "www.")


# ----------------------------------------------------------------------------------------
# Entrada de texto (prompt_toolkit con fallback a input())
# ----------------------------------------------------------------------------------------

def _crear_sesion_prompt():
    """Crea la sesión de prompt_toolkit con historial y autocompletado de comandos.

    Devuelve None si prompt_toolkit no está disponible o no hay terminal
    interactiva: en ese caso el chat usa `input()` y sigue funcionando.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
    except ImportError:
        return None

    comandos = WordCompleter(
        ["/ayuda", "/historial", "/modelo", "/pdf", "/salir"],
        ignore_case=True,
        sentence=True,
    )
    try:
        return PromptSession(history=InMemoryHistory(), completer=comandos)
    except Exception:      # sin TTY (CI, pipes) → el llamador usará input()
        return None


def _leer_entrada(sesion, consola) -> str | None:
    """Lee una línea del usuario. None = fin de la sesión (EOF/Ctrl+D)."""
    if sesion is not None:
        try:
            return sesion.prompt("› ")
        except EOFError:
            return None
        except KeyboardInterrupt:      # Ctrl+C limpia la línea, no cierra la sesión
            consola.print("[dim](Ctrl+D o /salir para terminar)[/dim]")
            return ""
    try:
        return input("› ")
    except EOFError:
        return None
    except KeyboardInterrupt:
        return ""


# ----------------------------------------------------------------------------------------
# Estado de la sesión
# ----------------------------------------------------------------------------------------

class SesionChat:
    """Estado mutable de una sesión de chat."""

    def __init__(self, ruta_pdf: str | Path | None = None):
        self.ruta_pdf = Path(ruta_pdf) if ruta_pdf else None
        self.historial: list[dict] = []
        self.sesion_expansion = None      # SesionExpansion viva mientras hay PDF
        self._resumen_pdf = None

    # -- PDF ---------------------------------------------------------------

    def cargar_pdf(self, ruta: str | Path) -> object:
        """Carga (o recarga) un PDF en la sesión de expansión dinámica."""
        from src.expansion_dinamica import SesionExpansion

        if self.sesion_expansion is None:
            self.sesion_expansion = SesionExpansion()
        resumen = self.sesion_expansion.cargar_pdf(Path(ruta))
        self.ruta_pdf = Path(ruta)
        self._resumen_pdf = resumen
        return resumen

    def descargar_pdf(self) -> None:
        """Cierra la sesión de expansión dinámica (descarta el PDF)."""
        if self.sesion_expansion is not None:
            try:
                self.sesion_expansion.cerrar()
            except Exception:
                pass
        self.sesion_expansion = None
        self.ruta_pdf = None
        self._resumen_pdf = None

    def cerrar(self) -> None:
        self.descargar_pdf()

    # -- evaluación --------------------------------------------------------

    def evaluar(self, consulta: str):
        """Evalúa una URL o una afirmación usando el orquestador."""
        from src.orquestador import evaluar_noticia

        es_url = consulta.lower().startswith(PREFIJOS_URL)
        if es_url:
            return evaluar_noticia(
                consulta,
                ruta_pdf=self.ruta_pdf,
                consultar_combinado=(
                    self.sesion_expansion.consultar_combinado
                    if self.sesion_expansion is not None else None
                ),
            )
        # Texto libre: se evalúa tal cual como premisa (no se descarga nada).
        return evaluar_noticia(
            url="",
            premisa=consulta,
            extraer_contenido=lambda _url: consulta,
            ruta_pdf=self.ruta_pdf,
            consultar_combinado=(
                self.sesion_expansion.consultar_combinado
                if self.sesion_expansion is not None else None
            ),
        )

    def registrar(self, consulta: str, resultado) -> None:
        self.historial.append({
            "consulta": consulta,
            "veredicto": resultado.veredicto,
            "probabilidad": resultado.probabilidad,
        })


# ----------------------------------------------------------------------------------------
# Comandos
# ----------------------------------------------------------------------------------------

def _procesar_comando(consola, sesion: SesionChat, linea: str) -> bool:
    """Ejecuta un comando `/...`. Devuelve False si hay que terminar la sesión."""
    partes = linea.split(maxsplit=1)
    comando = partes[0].lower()
    argumento = partes[1].strip() if len(partes) > 1 else ""

    if comando in ("/salir", "/exit", "/quit"):
        return False

    if comando == "/ayuda":
        mostrar_ayuda(consola)

    elif comando == "/historial":
        mostrar_historial(consola, sesion.historial)

    elif comando == "/modelo":
        mostrar_info_modelo(consola, _info_modelo())

    elif comando == "/pdf":
        if not argumento:
            if sesion.ruta_pdf:
                consola.print(f"[magenta]PDF en sesión:[/magenta] {sesion.ruta_pdf}")
            else:
                consola.print("[dim]No hay PDF cargado. Uso: /pdf RUTA · /pdf off[/dim]")
        elif argumento.lower() in ("off", "ninguno", "quitar"):
            sesion.descargar_pdf()
            consola.print("[dim]PDF descargado de la sesión.[/dim]")
        else:
            try:
                resumen = sesion.cargar_pdf(argumento)
                consola.print(
                    f"[magenta]PDF cargado:[/magenta] {resumen.nombre} "
                    f"({resumen.fragmentos} fragmentos, nivel {resumen.nivel_autoridad})"
                )
                for aviso in getattr(resumen, "advertencias", []):
                    consola.print(f"[yellow]  ⚠ {aviso}[/yellow]")
            except Exception as e:
                consola.print(f"[red]No se pudo cargar el PDF:[/red] {e}")

    else:
        consola.print(f"[red]Comando desconocido:[/red] {comando}. Prueba /ayuda")

    return True


def _info_modelo() -> dict:
    """Lee la información del modelo RF persistido (sin cargarlo entero si falla)."""
    from src.orquestador import modelo_ruta_por_defecto
    ruta = modelo_ruta_por_defecto()
    info = {"ruta": ruta, "clases": [], "n_features": "—", "n_estimators": "—"}
    try:
        from src.modelo.rf import cargar
        modelo = cargar(ruta)
        le = modelo.get("le")
        info["clases"] = list(le.classes_) if le is not None else []
        info["n_features"] = len(modelo.get("caracteristicas", []))
        info["n_estimators"] = getattr(modelo.get("clf"), "n_estimators", "—")
    except Exception as e:
        info["clases"] = [f"(no se pudo cargar: {type(e).__name__})"]
    return info


# ----------------------------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------------------------

def ejecutar_chat(ruta_pdf: str | Path | None = None, sin_color: bool = False) -> int:
    """Bucle interactivo del chat. Devuelve el código de salida del proceso."""
    consola = crear_consola(sin_color=sin_color)
    sesion = SesionChat(ruta_pdf=ruta_pdf)

    mostrar_bienvenida(consola, _info_modelo())

    # Carga inicial del PDF si se indicó por argumento.
    if ruta_pdf is not None:
        try:
            resumen = sesion.cargar_pdf(ruta_pdf)
            consola.print(f"[magenta]PDF cargado:[/magenta] {resumen.nombre} "
                          f"({resumen.fragmentos} fragmentos)\n")
        except Exception as e:
            consola.print(f"[red]No se pudo cargar {ruta_pdf}:[/red] {e}\n")
            sesion.ruta_pdf = None

    sesion_prompt = _crear_sesion_prompt()

    try:
        while True:
            entrada = _leer_entrada(sesion_prompt, consola)
            if entrada is None:                 # EOF / Ctrl+D
                break
            linea = (entrada or "").strip()
            if not linea:
                continue

            if linea.startswith("/"):
                if not _procesar_comando(consola, sesion, linea):
                    break
                continue

            # Evaluación con indicador de progreso.
            with consola.status("[cyan]Evaluando…[/cyan]", spinner="dots"):
                resultado = sesion.evaluar(linea)

            mostrar_resultado(consola, resultado, titulo_url=linea if linea.lower().startswith(PREFIJOS_URL) else None)
            sesion.registrar(linea, resultado)

    except KeyboardInterrupt:
        consola.print("\n[dim]Sesión interrumpida.[/dim]")
    finally:
        sesion.cerrar()

    consola.print("[dim]Hasta luego.[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="verificador chat",
        description="Sesión interactiva para probar el modelo de veracidad.",
    )
    ap.add_argument("--pdf", default=None, help="PDF a cargar en la sesión (expansión dinámica).")
    ap.add_argument("--sin-color", action="store_true", help="Desactiva el color (útil en pipes).")
    args = ap.parse_args(argv)
    return ejecutar_chat(ruta_pdf=args.pdf, sin_color=args.sin_color)


if __name__ == "__main__":
    sys.exit(main())
