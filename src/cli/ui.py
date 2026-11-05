"""Renderizado de la interfaz de terminal con Rich.

Separa PRESENTACIÓN de lógica: este módulo solo dibuja; no descarga, no consulta
el corpus ni predice. Así se puede testear capturando la salida a texto plano
(Rich permite `Console(file=StringIO())`) sin necesitar una terminal real.

Las funciones reciben el `ResultadoEvaluacion` del orquestador y lo convierten en
paneles, tablas y barras. No conocen nada del RAG ni del Random Forest.
"""
from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Color por clase de veredicto. El sistema mide SIMILITUD con el corpus, así que
# el color indica cercanía, no "verdad" ni "mentira".
ESTILO_VEREDICTO = {
    "proximo": ("green", "✓", "PRÓXIMO AL CORPUS"),
    "distante": ("yellow", "~", "DISTANTE DEL CORPUS"),
    "no_definido": ("blue", "?", "NO DEFINIDO"),
}

NIVEL_AUTORIDAD_LABEL = {
    1.0: "Nivel 3 · universal (CIC)",
    0.85: "Nivel 3 · Constitución (Vaticano II)",
    0.70: "Nivel 2 · Decreto/Declaración (Vaticano II)",
    0.55: "Nivel 2 · Compendio DSI",
    0.33: "Nivel 1 · normativa externa",
}


def crear_consola(sin_color: bool = False, archivo=None) -> Console:
    """Crea la consola. `archivo` permite capturar la salida en pruebas."""
    return Console(
        file=archivo,
        no_color=sin_color,
        force_terminal=None if archivo is None else False,
        width=None if archivo is None else 100,
        highlight=False,
    )


# ----------------------------------------------------------------------------------------
# Cabecera y ayuda
# ----------------------------------------------------------------------------------------

def mostrar_bienvenida(consola: Console, modelo_info: dict | None = None) -> None:
    """Cabecera de la sesión de chat con el estado del sistema."""
    lineas = [
        Text("Verificador de Veracidad — Enfoque Eclesial", style="bold"),
        Text(""),
        Text(
            "Contrasta noticias contra el corpus doctrinal de la Iglesia Católica.",
            style="dim",
        ),
        Text(
            "Mide SIMILITUD con el corpus, no verdad absoluta.",
            style="dim italic",
        ),
    ]
    if modelo_info:
        clases = ", ".join(modelo_info.get("clases", []))
        lineas.append(Text(""))
        lineas.append(Text(f"Modelo: {modelo_info.get('n_features', '?')} variables · clases: {clases}",
                           style="dim green"))

    consola.print(Panel(
        Group(*lineas),
        title="[bold]verificador[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    consola.print("[dim]Escribe una URL o una afirmación. /ayuda para los comandos.[/dim]\n")


def mostrar_ayuda(consola: Console) -> None:
    tabla = Table(box=None, show_header=False, padding=(0, 1))
    tabla.add_column("Comando", style="cyan", no_wrap=True)
    tabla.add_column("Qué hace", style="dim")

    filas = [
        ("https://...", "Evalúa la noticia de esa URL (descarga y analiza el cuerpo)"),
        ("texto libre", "Evalúa el texto como afirmación a contrastar"),
        ("/pdf RUTA", "Carga un PDF en la sesión (expansión dinámica)"),
        ("/pdf off", "Descarga el PDF de la sesión"),
        ("/modelo", "Información del modelo cargado"),
        ("/historial", "Evaluaciones previas de esta sesión"),
        ("/ayuda", "Esta ayuda"),
        ("/salir", "Termina la sesión (también Ctrl+D)"),
    ]
    for cmd, desc in filas:
        tabla.add_row(cmd, desc)

    consola.print(Panel(tabla, title="Comandos", border_style="cyan"))


# ----------------------------------------------------------------------------------------
# Resultado de una evaluación
# ----------------------------------------------------------------------------------------

def _barra(valor: float, ancho: int = 12) -> Text:
    """Barra de progreso textual para un valor 0..1.

    El número se muestra siempre (no se trunca), por eso la barra usa `ancho`
    fijo y el texto se añade después: en tablas estrechas se recorta la barra,
    nunca el valor.
    """
    valor = max(0.0, min(1.0, float(valor or 0.0)))
    llenos = int(round(valor * ancho))
    texto = Text(f"{valor:.3f} ", style="bold")
    texto.append("█" * llenos, style="cyan")
    texto.append("░" * (ancho - llenos), style="grey35")
    return texto


def _nivel(valor) -> str:
    try:
        return NIVEL_AUTORIDAD_LABEL.get(round(float(valor), 2), f"Nivel {valor}")
    except (TypeError, ValueError):
        return "—"


def mostrar_resultado(consola: Console, resultado, titulo_url: str | None = None) -> None:
    """Dibuja el informe completo de `evaluar_noticia()`."""
    if not resultado.ok:
        consola.print(Panel(
            "\n".join(resultado.errores) or "No se pudo completar la evaluación.",
            title="[red]Sin resultado[/red]",
            border_style="red",
        ))
        return

    clase = resultado.veredicto or "no_definido"
    color, icono, etiqueta = ESTILO_VEREDICTO.get(clase, ("white", "?", clase.upper()))

    # --- Panel de veredicto ---
    # El título del panel NO repite la clase (ya aparece en el cuerpo): se usa un
    # título neutro para evitar la lectura redundante "distante / DISTANTE DEL CORPUS".
    confianza = f"{resultado.probabilidad:.0%}" if resultado.probabilidad is not None else "—"
    cuerpo = Text()
    cuerpo.append(f"{icono} {etiqueta}", style=f"bold {color}")
    cuerpo.append(f"   ·   confianza en este veredicto: {confianza}\n\n", style="dim")
    cuerpo.append(_sentencia(clase), style="")

    consola.print(Panel(
        cuerpo,
        title="[bold]veredicto[/bold]",
        border_style=color,
        padding=(1, 2),
    ))

    # --- Datos de la noticia ---
    if titulo_url:
        consola.print(f"[dim]URL:[/dim] {titulo_url}")

    informe = resultado.informe_ivr or {}
    if resultado.titular_sugerido:
        consola.print(f"[dim]Titular:[/dim] {resultado.titular_sugerido}")
    if resultado.premisa:
        premisa = resultado.premisa if len(resultado.premisa) <= 120 else resultado.premisa[:117] + "..."
        consola.print(f"[dim]Premisa:[/dim] {premisa}")

    exp = informe.get("expansion_dinamica")
    if exp and exp.get("resumen") is not None:
        r_pdf = exp["resumen"]
        nombre = getattr(r_pdf, "nombre", None) or exp.get("ruta_pdf")
        frags = getattr(r_pdf, "fragmentos", "?")
        consola.print(f"[magenta]PDF en sesión:[/magenta] {nombre} ({frags} fragmentos)")

    if not informe or not informe.get("ok"):
        consola.print()
        return

    # Distribución de probabilidad: muestra cuándo el modelo dudaba entre clases
    # (caso habitual en la frontera distante <-> no_definido).
    probabilidades = resultado.probabilidades or {}
    if len(probabilidades) > 1:
        ordenadas = sorted(probabilidades.items(), key=lambda kv: kv[1], reverse=True)
        resumen = "   ".join(f"{c} {p:.0%}" for c, p in ordenadas)
        consola.print(f"[dim]Distribución: {resumen}[/dim]")
        consola.print()

    # --- Métricas agregadas ---
    consola.print()
    metricas = Table(box=None, show_header=False, padding=(0, 2))
    metricas.add_column(style="dim", no_wrap=True)
    metricas.add_column()
    metricas.add_row("Evidencias", str(informe.get("n_evidencias", 0)))
    metricas.add_row("IVR máx / medio / mín",
                     f"{informe.get('ivr_max', 0):.3f} / {informe.get('ivr_medio', 0):.3f} / {informe.get('ivr_min', 0):.3f}")
    conceptos = informe.get("conceptos_premisa") or {}
    metricas.add_row("Conceptos del tesauro",
                     ", ".join(conceptos) if conceptos else "[dim]ninguno[/dim]")
    consola.print(metricas)
    consola.print()

    # --- Tabla de evidencias ---
    evidencias = informe.get("evidencias") or []
    if evidencias:
        tabla = Table(
            title="[bold]Evidencias recuperadas[/bold]",
            title_justify="left",
            header_style="bold dim",
            border_style="grey35",
            show_lines=False,
            expand=True,
        )
        tabla.add_column("#", justify="right", width=2, style="dim")
        tabla.add_column("IVR", ratio=2, no_wrap=True)
        tabla.add_column("Cita", ratio=4, overflow="fold")
        tabla.add_column("Nivel de autoridad", ratio=3, style="dim", overflow="fold")
        tabla.add_column("Origen", width=10, style="dim")

        for i, ev in enumerate(evidencias[:8], 1):
            origen = "PDF" if ev.get("origen") == "expansion_dinamica" else "corpus"
            tabla.add_row(
                str(i),
                _barra(ev.get("ivr", 0.0)),
                ev.get("cita") or "—",
                _nivel(ev.get("nivel_autoridad")),
                origen,
            )
        consola.print(tabla)

        if len(evidencias) > 8:
            consola.print(f"[dim]… y {len(evidencias) - 8} evidencias más[/dim]")

    consola.print()


def _sentencia(clase: str) -> str:
    return {
        "proximo": "El contenido está próximo al magisterio y la normativa del corpus: "
                   "trata los mismos temas y en términos compatibles.",
        "distante": "La noticia es del ámbito eclesial, pero el corpus no la respalda: "
                    "no se encontró correspondencia suficiente.",
        "no_definido": "El corpus no cubre el tema de esta noticia: no hay base de "
                       "comparación para pronunciarse.",
    }.get(clase, "Veredicto no reconocido.")


# ----------------------------------------------------------------------------------------
# Historial de la sesión
# ----------------------------------------------------------------------------------------

def mostrar_historial(consola: Console, registros: list[dict]) -> None:
    """Tabla con las evaluaciones previas de la sesión."""
    if not registros:
        consola.print("[dim]Aún no hay evaluaciones en esta sesión.[/dim]")
        return

    tabla = Table(title="[bold]Historial de esta sesión[/bold]", border_style="grey35",
                  header_style="bold dim", expand=True)
    tabla.add_column("#", justify="right", width=3, style="dim")
    tabla.add_column("Consulta", ratio=5, overflow="fold")
    tabla.add_column("Veredicto", ratio=2, no_wrap=True)
    tabla.add_column("Conf.", justify="right", width=6)

    for i, r in enumerate(registros, 1):
        clase = r.get("veredicto") or "—"
        color = ESTILO_VEREDICTO.get(clase, ("white",))[0]
        conf = f"{r['probabilidad']:.0%}" if r.get("probabilidad") is not None else "—"
        tabla.add_row(str(i), r.get("consulta", "—"),
                      f"[{color}]{clase}[/{color}]", conf)
    consola.print(tabla)


def mostrar_info_modelo(consola: Console, info: dict) -> None:
    tabla = Table(box=None, show_header=False, padding=(0, 2))
    tabla.add_column(style="dim", no_wrap=True)
    tabla.add_column()
    tabla.add_row("Clases", ", ".join(info.get("clases", [])) or "—")
    tabla.add_row("Variables", str(info.get("n_features", "—")))
    tabla.add_row("Árboles", str(info.get("n_estimators", "—")))
    tabla.add_row("Ruta", str(info.get("ruta", "—")))
    consola.print(Panel(tabla, title="[bold]Modelo Random Forest[/bold]", border_style="cyan"))
