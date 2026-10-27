"""Interfaz de terminal del verificador (lógica de comandos).

Maneja los subcomandos y coordina la I/O (input/print). Delega la lógica de
negocio en `src.orquestador`. Mantiene el patrón del repo: argparse, módulos
ejecutables con `python -m`, mensajes claros y códigos de salida 0/1.

Expuesto como:
    python -m src.cli ...          (via __main__.py)
    verificador ...                (via entry point pyproject, src.cli:main)
    python -m verificador ...      (via verificador.py raíz)
"""
from __future__ import annotations

import argparse
import sys

# Ancla la raíz del proyecto en el path para que `import config`/`src.*` funcionen
# aunque se lance el CLI desde otro directorio (igual que main_etl.py y ivr.py).
from pathlib import Path

_PROYECTO_RAIZ = Path(__file__).resolve().parents[2]
if _PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, str(_PROYECTO_RAIZ))


def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="verificador",
        description="Verificador de veracidad con enfoque eclesial: contrasta noticias "
                    "frente a un corpus canónico mediante RAG + IVR + Random Forest.",
    )
    sub = ap.add_subparsers(dest="comando", required=True)

    # ---- evaluar ----
    p_eval = sub.add_parser("evaluar", help="Evalúa una noticia por URL hasta el veredicto.")
    p_eval.add_argument("url", help="URL absoluta de la noticia a evaluar.")
    p_eval.add_argument("--pdf", default=None, help="PDF externo para expandir dinámicamente la consulta.")
    p_eval.add_argument("--premisa", default=None, help="Premisa explícita (si no, se usa el titular).")
    p_eval.add_argument("--no-interactivo", action="store_true",
                        help="No pregunta por pantalla: usa la premisa dada o el titular tal cual.")
    p_eval.add_argument("--n-resultados", type=int, default=5, help="Evidencias a recuperar.")
    p_eval.set_defaults(func=_cmd_evaluar)

    # ---- setup ----
    p_setup = sub.add_parser("setup", help="ETL completo + indexa el corpus en Chroma.")
    p_setup.add_argument("--no-indexar", action="store_true",
                         help="Solo ejecuta el ETL, sin indexar en Chroma.")
    p_setup.set_defaults(func=_cmd_setup)

    # ---- dataset-generar ----
    p_ds = sub.add_parser("dataset-generar", help="Genera un CSV de entrenamiento para el RF.")
    p_ds.add_argument("--salida", default="data/etiquetado.csv", help="CSV de salida.")
    p_ds.add_argument("--premisas", help="Archivo de texto (una premisa por línea) con verdades/mentiras.")
    p_ds.add_argument("--fuente-pdf", help="PDF a usar como fuente de los ejemplos (bajo desarrollo).")
    p_ds.set_defaults(func=_cmd_dataset)

    # ---- entrenar ----
    p_tr = sub.add_parser("entrenar", help="Entrena y persiste el Random Forest.")
    p_tr.add_argument("--dataset", default="data/etiquetado.csv", help="CSV etiquetado (características + veredicto).")
    p_tr.add_argument("--salida", default="data/modelo_rf.joblib", help="Dónde guardar el modelo.")
    p_tr.set_defaults(func=_cmd_entrenar)

    return ap


# ----------------------------------------------------------------------------------------
# Implementación de cada subcomando
# ----------------------------------------------------------------------------------------

def _cmd_evaluar(args: argparse.Namespace) -> int:
    from src.orquestador import ResultadoEvaluacion, evaluar_noticia, veredicto_para_interfaz

    # 0. Resolución de la premisa en modo interactivo: proponemos el titular y
    #    dejamos que el usuario lo confirme o lo reescriba. En modo no-interactivo,
    #    si no hay --premisa, evaluar_noticia derivará el titular solo.
    premisa = args.premisa
    if premisa is None and not args.no_interactivo:
        from src.web_extractor import extraer_titular
        try:
            titular = extraer_titular(args.url)
        except Exception as e:
            titular = None
            print(f"⚠ No se pudo extraer el titular automáticamente ({type(e).__name__}).", file=sys.stderr)
        if titular:
            print(f"\nTitular detectado: {titular}")
            respuesta = input(
                "¿Usarlo como premisa? [Enter=usar, 'n'=escribir otra]: ").strip().lower()
            if respuesta in ("n", "no"):
                premisa = input("Escribe la premisa: ").strip()
            else:
                premisa = titular
        else:
            premisa = input("No se detectó titular; escribe la premisa: ").strip()
        if not premisa:
            print("✖ No se proporcionó una premisa.", file=sys.stderr)
            return 1

    print("\n⏳ Evaluando la noticia...")
    resultado: ResultadoEvaluacion = evaluar_noticia(
        args.url,
        premisa=premisa,
        ruta_pdf=args.pdf,
        n_resultados=args.n_resultados,
    )

    if not resultado.ok:
        print("\n✖ No se pudo completar la evaluación.")
        for err in resultado.errores or ["motivo desconocido"]:
            print(f"  • {err}")
        _sugerir_si_vacio()
        return 1

    _mostrar_informe(resultado)
    return 0


def _mostrar_informe(resultado) -> None:
    from src.orquestador import _nivel_label, veredicto_para_interfaz

    informe = resultado.informe_ivr or {}
    print("\n" + "=" * 62)
    print("   INFORME DE VERACIDAD")
    print("=" * 62)

    print(f"\nURL      : {resultado.url}")
    print(f"Premisa  : {resultado.premisa}")

    # Veredicto en grande
    print(f"\n{'─' * 62}")
    print(veredicto_para_interfaz(resultado))
    if resultado.probabilidad is not None:
        print(f"Confianza del modelo: {resultado.probabilidad:.2%}")
    print(f"{'─' * 62}")

    if informe:
        print(f"\nEvidencias recuperadas : {informe.get('n_evidencias', 0)}")
        print(f"IVR máximo / medio / min: "
              f"{informe.get('ivr_max', 0.0):.3f} / {informe.get('ivr_medio', 0.0):.3f} / {informe.get('ivr_min', 0.0):.3f}")
        conceptos = informe.get("conceptos_premisa") or {}
        if conceptos:
            print(f"Conceptos del tesauro detectados: {', '.join(conceptos)}")
        else:
            print("Conceptos del tesauro detectados: ninguno")

        print("\nEvidencias (top):")
        for i, ev in enumerate(informe.get("evidencias", [])[:5], 1):
            sen = ev.get("senales", {})
            so = "n/a" if sen.get("ontologico") is None else f"{sen['ontologico']:.3f}"
            nivel = _nivel_label(ev.get("nivel_autoridad"))
            print(f"\n  {i}. IVR={ev.get('ivr', 0.0):.3f} | SB={sen.get('sb', 0.0):.3f} "
                  f"| L={sen.get('literal', 0.0):.3f} | SO={so} | J={sen.get('autoridad', 0.0):.3f}")
            print(f"     Cita : {ev.get('cita', 'sin cita')}")
            print(f"     Nivel: {nivel}")
            texto = (ev.get("texto") or "")[:220]
            print(f"     Texto: {texto}{'…' if len(ev.get('texto', '')) > 220 else ''}")

    print("\n" + "=" * 62)


def _cmd_setup(args: argparse.Namespace) -> int:
    """ETL completo + indexado del corpus en ChromaDB.

    Fase 1) ETL: ejecuta `src.corp_extractor.main_etl` (extrae y clasifica el
            corpus en la zona GOLD, *_clasificado.json). Se lanza como módulo
            porque main_etl usa imports relativos del paquete; así respeta sus
            propias rutas y su modo (estatico/dinamico) leído de config.toml.
    Fase 2) Indexado: carga esos JSON en la colección `corpus_canonico` con
            `chromaDB.reindexar_corpus()` y verifica la sincronización.
    """
    from config import obtener_configuraciones
    settings = obtener_configuraciones()
    print("\n=== SETUP: preparación del corpus ===")
    print(f"▶ Modo (config.toml → [corp_extractor].modo_ejecucion): "
          f"{settings.corp_extractor.modo_ejecucion}")

    # ---- Fase 1: ETL ----
    print("\n--- Fase 1: ETL (extracción y clasificación) ---")
    codigo = _ejecutar_etl()
    if codigo != 0:
        print(f"✖ El ETL terminó con código {codigo}. Revisa los mensajes anteriores.",
              file=sys.stderr)
        return 1

    # ---- Fase 2: indexado ----
    if args.no_indexar:
        print("\n⏭ Se omite el indexado (--no-indexar).")
        return 0

    print("\n--- Fase 2: indexado en ChromaDB ---")
    from src.base_datos.chromaDB import reindexar_corpus, verificar_corpus_indexado
    errores = reindexar_corpus()
    if errores:
        print(f"⚠ Se indexó con {len(errores)} archivo(s) fallido(s): {errores}",
              file=sys.stderr)

    diag = verificar_corpus_indexado()
    if diag:
        print("\n=== Verificación final ===")
        print(f"documentos en BD      : {diag.get('total_bd', 0)}")
        print(f"documentos esperados  : {diag.get('total_esperado', 0)}")
        print(f"¿sincronizado?        : {'sí' if diag.get('ok') else 'no'}")
        for tipo, info in (diag.get("por_tipo") or {}).items():
            print(f"  · {tipo}: esperado {info['esperado']} | en BD {info['en_bd']} | ok {info['ok']}")
    return 0


def _ejecutar_etl() -> int:
    """Corre el pipeline ETL como subproceso y devuelve su código de salida."""
    import subprocess
    python = sys.executable
    proc = subprocess.run(
        [python, "-m", "src.corp_extractor.main_etl"],
        cwd=str(_PROYECTO_RAIZ),
    )
    return proc.returncode


def _cmd_dataset(args: argparse.Namespace) -> int:
    # Fase 2: genera el CSV a partir de un conjunto de premisas etiquetadas.
    if args.premisas:
        print(f"\nℹ dataset-generar requiere la colección indexada y una fuente de premisas etiquetadas.")
        print(f"   Recibido: premisas={args.premisas}  salida={args.salida}")
    else:
        print("Dataset de entrenamiento aún no disponible (requiere colección indexada "
              "y premisas etiquetadas). Se implementará en la fase 2.")
    return 1


def _cmd_entrenar(args: argparse.Namespace) -> int:
    import os
    if not os.path.exists(args.dataset):
        print(f"\n✖ No existe el dataset: {args.dataset}", file=sys.stderr)
        print("   Genera uno con `verificador dataset-generar` o crea el CSV manualmente.", file=sys.stderr)
        return 1
    try:
        import pandas as pd
        from src.modelo.rf import entrena, guardar
    except Exception as e:
        print(f"✖ No se pudo cargar el módulo de entrenamiento ({type(e).__name__}: {e}).", file=sys.stderr)
        return 1
    df = pd.read_csv(args.dataset)
    clf, le = entrena(df)
    guardar(clf, le, args.salida)
    print(f"\n✓ Modelo Random Forest entrenado y guardado en {args.salida}")
    return 0


def _sugerir_si_vacio() -> None:
    """Si el fallo sugiere colección vacía o sin modelo, orienta al usuario."""
    from src.orquestador import modelo_ruta_por_defecto
    if not Path(modelo_ruta_por_defecto()).exists():
        print("\n💡 Sugerencia: aún no hay modelo Random Forest entrenado.")
        print("   Ejecuta `verificador entrenar` tras generar el dataset.")
    from src.base_datos.chromaDB import verificar_corpus_indexado
    try:
        diag = verificar_corpus_indexado()
        if diag and diag.get("total_bd", 0) == 0:
            print("💡 Sugerencia: la colección está vacía. Ejecuta `verificador setup` para indexar el corpus.")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())