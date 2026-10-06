"""Prueba manual desde consola.

    python -m src.expansion_dinamica ruta/al/archivo.pdf --solo-segmentar
    python -m src.expansion_dinamica ruta/al/archivo.pdf --premisa "El obispo puede dispensar..."

--solo-segmentar NO carga el modelo de embeddings ni toca Chroma: sirve para revisar
cómo se está cortando un documento (esquema detectado, advertencias, fragmentos).
"""
from __future__ import annotations

import argparse
import sys

from .errores import ExpansionError
from .parametros import ParametrosExpansion
from .procesamiento import procesar_pdf


def _params() -> ParametrosExpansion:
    try:
        return ParametrosExpansion.desde_config()
    except ImportError:     # fuera del proyecto (sin config.py / pydantic): valores por defecto
        return ParametrosExpansion()
    except Exception as e:  # config.toml mal formado, falta una clave...: que no pase desapercibido
        print(f"⚠ No se pudo leer config.toml ({type(e).__name__}: {e}); se usan los valores por defecto.",
              file=sys.stderr)
        return ParametrosExpansion()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probar el módulo de expansión dinámica con un PDF.")
    ap.add_argument("pdf")
    ap.add_argument("--solo-segmentar", action="store_true", help="No carga embeddings ni Chroma.")
    ap.add_argument("--nivel", type=float, default=None, help="Nivel de autoridad: 0.33, 0.66 o 1.0")
    ap.add_argument("--premisa", help="Consulta de prueba contra el documento cargado.")
    ap.add_argument("--mostrar", type=int, default=5, help="Cuántos fragmentos/resultados imprimir.")
    args = ap.parse_args(argv)

    params = _params()
    try:
        if args.solo_segmentar or not args.premisa:
            doc = procesar_pdf(args.pdf, params=params)
            seg = doc.segmentacion
            print(f"Documento : {doc.nombre}  ({doc.total_paginas} págs.)")
            print(f"Esquema   : {seg.esquema}")
            print(f"Fragmentos: {len(seg.fragmentos)}")
            for a in doc.advertencias:
                print(f"  ⚠ {a}")
            print()
            for f in seg.fragmentos[: args.mostrar]:
                ctx = " > ".join(f.contexto) or "(sin contexto)"
                print(f"[{f.etiqueta}] p.{f.pagina_inicio}-{f.pagina_fin} | {ctx}")
                print(f"   {f.texto[:200]}{'…' if len(f.texto) > 200 else ''}\n")
            return 0

        from .sesion import SesionExpansion

        with SesionExpansion(params=params) as sesion:
            r = sesion.cargar_pdf(args.pdf, nivel_autoridad=args.nivel)
            print(f"Cargado: {r.nombre} → {r.fragmentos} fragmentos (esquema {r.esquema}, nivel {r.nivel_autoridad})")
            for a in r.advertencias:
                print(f"  ⚠ {a}")
            print()
            for res in sesion.consultar(args.premisa, n_resultados=args.mostrar):
                print(f"{res['similitud']:.3f}  {res['cita']}")
                print(f"       {res['texto'][:200]}{'…' if len(res['texto']) > 200 else ''}\n")
        return 0

    except ExpansionError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
