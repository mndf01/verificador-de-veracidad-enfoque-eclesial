import json
import os
import re

# =====================================================================
# 1. EL MAPA DE REGLAS (MOTOR DE CLASIFICACIÓN)
# =====================================================================

REGLAS_ETL = {
    "COMPENDIO": {
        "condicion_fuente": "Compendio de la Doctrina Social",
        "categoria_default": "Doctrina Social y Política",
        "autoridad_default": 0.66
    },
    "CANONICO": {
        "condicion_fuente": "Código de Derecho Canónico",
        "autoridad_default": 1.0,
        "mapeo_interno": {
            "LIBRO I": "Jerarquía y Gobierno Eclesiástico",
            "LIBRO II": "Derechos y Obligaciones de los Fieles (Laicado)",
            "LIBRO III": "Jerarquía y Gobierno Eclesiástico",
            "LIBRO IV": "Sacramentos y Liturgia",
            "LIBRO V": "Jerarquía y Gobierno Eclesiástico",
            "LIBRO VI": "Derecho Penal Eclesiástico",
            "LIBRO VII": "Derecho Penal Eclesiástico"
        }
    },
    "VATICANO_II": {
        "condicion_fuente": "Concilio Vaticano II", 
        "mapeo_documentos": {
            "Lumen gentium": {"categoria": "Jerarquía y Gobierno Eclesiástico", "autoridad": 1.0},
            "Dei Verbum": {"categoria": "Jerarquía y Gobierno Eclesiástico", "autoridad": 1.0},
            "Sacrosanctum concilium": {"categoria": "Sacramentos y Liturgia", "autoridad": 1.0},
            "Gaudium et spes": {"categoria": "Doctrina Social y Política", "autoridad": 0.66},
            "Unitatis redintegratio": {"categoria": "Relaciones Internacionales y Ecumenismo", "autoridad": 0.66},
            "Nostra aetate": {"categoria": "Relaciones Internacionales y Ecumenismo", "autoridad": 0.66},
            "Dignitatis humanae": {"categoria": "Doctrina Social y Política", "autoridad": 0.66},
            "Ad gentes": {"categoria": "Doctrina Social y Política", "autoridad": 0.66}
        },
        "categoria_default": "Derechos y Obligaciones de los Fieles (Laicado)",
        "autoridad_default": 0.66
    }
}

# =====================================================================
# 2. EL MOTOR DE INYECCIÓN DE METADATOS
# =====================================================================

def clasificar_archivo(archivo_entrada, archivo_salida):
    if not os.path.exists(archivo_entrada):
        print(f"[!] No se encontró el archivo: {archivo_entrada}")
        return

    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fuente_global = data.get("fuente", "")
    print(f"[*] Clasificando: {fuente_global}...")

    # Identificar qué conjunto de reglas usar
    reglas_activas = None
    tipo_regla = None
    for clave, reglas in REGLAS_ETL.items():
        if reglas["condicion_fuente"] in fuente_global:
            reglas_activas = reglas
            tipo_regla = clave
            break

    if not reglas_activas:
        print(f"[!] No hay reglas definidas para la fuente: {fuente_global}")
        return

    # Inyectar metadatos a cada fragmento
    for item in data["datos"]:
        contexto = item.get("contexto_jerarquico", "")
        
        categoria_asignada = reglas_activas.get("categoria_default", "General")
        autoridad_asignada = reglas_activas.get("autoridad_default", 0.33)

        if tipo_regla == "CANONICO":
            for libro, cat in reglas_activas["mapeo_interno"].items():
                if libro in contexto:
                    categoria_asignada = cat
                    break
        
        elif tipo_regla == "VATICANO_II":
            for doc, meta in reglas_activas["mapeo_documentos"].items():
                if re.search(doc, contexto, re.IGNORECASE):
                    categoria_asignada = meta["categoria"]
                    autoridad_asignada = meta["autoridad"]
                    break

        item["categoria"] = categoria_asignada
        item["nivel_autoridad"] = autoridad_asignada

    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"[+] Éxito. {len(data['datos'])} items clasificados guardados en: {archivo_salida}\n")

# =====================================================================
# 3. EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == '__main__':
    directorio_datos_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extractores", "data")
    directorio_datos_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
    
    # Mapeo de archivos de entrada -> archivos de salida clasificados
    archivos_a_procesar = [
        ("corpus_compendio_v1.json", "corpus_compendio_clasificado.json"),
        ("corpus_derecho_canonico_v1.json", "corpus_derecho_canonico_clasificado.json"),
        ("corpus_constituciones_v1.json", "corpus_constituciones_clasificada.json"),
        ("corpus_declaraciones_v1.json", "corpus_declaraciones_clasificada.json"),
        ("corpus_decretos_v1.json", "corpus_decretos_clasificado.json")
    ]
    
    print("=== INICIANDO MOTOR DE CLASIFICACIÓN Y METADATOS ===")
    for arch_in, arch_out in archivos_a_procesar:
        ruta_in = os.path.join(directorio_datos_in, arch_in)
        ruta_out = os.path.join(directorio_datos_out, arch_out)
        clasificar_archivo(ruta_in, ruta_out)
        
    print("=== PROCESO FINALIZADO ===")