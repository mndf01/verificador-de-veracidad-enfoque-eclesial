import os
import sys

# Aseguramos que Python encuentre la raíz del proyecto para importar config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import obtener_configuraciones

# Importamos las funciones principales de tus scrapers
from extractores.scraper_canones_v1 import obtener_corpus_canonico
from extractores.scraper_compendio_v1 import obtener_corpus_compendio
from extractores.scraper_constituciones_v1 import obtener_corpus_constituciones
from extractores.scraper_declaraciones_v1 import obtener_corpus_declaraciones
from extractores.scraper_decretos_v1 import obtener_corpus_decretos

# Importamos el clasificador
from clasificador_corpus.clasificador_metadatos_v1 import clasificar_archivo

def ejecutar_pipeline():
    print("=====================================================")
    print("   INICIANDO PIPELINE ETL - CORPUS ECLESIÁSTICO      ")
    print("=====================================================\n")

    # 1. Cargar configuraciones del proyecto
    settings = obtener_configuraciones()
    modo = settings.corp_extractor.modo_ejecucion

    # 2. Definición exacta de tus rutas (Arquitectura Medallón)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ZONA BRONZE: Datos Crudos
    dir_crudos = os.path.join(base_dir, "extractores", "data")
    # ZONA GOLD: Datos Enriquecidos y listos para la IA
    dir_procesados = os.path.join(base_dir, "datos")

    # Aseguramos que existan ambas carpetas
    os.makedirs(dir_crudos, exist_ok=True)
    os.makedirs(dir_procesados, exist_ok=True)

    print(f"[*] Modo de ejecución: {modo.upper()}")
    print(f"[*] Carpeta RAW (Descargas): {dir_crudos}")
    print(f"[*] Carpeta GOLD (Finales): {dir_procesados}\n")

    # 3. FASE DE EXTRACCIÓN (Scrapers)
    print("--- FASE 1: EXTRACCIÓN DE DATOS CRUDOS ---")
    archivos_crudos = [
        "corpus_derecho_canonico_v1.json",
        "corpus_compendio_v1.json",
        "corpus_constituciones_v1.json",
        "corpus_declaraciones_v1.json",
        "corpus_decretos_v1.json"
    ]

    # Ejecutamos pasando la ruta de la Zona Bronze
    obtener_corpus_canonico(modo=modo, archivo_backup=archivos_crudos[0])
    obtener_corpus_compendio(modo=modo, archivo_backup=archivos_crudos[1])
    obtener_corpus_constituciones(modo=modo, archivo_backup=archivos_crudos[2])
    obtener_corpus_declaraciones(modo=modo, archivo_backup=archivos_crudos[3])
    obtener_corpus_decretos(modo=modo, archivo_backup=archivos_crudos[4])


    # 4. FASE DE TRANSFORMACIÓN (Clasificador de Metadatos)
    print("\n--- FASE 2: CLASIFICACIÓN E INYECCIÓN DE METADATOS ---")
    for archivo_crudo in archivos_crudos:
        # Lee desde la carpeta de datos crudos (extractores/data/)
        ruta_in = os.path.join(dir_crudos, archivo_crudo)
        
        # Genera el nombre final y guarda en la carpeta de listos (datos/)
        nombre_out = archivo_crudo.replace("_v1.json", "_clasificado.json")
        ruta_out = os.path.join(dir_procesados, nombre_out)
        
        if os.path.exists(ruta_in):
            clasificar_archivo(ruta_in, ruta_out)
        else:
            print(f"[!] Omitido: No se encontró {archivo_crudo} en {dir_crudos}")

    print("=====================================================")
    print("       PIPELINE FINALIZADO CON ÉXITO                 ")
    print("=====================================================")

if __name__ == '__main__':
    ejecutar_pipeline()