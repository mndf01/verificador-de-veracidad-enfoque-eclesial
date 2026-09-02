import requests
from config import obtener_configuraciones, resolver_ruta
from bs4 import BeautifulSoup
import re
import json
import time
import os
from datetime import datetime

BASE_URL = "https://www.vatican.va/archive/cod-iuris-canonici/"
INDEX_URL = "https://www.vatican.va/archive/cod-iuris-canonici/cic_index_sp.html"

def obtener_texto_nodo(nodo_li):
    """Extrae el texto del nodo <li> aislando solo el título de su nivel."""
    texto = []
    for child in nodo_li.children:
        if child.name in ['ul', 'ol']:  # Ignoramos sub-listas estándar
            continue
        if isinstance(child, str):
            texto.append(child)
        else:
            texto.append(child.get_text(separator=' '))
            
    limpio = " ".join(texto)
    limpio = re.sub(r'\s+', ' ', limpio).strip()
    
    # --- CIRUGÍA NLP PARA LA TAXONOMÍA ---
    match = re.search(r'^(.*?\(\s*(?:Cann?\.?)?[^)]*\d+[^)]*\))', limpio, re.IGNORECASE)
    
    if match:
        limpio = match.group(1).strip()
    else:
        # Fallback de seguridad: si no hay paréntesis, cortamos si empieza un nuevo sub-nivel
        limpio = re.split(r'\s+(?=TÍTULO\b|CAPÍTULO\b|PARTE\b|Art\.\b)', limpio)[0]
        
    return limpio.strip()

def obtener_mapa_jerarquico(url_indice):
    """Navega el árbol DOM y crea un diccionario: {url: 'LIBRO > TÍTULO > CAPÍTULO'}"""
    print(f"[*] Analizando el árbol DOM para Extracción Dinámica de Taxonomías...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url_indice, headers=headers)
    response.encoding = 'utf-8'
    
    mapa_urls = {}
    if response.status_code != 200:
        print("[!] Error al acceder al índice.")
        return mapa_urls
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for li in soup.find_all('li'):
        a_tag = li.find('a')
        if a_tag and a_tag.get('href') and a_tag.get('href').startswith('esp/documents/cic_'):
            href = a_tag.get('href')
            url_completa = BASE_URL + href
            
            ruta_partes = []
            ancestros = li.find_parents('li')
            for anc in reversed(ancestros):
                anc_text = obtener_texto_nodo(anc)
                if anc_text:
                    ruta_partes.append(anc_text)
            
            nodo_text = obtener_texto_nodo(li)
            if nodo_text:
                ruta_partes.append(nodo_text)
                
            contexto_final = " > ".join(ruta_partes)
            mapa_urls[url_completa] = contexto_final
            
    return mapa_urls

def limpiar_ruido_nlp(texto):
    """Limpia la basura semántica y amputa el historial derogado sin destruir referencias cruzadas legales."""
    # --- CORTE DEL HISTORIAL DEROGADO ---
    texto = re.split(r'\(\s*n\s*Indica que el texto', texto, flags=re.IGNORECASE)[0]
    texto = re.split(r'\[\s*Redacción original', texto, flags=re.IGNORECASE)[0]
    
    # Eliminamos símbolos de párrafo de inicio y "n" minúscula de nueva versión
    t = re.sub(r'§\s*\d+\.\s*', '', texto)
    t = re.sub(r'^n\s+', '', t)
    
    return t.strip()

def extraer_canones(url, contexto_jerarquico):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    
    # --- FAIL-FAST: Si el servidor falla, rompemos el flujo inmediatamente ---
    if response.status_code != 200:
        raise ConnectionError(f"Fallo de conexión HTTP {response.status_code} al intentar acceder a: {url}")

    html_crudo = response.text
    
    # --- LIMPIEZA PROFUNDA ---
    html_crudo = html_crudo.replace('&nbsp;', ' ')
    html_crudo = re.sub(r'[\r\n\t]+', ' ', html_crudo)
    html_crudo = re.sub(r'\s+', ' ', html_crudo)
    html_crudo = re.sub(r'</?(?:a|font|span|i|em|u)[^>]*>', '', html_crudo)
    html_crudo = html_crudo.replace('[*]', '')
    
    # Limpieza de Notas Modificatorias Papales
    html_crudo = re.sub(r'\s*n\s*\(\s*(Cf\.[^)]+)\)', r' [Nota de actualización: \1]', html_crudo)
    
    # Inyectamos el <b> faltante por error humano del Vaticano en los rebeldes
    rebeldes = [196, 642, 838, 902, 994, 1097, 1102, 1201, 1261, 1460, 1485, 1559, 1675, 1679, 1681]
    for r in rebeldes:
        html_crudo = re.sub(rf'<p>\s*{r}\s+', f'<p><b>{r}</b> ', html_crudo)
    
    # --- EXTRACCIÓN ---
    patron_principal = r'<(?:b|strong)[^>]*>\s*(\d+)\s*</(?:b|strong)>\s*(?:-?\s*)?(\d*)(.*?)(?=<(?:b|strong)[^>]*>\s*\d+\s*</(?:b|strong)>|<hr>|<!--|$)'
    coincidencias = re.findall(patron_principal, html_crudo, re.DOTALL | re.IGNORECASE)
    
    canones = []
    if coincidencias:
        for num_dentro, num_fuera, contenido in coincidencias:
            numero_real = int(num_dentro.strip() + num_fuera.strip())
            texto_limpio = BeautifulSoup(contenido, 'html.parser').get_text(separator=' ', strip=True)
            
            # --- LA BOMBA NUCLEAR PARA EL 1482 ---
            if numero_real == 1481 and "1482" in texto_limpio:
                partes = re.split(r'1482\s*§\s*1\.', texto_limpio)
                if len(partes) == 2:
                    canones.append({
                        "canon_id": 1481,
                        "contexto_jerarquico": contexto_jerarquico,
                        "texto": limpiar_ruido_nlp(partes[0]), 
                        "url_origen": url
                    })
                    canones.append({
                        "canon_id": 1482,
                        "contexto_jerarquico": contexto_jerarquico,
                        "texto": limpiar_ruido_nlp(partes[1]), 
                        "url_origen": url
                    })
                    continue 
            
            canones.append({
                "canon_id": numero_real,
                "contexto_jerarquico": contexto_jerarquico,
                "texto": limpiar_ruido_nlp(texto_limpio), 
                "url_origen": url
            })
    else:
        # FALLBACK PARA EL LIBRO VI
        patron_secundario = r'Can\.\s*(\d+)(.*?)(?=Can\.\s*\d+|<hr>|<!--|$)'
        coincidencias_sec = re.findall(patron_secundario, html_crudo, re.DOTALL | re.IGNORECASE)
        for numero, contenido in coincidencias_sec:
            texto_limpio = BeautifulSoup(contenido, 'html.parser').get_text(separator=' ', strip=True)
            texto_limpio = re.sub(r'^-\s*', '', texto_limpio)
            canones.append({
                "canon_id": int(numero),
                "contexto_jerarquico": contexto_jerarquico,
                "texto": limpiar_ruido_nlp(texto_limpio), 
                "url_origen": url
            })
            
    return canones

def ejecutar_scraping():
    """Ejecuta toda la lógica de ingesta web y retorna el corpus consolidado"""
    mapa_jerarquico = obtener_mapa_jerarquico(INDEX_URL)
    urls_canones = list(mapa_jerarquico.keys())
    print(f"[*] Se construyó la taxonomía para {len(urls_canones)} enlaces.\n")
    
    corpus_canonico = []
    
    for i, url in enumerate(urls_canones):
        contexto = mapa_jerarquico[url]
        print(f"[{i+1}/{len(urls_canones)}] Extrayendo de: {url}")
        canones_pagina = extraer_canones(url, contexto)
        corpus_canonico.extend(canones_pagina)
        time.sleep(0.5) 
        
    corpus_unico = {c['canon_id']: c for c in corpus_canonico}
    corpus_final = sorted(corpus_unico.values(), key=lambda x: x['canon_id'])
    
    # Validación de integridad para el Try-Catch
    if len(corpus_final) < 1752:
        raise ValueError(f"Faltan cánones. Solo se extrajeron {len(corpus_final)} de 1752.")
        
    return corpus_final

def obtener_corpus_canonico(modo="estatico", archivo_backup="corpus_derecho_canonico_v1.json", max_intentos=3):
    """
    Gestor principal de datos con interruptor de entorno.
    modo='estatico': Carga la verdad absoluta desde el disco local.
    modo='dinamico': Fuerza la extracción web ignorando el disco para crear una nueva versión.
    """
    # =====================================================================
    # NUEVA GESTIÓN DE RUTAS (Ruta Absoluta Dinámica)
    # Esto asegura que sin importar desde dónde ejecutes el script, 
    # siempre apuntará a la carpeta 'datos' en la raíz del proyecto.
    # =====================================================================
    settings = obtener_configuraciones()
    # Zona Bronze: crudos (_v1.json) guardados en la carpeta configurada
    carpeta_salida = resolver_ruta(settings.paths.raw_corpus_dir)
    ruta_completa = os.path.join(carpeta_salida, archivo_backup)

    if modo == "estatico":
        print(f"[*] MODO ESTÁTICO ACTIVO: Cargando la verdad absoluta desde '{ruta_completa}'...")
        if os.path.exists(ruta_completa):
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"[!] ERROR FATAL: No se encontró el archivo en '{ruta_completa}'. Ejecuta el script en modo 'dinamico' primero.")
            return None

    elif modo == "dinamico":
        print(f"\n[*] MODO DINÁMICO ACTIVO: Ignorando archivos locales. Iniciando extracción web directa desde el Vaticano...")
        
        # --- LÓGICA DE REINTENTOS (TRY-CATCH) ---
        for intento in range(1, max_intentos + 1):
            print(f"--- Iniciando intento de conexión {intento}/{max_intentos} ---")
            try:
                corpus_fresco = ejecutar_scraping()
                
                # Aseguramos que la carpeta de salida exista antes de intentar guardar
                os.makedirs(carpeta_salida, exist_ok=True) 

                # --- EMPAQUETADO DEL DOCUMENTO MAESTRO ---
                documento_maestro = {
                    "nivel_jerarquico": 2,
                    "tipo_corpus": "corpus_canones",
                    "fuente": "Código de Derecho Canónico",
                    "fecha_generacion": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                    "total_canones": len(corpus_fresco),
                    "datos": corpus_fresco
                }
                
                # Guardamos el nuevo backup dentro de la carpeta
                with open(ruta_completa, "w", encoding="utf-8") as f:
                    json.dump(documento_maestro, f, ensure_ascii=False, indent=4)
                    
                print(f"\n[+] ¡ÉXITO! Nueva versión guardada en la carpeta: '{ruta_completa}'.")
                return documento_maestro

            except Exception as e:
                print(f"\n[!] Error crítico durante el intento {intento}: {str(e)}")
                if intento < max_intentos:
                    print("[*] Esperando 5 segundos antes de reintentar para no saturar el servidor...")
                    time.sleep(5)
                else:
                    print("\n[!] TODOS LOS INTENTOS DE EXTRACCIÓN HAN FALLADO.")

        # --- FALLBACK DE EMERGENCIA ---
        if os.path.exists(ruta_completa):
            print(f"[*] ATENCIÓN: El scraping dinámico falló por completo, pero se utilizará '{ruta_completa}' como medida de contingencia.")
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("[!] ERROR FATAL: No hay conexión a internet y no existe ningún backup local para iniciar el sistema.")
            return None
            
    else:
        print("[!] ERROR: Modo no reconocido. Utiliza 'estatico' o 'dinamico'.")
        return None

if __name__ == '__main__':
    # =====================================================================
    # INTERRUPTOR MAESTRO DEL DATA ENGINEER
    # Cambia a "dinamico" solo cuando necesites actualizar la base de datos
    # Cambia a "estatico" para cargar la información congelada al instante
    # =====================================================================
    
    MODO_EJECUCION = "dinamico" 
    ARCHIVO_SALIDA = "corpus_derecho_canonico_v1.json"
    
    corpus_maestro = obtener_corpus_canonico(modo=MODO_EJECUCION, archivo_backup=ARCHIVO_SALIDA)
    
    if corpus_maestro:
        # Validación compatible tanto con el nuevo diccionario maestro como con las listas antiguas
        total = corpus_maestro.get("total_canones", len(corpus_maestro)) if isinstance(corpus_maestro, dict) else len(corpus_maestro)
        print(f"\n[+] Motor de Datos inicializado: {total} cánones listos para procesar.")