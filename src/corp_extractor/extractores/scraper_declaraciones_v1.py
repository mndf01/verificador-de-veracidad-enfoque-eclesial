import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
from datetime import datetime

# --- URLs BASE ---
BASE_URL_VAT2 = "https://www.vatican.va/archive/hist_councils/ii_vatican_council/"
INDEX_URL = "https://www.vatican.va/archive/hist_councils/ii_vatican_council/index_sp.htm"

def obtener_enlaces_declaraciones(url_indice):
    """Navega el índice y extrae dinámicamente las URLs de las Declaraciones sin duplicados."""
    print(f"[*] Analizando el árbol DOM del índice para mapear Declaraciones...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url_indice, headers=headers)
    response.encoding = 'utf-8'
    
    if response.status_code != 200:
        raise ConnectionError("No se pudo acceder al índice principal.")

    soup = BeautifulSoup(response.text, 'html.parser')
    enlaces = soup.find_all('a', href=True)
    
    documentos_objetivo = []
    urls_vistas = set() 
    
    for a in enlaces:
        href = a['href']
        # Identificamos Declaraciones ('_decl_') en español ('_sp.html')
        if '_decl_' in href and href.endswith('_sp.html'):
            url_completa = BASE_URL_VAT2 + href
            if url_completa not in urls_vistas:
                titulo = a.get_text(strip=True)
                titulo = re.sub(r'\s+', ' ', titulo).strip()
                documentos_objetivo.append({'titulo': titulo, 'url': url_completa})
                urls_vistas.add(url_completa)
            
    print(f"[*] Se detectaron {len(documentos_objetivo)} Declaraciones únicas para la ingesta.\n")
    return documentos_objetivo

def extraer_numerales(url, titulo_documento):
    """Extrae los numerales y apéndices procesando el texto línea por línea."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    
    if response.status_code != 200:
        raise ConnectionError(f"Fallo HTTP {response.status_code} en: {url}")

    html_crudo = response.text 
    
    # Transformamos etiquetas de bloque en verdaderos saltos de línea físicos
    html_principal = re.sub(r'</(p|div|h[1-6]|li)>', '\n\n', html_crudo, flags=re.IGNORECASE)
    html_principal = re.sub(r'<br\s*/?>', '\n', html_principal, flags=re.IGNORECASE)
    
    soup = BeautifulSoup(html_principal, 'html.parser')
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    texto_plano = soup.get_text(separator=' ')
    
    # 1. LIMPIEZA PREVIA AL SPLIT
    texto_plano = re.sub(r'\[\s*\d+\s*\]', '', texto_plano)
    texto_plano = re.sub(r'\[\s*(?:AR|DE|EN|ES|FR|IT)\s*-.*?\]', '', texto_plano, flags=re.IGNORECASE | re.DOTALL)
    
    numerales = []
    current_id = None
    current_text = []
    
    contexto_base = f"Concilio Vaticano II > Declaraciones > {titulo_documento}"
    contexto_actual = contexto_base
    en_seccion_especial = False # Bandera para forzar ID 0
    
    for linea in texto_plano.split('\n'):
        linea = linea.strip()
        linea = re.sub(r'\s+', ' ', linea)
        
        if not linea:
            continue
            
        linea_upper = linea.upper()
        
        # =========================================================
        # 2. CORTAFUEGOS DEFINITIVO 
        # =========================================================
        if linea_upper == 'NOTAS' or linea_upper.startswith('NOTAS '):
            break
            
        # =========================================================
        # 3. FILTROS DE RUIDO Y METADATOS 
        # =========================================================
        if re.search(r'^(ROMA, EN SAN PEDRO|YO, PABLO|\* CONSTITUCIÓN PROMULGADA|\* DECLARACIÓN PROMULGADA|PERICLES FELICI|ARZOBISPO TITULAR|SECRETARIO GENERAL|PABLO OBISPO)', linea_upper):
            continue
        
        if re.search(r'^(TODAS Y CADA UNA DE LAS COSAS)', linea_upper):
            continue

        if re.match(r'^(CAP[ÍI]TULO|SECCI[ÓO]N)\b', linea_upper):
            continue
            
        # =========================================================
        # 4. DETECCIÓN DE SECCIONES ANEXAS E INTRODUCTORIAS
        # =========================================================
        match_seccion = re.match(r'^(A P É N D I C E|APÉNDICE|DE LAS ACTAS|NOTIFICACIONES|NOTA EXPLICATIVA|CONCLUSI[ÓO]N|INTRODUCCI[ÓO]N|PROEMIO)\b', linea_upper)
        if match_seccion:
            # Si traíamos texto arrastrado (ej. el último numeral antes de la conclusión), lo guardamos.
            if current_text:
                texto_limpio = " ".join(current_text).strip().replace('"', "'")
                if texto_limpio:
                    numerales.append({
                        "numeral_id": current_id if current_id is not None else 0,
                        "contexto_jerarquico": contexto_actual,
                        "texto": texto_limpio,
                        "url_origen": url
                    })
            
            # Cambiamos el contexto y preparamos variables para la nueva sección
            nombre_seccion = match_seccion.group(1).replace('A P É N D I C E', 'APÉNDICE')
            contexto_actual = f"{contexto_base} > {nombre_seccion}"
            
            # Forzamos ID 0 para estas secciones sin número
            current_id = 0
            current_text = []
            en_seccion_especial = True
            continue

        # =========================================================
        # 5. DETECCIÓN DE NUMERALES
        # =========================================================
        match_numeral = re.match(r'^(\d+)(?:\.|\.\s*[ªºaA]|[ªºaA])?\s+(.*)', linea)
        
        if match_numeral:
            if current_text:
                texto_limpio = " ".join(current_text).strip().replace('"', "'")
                if texto_limpio:
                    numerales.append({
                        "numeral_id": current_id if current_id is not None else 0,
                        "contexto_jerarquico": contexto_actual,
                        "texto": texto_limpio,
                        "url_origen": url
                    })
            
            # Al detectar un numeral, volvemos a la normalidad
            current_id = int(match_numeral.group(1))
            texto_restante = match_numeral.group(2).strip()
            current_text = [texto_restante] if texto_restante else []
            en_seccion_especial = False 
            
            # Si el numeral pertenece al documento principal, reseteamos el contexto
            if contexto_actual != contexto_base and not ("APÉNDICE" in contexto_actual or "NOTA EXPLICATIVA" in contexto_actual):
               contexto_actual = contexto_base

            continue

        # =========================================================
        # 6. FILTRO DE SUBTÍTULOS INTRUSOS
        # =========================================================
        if len(linea) < 150 and not re.search(r'[.:;!?\)\]"”\']$', linea):
            continue
            
        # =========================================================
        # 7. ASIGNACIÓN DEL TEXTO
        # =========================================================
        if en_seccion_especial:
            current_text.append(linea) # Todo va al ID 0 actual
        elif current_id is not None:
            current_text.append(linea)
        else:
            current_id = 0
            current_text.append(linea)
            
    # Guardar el último bloque en memoria (ej. la Conclusión)
    if current_text:
        texto_limpio = " ".join(current_text).strip().replace('"', "'")
        if texto_limpio:
            numerales.append({
                "numeral_id": current_id if current_id is not None else 0,
                "contexto_jerarquico": contexto_actual,
                "texto": texto_limpio,
                "url_origen": url
            })
            
    return numerales

def ejecutar_scraping_declaraciones():
    """Ejecuta toda la lógica de ingesta web dinámica para las Declaraciones."""
    documentos = obtener_enlaces_declaraciones(INDEX_URL)
    corpus_declaraciones = []
    
    for i, doc in enumerate(documentos):
        print(f"[{i+1}/{len(documentos)}] Extrayendo: {doc['titulo']}")
        numerales_doc = extraer_numerales(doc['url'], doc['titulo'])
        corpus_declaraciones.extend(numerales_doc)
        time.sleep(1) 
        
    return corpus_declaraciones

def obtener_corpus_declaraciones(modo="estatico", archivo_backup="corpus_declaraciones_v1.json", max_intentos=3):
    """Gestor principal de datos (ETL)."""
    
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    carpeta_salida = os.path.join(directorio_script, "..", "datos")
    ruta_completa = os.path.join(carpeta_salida, archivo_backup)

    if modo == "estatico":
        print(f"[*] Cargando desde '{ruta_completa}'...")
        if os.path.exists(ruta_completa):
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"[!] Error: No se encontró el archivo. Ejecuta en modo 'dinamico'.")
            return None

    elif modo == "dinamico":
        print(f"\n[*] MODO DINÁMICO: Extrayendo datos desde vatican.va...")
        for intento in range(1, max_intentos + 1):
            try:
                corpus_fresco = ejecutar_scraping_declaraciones()
                os.makedirs(carpeta_salida, exist_ok=True)
                
                documento_maestro = {
                    "nivel_jerarquico": 1, 
                    "fuente": "Declaraciones - Concilio Vaticano II",
                    "fecha_generacion": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                    "total_numerales": len(corpus_fresco),
                    "datos": corpus_fresco
                }
                
                with open(ruta_completa, "w", encoding="utf-8") as f:
                    json.dump(documento_maestro, f, ensure_ascii=False, indent=4)
                    
                print(f"\n[+] ¡ÉXITO! Guardado en: '{ruta_completa}'.")
                return documento_maestro
            except Exception as e:
                print(f"\n[!] Error en intento {intento}: {str(e)}")
                if intento < max_intentos: time.sleep(5)
    return None

if __name__ == '__main__':
    MODO_EJECUCION = "dinamico"
    ARCHIVO_SALIDA = "corpus_declaraciones_v1.json"
    
    corpus_maestro = obtener_corpus_declaraciones(modo=MODO_EJECUCION, archivo_backup=ARCHIVO_SALIDA)
    
    if corpus_maestro:
        total = corpus_maestro.get("total_numerales", 0)
        print(f"\n[+] Motor inicializado: {total} fragmentos extraídos correctamente.")