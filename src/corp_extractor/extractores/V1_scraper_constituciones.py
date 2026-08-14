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

def obtener_enlaces_constituciones(url_indice):
    """Navega el índice y extrae dinámicamente las URLs sin duplicados."""
    print(f"[*] Analizando el árbol DOM del índice para mapear Constituciones...")
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
        if '_const_' in href and href.endswith('_sp.html'):
            url_completa = BASE_URL_VAT2 + href
            if url_completa not in urls_vistas:
                titulo = a.get_text(strip=True)
                titulo = re.sub(r'\s+', ' ', titulo).strip()
                documentos_objetivo.append({'titulo': titulo, 'url': url_completa})
                urls_vistas.add(url_completa)
            
    print(f"[*] Se detectaron {len(documentos_objetivo)} Constituciones únicas para la ingesta.\n")
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
    
    # Parseamos el HTML ya modificado
    soup = BeautifulSoup(html_principal, 'html.parser')
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    texto_plano = soup.get_text(separator=' ')
    # Limpiamos las llamadas a notas al pie [1], [2], etc.
    texto_plano = re.sub(r'\[\s*\d+\s*\]', '', texto_plano)
    
    numerales = []
    current_id = None
    current_text = []
    
    contexto_base = f"Concilio Vaticano II > Constituciones > {titulo_documento}"
    contexto_actual = contexto_base
    
    for linea in texto_plano.split('\n'):
        linea = linea.strip()
        linea = re.sub(r'\s+', ' ', linea) # Normalizamos espacios internos
        
        if not linea:
            continue
            
        linea_upper = linea.upper()
        
        # =========================================================
        # 1. CORTAFUEGOS DEFINITIVO (El fin del documento real)
        # =========================================================
        # Si llegamos a la sección de NOTAS (bibliografía), se acabó el documento.
        if linea_upper == 'NOTAS' or linea_upper.startswith('NOTAS '):
            break
            
        # Filtro del menú de idiomas superior/inferior
        if re.match(r'^\[.*AR.*BE.*CS.*\]$', linea_upper):
            continue
            
        # =========================================================
        # 2. FILTROS DE RUIDO Y METADATOS 
        # =========================================================
        # Elimina las fórmulas de promulgación y las firmas
        if re.search(r'^(ROMA, EN SAN PEDRO|YO, PABLO|\* CONSTITUCIÓN PROMULGADA|PERICLES FELICI|ARZOBISPO TITULAR|SECRETARIO GENERAL|PABLO OBISPO)', linea_upper):
            continue
        
        # Eliminar el párrafo de promulgación entero
        if re.search(r'^(TODAS Y CADA UNA DE LAS COSAS)', linea_upper):
            continue

        if re.match(r'^(CAP[ÍI]TULO|SECCI[ÓO]N|PROEMIO|CONCLUSI[ÓO]N|INTRODUCCI[ÓO]N)\b', linea_upper):
            continue
            
        # =========================================================
        # 3. DETECCIÓN DE SECCIONES ANEXAS (Nivel Jerárquico Extra)
        # =========================================================
        match_seccion = re.match(r'^(A P É N D I C E|APÉNDICE|DE LAS ACTAS|NOTIFICACIONES|NOTA EXPLICATIVA)\b', linea_upper)
        if match_seccion:
            if current_text:
                texto_limpio = " ".join(current_text).strip().replace('"', "'")
                if texto_limpio:
                    numerales.append({
                        "numeral_id": current_id if current_id is not None else 0,
                        "contexto_jerarquico": contexto_actual,
                        "texto": texto_limpio,
                        "url_origen": url
                    })
            
            nombre_seccion = match_seccion.group(1).replace('A P É N D I C E', 'APÉNDICE')
            contexto_actual = f"{contexto_base} > {nombre_seccion}"
            current_id = 0
            current_text = []
            continue

        # =========================================================
        # 4. DETECCIÓN DE NUMERALES (Caza mutantes como 1.ª, 2. a)
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
            
            current_id = int(match_numeral.group(1))
            texto_restante = match_numeral.group(2).strip()
            current_text = [texto_restante] if texto_restante else []
            continue

        # =========================================================
        # 5. FILTRO DE SUBTÍTULOS INTRUSOS
        # =========================================================
        # Si la línea tiene menos de 150 caracteres y no termina con puntuación lógica, ES UN TÍTULO.
        if len(linea) < 150 and not re.search(r'[.:;!?\)\]"”\']$', linea):
            continue
            
        # Si sobrevivió a todo, es texto real
        if current_id is not None:
            current_text.append(linea)
        else:
            current_id = 0
            current_text.append(linea)
            
    # Guardar el ultimísimo bloque en memoria
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

def ejecutar_scraping_constituciones():
    documentos = obtener_enlaces_constituciones(INDEX_URL)
    corpus_constituciones = []
    
    for i, doc in enumerate(documentos):
        print(f"[{i+1}/{len(documentos)}] Extrayendo: {doc['titulo']}")
        numerales_doc = extraer_numerales(doc['url'], doc['titulo'])
        corpus_constituciones.extend(numerales_doc)
        time.sleep(1) 
        
    return corpus_constituciones

def obtener_corpus_constituciones(modo="estatico", archivo_backup="corpus_constituciones_v1.json", max_intentos=3):
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
                corpus_fresco = ejecutar_scraping_constituciones()
                os.makedirs(carpeta_salida, exist_ok=True)
                
                documento_maestro = {
                    "nivel_jerarquico": 1, 
                    "fuente": "Constituciones - Concilio Vaticano II",
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
    MODO_EJECUCION = "estatico"
    ARCHIVO_SALIDA = "corpus_constituciones_v1.json"
    
    corpus_maestro = obtener_corpus_constituciones(modo=MODO_EJECUCION, archivo_backup=ARCHIVO_SALIDA)
    
    if corpus_maestro:
        total = corpus_maestro.get("total_numerales", 0)
        print(f"\n[+] Motor inicializado: {total} fragmentos extraídos correctamente.")