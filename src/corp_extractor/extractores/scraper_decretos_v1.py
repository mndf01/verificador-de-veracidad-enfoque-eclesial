import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
from datetime import datetime
from config import obtener_configuraciones, resolver_ruta

# --- URLs BASE ---
BASE_URL_VAT2 = "https://www.vatican.va/archive/hist_councils/ii_vatican_council/"
INDEX_URL = "https://www.vatican.va/archive/hist_councils/ii_vatican_council/index_sp.htm"

def obtener_enlaces_decretos(url_indice):
    """Navega el índice y extrae dinámicamente las URLs de los Decretos sin duplicados."""
    print(f"[*] Analizando el árbol DOM del índice para mapear Decretos...")
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
        # Identificamos Decretos ('_decree_') en español ('_sp.html')
        if '_decree_' in href and href.endswith('_sp.html'):
            url_completa = BASE_URL_VAT2 + href
            if url_completa not in urls_vistas:
                titulo = a.get_text(strip=True)
                titulo = re.sub(r'\s+', ' ', titulo).strip()
                documentos_objetivo.append({'titulo': titulo, 'url': url_completa})
                urls_vistas.add(url_completa)
            
    print(f"[*] Se detectaron {len(documentos_objetivo)} Decretos únicos para la ingesta.\n")
    return documentos_objetivo

def extraer_numerales(url, titulo_documento):
    """Extrae los numerales procesando el texto, sin arrastrar subtítulos."""
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
    
    # =========================================================
    # 1. LIMPIEZA PREVIA AL SPLIT
    # =========================================================
    texto_plano = re.sub(r'\[\s*\d+\s*\]', '', texto_plano)
    # Limpiar menú de idiomas (incluso multilínea)
    texto_plano = re.sub(r'\[\s*(?:AR|DE|EN|ES|FR|IT|HE|HR|HU|LA|LV|PT|SW|ZH|CS|BE).*?\]', '', texto_plano, flags=re.IGNORECASE | re.DOTALL)
    # Limpiar fórmula de promulgación y todo lo que le siga
    texto_plano = re.sub(r'Todas y cada una de las cosas\s*(contenidas|establecidas|que en est).*', '', texto_plano, flags=re.IGNORECASE | re.DOTALL)
    
    numerales = []
    current_id = None
    current_text = []
    
    contexto_base = f"Concilio Vaticano II > Decretos > {titulo_documento}"
    contexto_actual = contexto_base
    en_seccion_especial = False
    
    for linea in texto_plano.split('\n'):
        linea = linea.strip()
        linea = re.sub(r'\s+', ' ', linea) 
        
        if not linea:
            continue
            
        linea_upper = linea.upper()
        
        # =========================================================
        # 2. CORTAFUEGOS Y FILTROS DE RUIDO
        # =========================================================
        if linea_upper == 'NOTAS' or linea_upper.startswith('NOTAS '):
            break
            
        if re.search(r'^(ROMA, EN SAN PEDRO|YO, PABLO|\* CONSTITUCIÓN PROMULGADA|\* DECLARACIÓN PROMULGADA|\* DECRETO PROMULGADO|PERICLES FELICI|ARZOBISPO TITULAR|SECRETARIO GENERAL|PABLO OBISPO|DECRETO)$', linea_upper):
            continue

        if re.match(r'^(CAP[ÍI]TULO|SECCI[ÓO]N|ART\.)\b', linea_upper):
            continue
            
        # Ignorar números romanos seguidos de un punto que usan en los Decretos (ej: "I. En cada nación...")
        if re.match(r'^(I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+', linea_upper):
             continue

        # =========================================================
        # 3. DETECCIÓN DE SECCIONES ANEXAS (Proemio, Conclusión)
        # =========================================================
        match_seccion = re.match(r'^(A P É N D I C E|APÉNDICE|DE LAS ACTAS|NOTIFICACIONES|NOTA EXPLICATIVA|CONCLUSI[ÓO]N|INTRODUCCI[ÓO]N|PROEMIO)\b', linea_upper)
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
            en_seccion_especial = True
            continue

        # =========================================================
        # 4. DETECCIÓN DE NUMERALES
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
            en_seccion_especial = False 
            
            if not ("APÉNDICE" in contexto_actual or "NOTA EXPLICATIVA" in contexto_actual):
               contexto_actual = contexto_base
               
            continue

        # =========================================================
        # 5. FILTRO DE SUBTÍTULOS INTRUSOS (Destrucción directa)
        # =========================================================
        if len(linea) < 150 and not re.search(r'[.:;!?\)\]"”\']$', linea):
            continue

        # =========================================================
        # 6. ASIGNACIÓN DEL TEXTO
        # =========================================================
        if en_seccion_especial:
            current_text.append(linea) 
        elif current_id is not None:
            current_text.append(linea)
        else:
            current_id = 0
            current_text.append(linea)
            
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

def ejecutar_scraping_decretos():
    documentos = obtener_enlaces_decretos(INDEX_URL)
    corpus_decretos = []
    
    for i, doc in enumerate(documentos):
        print(f"[{i+1}/{len(documentos)}] Extrayendo: {doc['titulo']}")
        numerales_doc = extraer_numerales(doc['url'], doc['titulo'])
        corpus_decretos.extend(numerales_doc)
        time.sleep(1) 
        
    return corpus_decretos

def obtener_corpus_decretos(modo="estatico", archivo_backup="corpus_decretos_v1.json", max_intentos=3):
    settings = obtener_configuraciones()
    # Zona Bronze: crudos (_v1.json) guardados en la carpeta configurada
    carpeta_salida = resolver_ruta(settings.paths.raw_corpus_dir)
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
                corpus_fresco = ejecutar_scraping_decretos()
                os.makedirs(carpeta_salida, exist_ok=True)
                
                documento_maestro = {
                    "nivel_jerarquico": 1,
                    "tipo_corpus": "corpus_decretos",
                    "fuente": "Decretos - Concilio Vaticano II",
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
    ARCHIVO_SALIDA = "corpus_decretos_v1.json"
    
    corpus_maestro = obtener_corpus_decretos(modo=MODO_EJECUCION, archivo_backup=ARCHIVO_SALIDA)
    
    if corpus_maestro:
        total = corpus_maestro.get("total_numerales", 0)
        print(f"\n[+] Motor inicializado: {total} fragmentos extraídos correctamente.")