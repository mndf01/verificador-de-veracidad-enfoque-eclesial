import requests
from bs4 import BeautifulSoup
import re
import json
import os
from datetime import datetime

# URL directa del Compendio
URL_COMPENDIO = "https://www.vatican.va/roman_curia/pontifical_councils/justpeace/documents/rc_pc_justpeace_doc_20060526_compendio-dott-soc_sp.html"

def limpiar_espacios_puntuacion(texto):
    """Limpia los espacios residuales dejados por la eliminación de etiquetas HTML."""
    texto = texto.replace('"', "'")
    texto = re.sub(r'\s+([.,:;!?\)\]»”’])', r'\1', texto)
    texto = re.sub(r'([\({\[«“‘])\s+', r'\1', texto)
    texto = re.sub(r'\s{2,}', ' ', texto)
    return texto.strip()

def extraer_compendio(url):
    """Extrae exclusivamente los numerales del Compendio."""
    print(f"[*] Descargando el Compendio de la Doctrina Social...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    
    if response.status_code != 200:
        raise ConnectionError(f"Fallo HTTP {response.status_code} en: {url}")

    html_crudo = response.text 
    
    html_principal = re.sub(r'</(p|div|h[1-6]|li)>', '\n\n', html_crudo, flags=re.IGNORECASE)
    html_principal = re.sub(r'<br\s*/?>', '\n', html_principal, flags=re.IGNORECASE)
    
    soup = BeautifulSoup(html_principal, 'html.parser')
    
    for sup in soup.find_all('sup'):
        sup.decompose()
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    texto_plano = soup.get_text(separator=' ')
    texto_plano = re.sub(r'\[\s*(?:AR|DE|EN|ES|FR|IT|HE|HR|HU|LA|LV|PT|SW|ZH|CS|BE|NL|PL|SQ|UK|VI).*?\]', '', texto_plano, flags=re.IGNORECASE | re.DOTALL)
    
    # =========================================================
    # CIRUGÍA DE PRECISIÓN: Cortar las notas al pie
    # El numeral 583 termina con "posesión eterna de Ti mismo...»."
    # =========================================================
    marcador_final = "posesión eterna de Ti mismo"
    if marcador_final in texto_plano:
        # Cortamos el string completo exactamente ahí
        texto_plano = texto_plano[:texto_plano.find(marcador_final) + len(marcador_final)] + "...»."
    
    numerales = []
    current_id = None
    current_text = []
    contexto_base = "Compendio de la Doctrina Social de la Iglesia"
    documento_iniciado = False 
    
    for linea in texto_plano.split('\n'):
        linea = linea.strip()
        linea = re.sub(r'\s+', ' ', linea) 
        
        if not linea:
            continue
            
        linea_upper = linea.upper()

        if not documento_iniciado:
            # Esperamos a la firma de Crepaldi para iniciar la Introducción (Numeral 1)
            if "CREPALDI" in linea_upper:
                documento_iniciado = True
            continue 
            
        match_numeral = re.match(r'^(\d+)\s+(.*)', linea)
        
        if match_numeral and not re.match(r'^[A-Za-z]{1,3}\s+\d+', match_numeral.group(2)):
            num_capturado = int(match_numeral.group(1))
            
            # =========================================================
            # LÓGICA SECUENCIAL ESTRICTA
            # =========================================================
            if current_id is None:
                if num_capturado != 1:
                    continue # Aún no encontramos el numeral 1
            else:
                # Un numeral válido no puede retroceder ni saltarse al azar
                # Ej: Si current_id es 250, "30" es rechazado y tratado como texto normal.
                if num_capturado <= current_id or (num_capturado - current_id) > 2:
                    current_text.append(linea)
                    continue

            # Guardar el bloque anterior cuando encontramos un numeral nuevo
            if current_text:
                texto_limpio = limpiar_espacios_puntuacion(" ".join(current_text))
                if texto_limpio:
                    numerales.append({
                        "numeral_id": current_id,
                        "contexto_jerarquico": contexto_base,
                        "texto": texto_limpio
                    })
            
            # Iniciar el nuevo bloque
            current_id = num_capturado
            texto_restante = match_numeral.group(2).strip()
            current_text = [texto_restante] if texto_restante else []
            continue

        if current_id is not None:
            if len(linea) < 60 and not re.search(r'[.:;!?\)\]”\']$', linea):
                continue
            current_text.append(linea)
            
    # Guardar el último numeral válido (el 583)
    if current_text and current_id is not None:
        texto_limpio = limpiar_espacios_puntuacion(" ".join(current_text))
        if texto_limpio:
            numerales.append({
                "numeral_id": current_id,
                "contexto_jerarquico": contexto_base,
                "texto": texto_limpio
            })
            
    return numerales

def obtener_corpus_compendio(modo="estatico", archivo_backup="corpus_compendio_v1.json"):
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
        print(f"\n[*] MODO DINÁMICO: Extrayendo el Compendio desde vatican.va...")
        try:
            corpus_fresco = extraer_compendio(URL_COMPENDIO)
            os.makedirs(carpeta_salida, exist_ok=True)
            
            documento_maestro = {
                "nivel_jerarquico": 1, 
                "fuente": "Compendio de la Doctrina Social de la Iglesia",
                "url_origen": URL_COMPENDIO, 
                "fecha_generacion": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                "total_numerales": len(corpus_fresco),
                "datos": corpus_fresco
            }
            
            with open(ruta_completa, "w", encoding="utf-8") as f:
                json.dump(documento_maestro, f, ensure_ascii=False, indent=4)
                
            print(f"\n[+] ¡ÉXITO! Compendio guardado en: '{ruta_completa}'.")
            return documento_maestro
            
        except Exception as e:
            print(f"\n[!] Error crítico durante la extracción: {str(e)}")
            return None

if __name__ == '__main__':
    MODO_EJECUCION = "dinamico" 
    ARCHIVO_SALIDA = "corpus_compendio_v1.json"
    
    corpus_maestro = obtener_corpus_compendio(modo=MODO_EJECUCION, archivo_backup=ARCHIVO_SALIDA)
    
    if corpus_maestro:
        total = corpus_maestro.get("total_numerales", 0)
        print(f"\n[+] Motor inicializado: {total} numerales extraídos correctamente.")