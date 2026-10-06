# Verificador de Veracidad — Enfoque Eclesial

Sistema que detecta inconsistencias entre **noticias/artículos digitales** y una **base cerrada de documentos normativos de la Iglesia Católica** (Código de Derecho Canónico, Compendio de la Doctrina Social, Constituciones/Declaraciones/Decretos del Concilio Vaticano II).

**Pipeline general:**

```
Corpus canónico (vatican.va)
   │  scrapers (modo dinámico)  o  archivos locales (modo estático)
   ▼
RAW (_v1.json) ──► clasificador de metadatos ──► GOLD (_clasificado.json)
                                                        │
                                                        ▼
                                             ChromaDB (RAG: recuperación semántica)
                                                        │
   Noticia/web ──► web_extractor ──► heurísticas IVR + Random Forest ──► veredicto
```

---

## Requisitos

- **Python ≥ 3.12** (probado en 3.14)
- `git`
- Internet solo para el primer `pip install` y para el modo `"dinamico"` de los scrapers

> ⚠️ **torch (CPU):** `sentence-transformers` descarga `torch` solo. Para evitar la versión CUDA (~2 GB), instalar primero la variante CPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Instalación (desde cero)

```bash
git clone <url-del-repo>
cd tesis

# 1. Entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Dependencias (ver aviso de torch arriba)
#    -e = modo editable: vincula `config` a la raíz del repo.
#    (Con `pip install .` a secas se copia config.py a site-packages y
#    queda una copia vieja que puede importarse por error desde otras rutas.)
pip install -e .

# 3. Smoke test: abre la colección ChromaDB y muestra cuántos docs tiene
python main.py
# Salida esperada si ya se indexó: un número (2965 con el corpus completo)
# Salida si está vacía: 0
```

---

## Estructura del proyecto

```
tesis/
├── config.py            # Modelos pydantic que leen config.toml + resolver_ruta()
├── config.toml          # TODA la configuración (rutas, modelos, modo ETL)
├── main.py              # Smoke test de la capa base_datos
├── pyproject.toml       # Dependencias del proyecto
│
├── src/
│   ├── base_datos/          # Capa RAG (ChromaDB + embeddings)
│   │   ├── embeddings.py    # Función de embedding (modelo SBERT desde config)
│   │   └── chromaDB.py      # Cliente, colección, normalizar item, indexar corpus
│   │
│   ├── corp_extractor/      # Pipeline ETL del corpus canónico
│   │   ├── main_etl.py      # Orquestador (raw → clasificado)
│   │   ├── extractores/     # 5 scrapers (cánones, compendio, constituciones,
│   │   │                    #   declaraciones, decretos) → generan los _v1.json
│   │   └── clasificador_corpus/
│   │       └── clasificador_metadatos_v1.py   # Inyecta categoria + nivel_autoridad
│   │
│   ├── expansion_dinamica/  # Carga transitoria de UN PDF externo (aislada del corpus base)
│   │   ├── sesion.py        # SesionExpansion: API pública (cargar_pdf / consultar / consultar_combinado)
│   │   ├── procesamiento.py # procesar_pdf(): extraer → limpiar → segmentar (sin Chroma)
│   │   ├── extractor_pdf.py # PDF → texto por página (sin OCR; valida tamaño/escaneado/contraseña)
│   │   ├── columnas.py      # PDF de 2 columnas: detecta el canal y reconstruye el orden de lectura
│   │   ├── limpieza.py      # encabezados, pies, n.º de página, guiones, ligaturas
│   │   ├── segmentador.py   # Regex: artículos/cánones/numerales → fragmentos + contexto jerárquico
│   │   ├── parametros.py    # ParametrosExpansion (lee [expansion_dinamica] de config.toml)
│   │   ├── errores.py       # ExpansionError y subclases (mensajes aptos para la interfaz)
│   │   └── tipos.py         # dataclasses internas
│   │
│   ├── procesamiento_nlp/
│   │   └── tesauro.json     # Tesauro de categorías (insumo del IVR)
│   │
│   └── web_extractor/       # Extracción de contenido de noticias (pendiente)
│       └── web_extractor.py # Boceto con trafilatura
│
├── tests/
│   ├── test_expansion_dinamica.py   # pruebas del módulo de expansión dinámica
│   └── ejemplos/                    # PDF de ejemplo (1 y 2 columnas) para probarlo a mano
│
└── data/               # ⚠️ IGNORADA por git — se regenera con el ETL
   ├── corp_extractor/
   │   ├── raw/         #    BRONCE: corpus_*_v1.json (crudos)
   │   └── clasificado/ #    GOLD:   corpus_*_clasificado.json (para Chroma)
   └── chroma/          # Base vectorial persistente (chroma.sqlite3)

```

**Regla de oro:** el MISMO modelo de embeddings (`models.sb_activo`) alimenta la base vectorial **y** el IVR. Nunca cambiar uno sin el otro.

---

## Configuración (`config.toml` + `config.py`)

Todas las rutas y parámetros viven en `config.toml` y se leen con **pydantic** vía `obtener_configuraciones()`.

```toml
[models]
sb_activo = "paraphrase-multilingual-MiniLM-L12-v2"   # modelo de embeddings

[paths]
data_dir = "data"
raw_corpus_dir = "data/corp_extractor/raw"             # BRONCE: _v1.json
corpus_dir = "data/corp_extractor/clasificado"         # GOLD: _clasificado.json
chroma_dir = "data/chroma"                             # BD vectorial

[chroma]
coleccion = "corpus_canonico"     # colección única para los 5 corpus
espacio_hnsw = "cosine"           # similitud = 1 - distancia
batch_size = 200                  # lote de upsert a Chroma

[corp_extractor]
modo_ejecucion = "estatico"       # "estatico" (local) | "dinamico" (descarga web)
```

**`resolver_ruta()`:** las rutas del `config.toml` son relativas y se resuelven **contra la raíz del proyecto** (no contra el directorio actual). Siempre usar `resolver_ruta(settings.paths.X)` y nunca rutas hardcodeadas.

### Cómo leer una variable

```python
from config import obtener_configuraciones

settings = obtener_configuraciones()
print(settings.chroma.coleccion)
```

### Cómo agregar una variable nueva

1. En `config.toml`:
   ```toml
   [mi_seccion]
   mi_variable = "valor"
   batch_size = 200
   ```
2. En `config.py`, agregar el modelo pydantic y registrarlo en `Settings`:
   ```python
   class MiSeccionConfig(BaseModel):
       mi_variable: str
       batch_size: int

   class Settings(BaseSettings):
       ...
       mi_seccion: MiSeccionConfig
   ```
3. Leer con `settings.mi_seccion.mi_variable`.

---

## Flujo de datos y comandos

### 1. ETL del corpus (raw → clasificado)

```bash
python src/corp_extractor/main_etl.py
```

El orquestador lee los 5 crudos de `raw_corpus_dir`, les inyecta `categoria` + `nivel_autoridad` con el clasificador y escribe los `_clasificado.json` en `corpus_dir`.

- **`modo_ejecucion = "estatico"`**: no usa internet, toma los `_v1.json` ya descargados.
- **`modo_ejecucion = "dinamico"`**: descarga desde `vatican.va` y regenera los crudos.

### 2. Indexar el corpus en ChromaDB (RAG)

```bash
python -m src.base_datos.chromaDB    # ejecuta indexar_corpus()
```

Lee cada `*_clasificado.json` de `corpus_dir`, normaliza items a `(ids, documents, metadatas)` y hace `upsert` por lotes en la colección única `corpus_canonico`.

**Detalles de implementación importantes:**

- **ID de Chroma** = `f"{tipo_corpus}:{indice:04d}"` — posición dentro del archivo (único y determinístico). El número real del canon/numeral va en metadata como `id_norma` (NO es único en algunos corpus).
- **`canon_id` vs `numeral_id`:** el Código usa `canon_id`, los otros 4 corpus usan `numeral_id` (mutuamente excluyentes) → normalizar con `.get('numeral_id', .get('canon_id'))`.
- **`url_origen` es condicional:** el Compendio no la tiene en sus items (0/583) → solo guardarla si existe, nunca `None`.
- **Metadata solo escalares** (`str/int/float/bool`), sin `None`, sin listas/dicts anidados.

### 3. Consulta a ChromaDB (próximamente)

*Pendiente: capa de consulta (recuperación semántica) sobre `corpus_canonico`.*

### 4. Web / noticias → veredicto (próximamente)

*Pendiente: `web_extractor` (trafilatura) + heurísticas IVR + Random Forest.*

### 5. Expansión dinámica (PDF externo, transitorio)

Permite que el usuario cargue **un PDF normativo propio** (p. ej. un decreto diocesano) durante la sesión de validación, y que la noticia se contraste **también** contra ese documento, **sin tocar el corpus base** (`corpus_canonico`). Todo vive en memoria y desaparece al cerrar la sesión.

> **Alcance:** solo PDF con **texto seleccionable**. No hace OCR: un PDF escaneado (imágenes) se rechaza con un mensaje claro.

#### Uso mínimo

```python
from src.expansion_dinamica import SesionExpansion, ExpansionError

sesion = SesionExpansion()          # UNA por usuario (en Streamlit: guardarla en st.session_state)

try:
    resumen = sesion.cargar_pdf(archivo, nivel_autoridad=0.33)   # ruta, bytes o archivo subido
except ExpansionError as e:
    print(e)                        # mensaje listo para mostrar al usuario
else:
    print(resumen.fragmentos, resumen.esquema, resumen.advertencias)

# Corpus base + PDF cargado, ordenados juntos por similitud (n_resultados en total)
evidencias = sesion.consultar_combinado("El obispo puede dispensar...", n_resultados=5)

sesion.cerrar()                     # descarta todo (o usar:  with SesionExpansion() as s: ...)
```

| Método | Qué hace |
|---|---|
| `cargar_pdf(archivo, nombre=None, *, nivel_autoridad=None, categoria=None, url_origen=None)` | Procesa e indexa un PDF. Devuelve un `ResumenDocumento` (`doc_id`, `nombre`, `paginas`, `fragmentos`, `esquema`, `nivel_autoridad`, `categoria`, `advertencias`, `ya_cargado`). Cargar el mismo PDF dos veces no lo duplica. |
| `consultar(premisa, n_resultados=5)` | Busca **solo** en los PDF cargados. |
| `consultar_combinado(premisa, n_resultados=5, where_base=None)` | Busca en el corpus base **y** en los PDF cargados. Sin PDF cargados devuelve exactamente lo que devolvería `chromaDB.consultar`. |
| `documentos()` / `eliminar_documento(doc_id)` / `cerrar()` | Gestión de lo cargado en la sesión. |

#### Qué devuelve cada resultado

Las mismas claves que `chromaDB.consultar()`, más cuatro extras. **Quien calcule el IVR puede tratar ambas fuentes igual**: las similitudes son comparables (mismo modelo de embeddings y mismo espacio coseno).

| Clave | Contenido |
|---|---|
| `id_chroma`, `texto`, `similitud`, `contexto`, `fuente`, `id_norma`, `tipo_corpus`, `categoria`, `nivel_autoridad`, `url_origen`, `cita` | Igual que el corpus base. En un PDF cargado: `tipo_corpus = "expansion_dinamica"`, `id_norma` = n.º de artículo/canon/numeral (0 si la unidad no está numerada), `cita` = p. ej. `Decreto 12/2024, Artículo 4, p. 2`. |
| `origen` | `"corpus_base"` o `"expansion_dinamica"`. Sirve para distinguir de dónde viene cada evidencia. |
| `doc_id`, `pagina_inicio`, `pagina_fin` | Solo en resultados de un PDF cargado. |

**`nivel_autoridad`** (Coeficiente Jerárquico del IVR): lo elige el usuario entre **0.33** (Nivel 1, norma local — valor por defecto), **0.66** (Nivel 2) y **1.0** (Nivel 3). Cualquier otro valor lanza `NivelAutoridadInvalidoError`.

#### Parámetros (`[expansion_dinamica]` en `config.toml`)

| Parámetro | Por defecto | Qué controla |
|---|---|---|
| `max_mb` | `20` | Tamaño máximo del PDF. |
| `max_paginas` | `300` | Páginas máximas. |
| `min_caracteres_por_pagina` | `100` | Una página con menos texto se considera "sin texto" (escaneada o imagen). |
| `max_fraccion_paginas_vacias` | `0.5` | Si más de esta fracción de páginas no tiene texto, el PDF se rechaza por escaneado. |
| `detectar_columnas` | `true` | Reordena PDFs de dos columnas (ver más abajo). |
| `palabras_max_chunk` | `500` | Una unidad (artículo/canon) más larga se divide en partes de este tamaño máximo. |
| `palabras_min_chunk` | `20` | Por debajo se emite una advertencia (los textos muy breves dan embeddings poco fiables). |
| `nivel_autoridad_default` | `0.33` | Nivel por defecto si el usuario no elige. |
| `categoria_default` | `"Normativa Externa (carga dinámica)"` | Categoría por defecto de lo cargado. |

También reutiliza `chroma.espacio_hnsw`, `chroma.batch_size`, `chroma.max_results` y `chroma.telemetria`.

#### Errores y advertencias que la interfaz debe mostrar

Todas las excepciones heredan de `ExpansionError`, así que alcanza con un solo `except`; el mensaje ya está redactado para el usuario.

| Excepción | Cuándo |
|---|---|
| `PdfInvalidoError` | No es un PDF, está vacío, dañado o protegido con contraseña. |
| `PdfExcedeLimiteError` | Supera `max_mb` o `max_paginas`. |
| `PdfSinTextoNativoError` | Escaneado o formado por imágenes (no hay OCR). |
| `DocumentoSinContenidoError` | Tras limpiar el texto no quedó nada utilizable. |
| `NivelAutoridadInvalidoError` | `nivel_autoridad` no es 0.33, 0.66 ni 1.0. |

`resumen.advertencias` **no son errores**: avisos para mostrar al usuario (p. ej. "no se detectó estructura normativa, se cortó por ventanas", "numeración con huecos", "se detectó maquetación en dos columnas", "páginas sin texto").

#### Cómo funciona por dentro

1. **Validación:** formato PDF, tamaño, páginas, contraseña y si es un escaneado.
2. **Extracción** del texto de cada página. Si el PDF es de **dos columnas**, se leen las coordenadas de cada palabra (`pdfplumber`) y se reconstruye el orden de lectura: título a todo el ancho → columna izquierda → columna derecha → pie. Es conservador: ante tres columnas, tablas o notas al margen deja el texto como está.
3. **Limpieza:** quita encabezados y pies repetidos y números de página, une las palabras cortadas con guion al final de línea y normaliza ligaturas.
4. **Segmentación por expresiones regulares**, en este orden: *articulado* (`Artículo N`, `Can. N`, con `LIBRO/TÍTULO/CAPÍTULO` y subtítulos como contexto jerárquico) → *numerado* (`12. Texto…` con numeración creciente) → *ventanas* de ~500 palabras si no hay estructura.
5. **Indexación** en una colección propia de un cliente Chroma **en memoria** (`EphemeralClient`), con el mismo modelo de embeddings que el corpus base. Nunca se escribe en `chroma_dir`. Si falla a mitad de camino, la carga se revierte.
6. **Consulta:** busca en el PDF (y, si se pide, en el corpus base) y devuelve todo en el mismo formato.

#### Límites conocidos

- **Sin OCR.** Tres o más columnas y tablas: se deja el texto como sale, con riesgo de mezcla.
- Un documento con formato muy atípico cae a **ventanas** (la cita apunta a páginas, no a artículos); siempre hay una advertencia.
- La sesión vive en la memoria del proceso: si se reinicia la aplicación hay que volver a cargar el PDF.
- Costo: un PDF de una columna añade <1 s de análisis; uno de dos columnas, ~0.1 s por página (300 páginas ≈ 30 s). Se apaga con `detectar_columnas = false`.
- Probado con PDF generados a partir del propio corpus (cánones y numerales: 98–100 % de unidades idénticas al original) y con PDF de ejemplo. **Falta probarlo con PDF reales de otras fuentes.**

#### Probar

```bash
python -m unittest discover -s tests         # pruebas (o: pytest tests — requiere pip install -e ".[dev]")

# Solo cortar el documento y ver cómo lo segmenta (NO carga el modelo SBERT):
python -m src.expansion_dinamica tests/ejemplos/decreto_diocesano_ejemplo.pdf --solo-segmentar
python -m src.expansion_dinamica tests/ejemplos/decreto_dos_columnas_ejemplo.pdf --solo-segmentar

# Cargar y consultar de verdad (usa el modelo SBERT):
python -m src.expansion_dinamica ruta/al/documento.pdf --premisa "texto de la noticia"

# Comprobar que config.toml se lee bien:
python -c "from config import obtener_configuraciones as o; print(o().expansion_dinamica)"
```

**Problemas frecuentes**

- `No module named 'chromadb'` (u otra librería): activar el entorno virtual e instalar con `pip install -e .`.
- `CERTIFICATE_VERIFY_FAILED` al cargar el modelo: lo causa un proxy o antivirus que intercepta la conexión. No es grave si el modelo ya está en la caché local; el programa sigue con él.
- La primera ejecución descarga el modelo SBERT (~470 MB).

---

## Notas para el equipo

- **`data/` NO se commitea.** Está en `.gitignore`. Tras clonar, regenerar con el ETL (modo estatico necesita los `_v1.json`; si no los tenés, corré en modo `"dinamico"` para descargarlos).
- **`main.py`** es solo un smoke test (abre la colección y cuenta). No borrar.
- **Si instalás una librería nueva**, agregala a `dependencies` en `pyproject.toml`.
- **Si agregás una sección a `config.toml`**, actualizá `config.py` (ver arriba).
- Los artefactos `build/`, `*.egg-info/`, `.idea/` son locales y no se versionan.

## Estado actual (hoja de ruta)

| Fase | Estado |
|---|---|
| Config centralizada (config.toml/pydantic) | ✅ |
| Scrapers + ETL → raw → clasificado | ✅ |
| Capa ChromaDB (cliente, colección, indexar) | ✅ en desarrollo |
| Consulta semántica (RAG) | ⏳ |
| Extracción de noticias (web_extractor) | ⏳ |
| IVR (heurísticas semánticas + tesauro) | ⏳ |
| Random Forest + veredicto | ⏳ |
| Expansión dinámica (PDF externo, transitorio) | ✅ módulo + pruebas · ⏳ conectar con la interfaz y con el IVR |
