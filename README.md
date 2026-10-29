# Verificador de Veracidad — Enfoque Eclesial

Sistema que detecta inconsistencias entre **noticias/artículos digitales** y una **base cerrada de documentos normativos de la Iglesia Católica** (Código de Derecho Canónico, Compendio de la Doctrina Social, Constituciones/Declaraciones/Decretos del Concilio Vaticano II), con soporte para **expansión dinámica transitoria** mediante documentos PDF externos.

**Pipeline general:**

```
Corpus canónico (vatican.va)
   │  scrapers (modo dinámico)  o  archivos locales (modo estático)
   ▼
RAW (_v1.json) ──► clasificador de metadatos (categoría + J graduado) ──► GOLD (_clasificado.json)
                                                                                 │
                                                                                 ▼
PDF externo opcional (--pdf) ──► SesionExpansion (EphemeralClient) ──┐    ChromaDB (RAG: corpus_canonico)
                                                                     ▼           ▼
   Noticia/URL ──► web_extractor ──► premisa ──► consultar_combinado / consultar
                                                                 │
                                                                 ▼
                                           IVR (SB + L + SO + J) ──► Random Forest ──► veredicto
```

---

## Requisitos

- **Python ≥ 3.12** (probado en 3.13 y 3.14, Linux/macOS y Windows PowerShell)
- `git`
- Internet solo para el primer `pip install`, la descarga inicial del modelo SBERT y para consultar URLs en vivo o el modo `"dinamico"` de los scrapers.

> ⚠️ **torch (CPU):** `sentence-transformers` descarga `torch` solo. Para evitar la versión CUDA (~2 GB), instalar primero la variante CPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Instalación (desde cero)

```bash
git clone <url-del-repo>
cd verificador-de-veracidad-enfoque-eclesial

# 1. Entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .\venv\Scripts\Activate.ps1

# 2. Dependencias (ver aviso de torch arriba)
#    -e = modo editable: vincula `config` y `src` a la raíz del repo.
#    (Con `pip install .` a secas se copia config.py a site-packages y
#    queda una copia vieja que puede importarse por error desde otras rutas.)
pip install -e ".[dev]"

# 3. Smoke test: abre la colección ChromaDB y muestra cuántos docs tiene
python main.py
# Salida esperada si ya se indexó: un número (2965 con el corpus completo)
# Salida si está vacía: 0 (ejecutar `python -m src.cli setup`)
```

---

## Estructura del proyecto

```
verificador-de-veracidad-enfoque-eclesial/
├── config.py            # Modelos pydantic que leen config.toml + resolver_ruta()
├── config.toml          # TODA la configuración (rutas, modelos, modo ETL, IVR, expansión)
├── main.py              # Smoke test de la capa base_datos
├── verificador.py       # Entry point de consola (python -m verificador) → src.cli
├── pyproject.toml       # Dependencias + script de consola `verificador`
│
├── src/
│   ├── orquestador.py       # Capa lógica que une todas las piezas (URL [+ PDF] → IVR → RF → veredicto)
│   ├── cli/                 # Interfaz de terminal (compatible UTF-8 en Windows/Linux/macOS)
│   │   ├── __init__.py      # API pública (main)
│   │   ├── cli.py           # Subcomandos: evaluar, setup, dataset-generar, entrenar
│   │   └── __main__.py      # lanzador (python -m src.cli)
│   │
│   ├── base_datos/          # Capa RAG (ChromaDB + embeddings)
│   │   ├── embeddings.py    # Función de embedding (modelo SBERT desde config)
│   │   └── chromaDB.py      # Cliente, colección, normalizar item, indexar y consultar corpus
│   │
│   ├── corp_extractor/      # Pipeline ETL del corpus canónico
│   │   ├── main_etl.py      # Orquestador (raw → clasificado)
│   │   ├── extractores/     # 5 scrapers (cánones, compendio, constituciones,
│   │   │                    #   declaraciones, decretos) → generan los _v1.json
│   │   └── clasificador_corpus/
│   │       └── clasificador_metadatos_v1.py   # Inyecta categoria + nivel_autoridad graduado (J)
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
│   │   ├── tesauro_tools.py # Valida el tesauro y busca candidatos en el corpus (solo lectura)
│   │   └── tesauro.json     # Tesauro ontológico (61 conceptos, 6 categorías canónicas)
│   │
│   ├── vectorizacion/
│   │   └── ivr.py            # IVR: stemming Snowball, señales SB/L/SO/J + vector_caracteristicas()
│   │
│   ├── dataset/             # Generación de dataset con ventana deslizante (175 palabras, solape 50)
│   │   ├── __init__.py
│   │   ├── generador.py     # Lee iglesia.db, descarga URLs, aplica ventanas y calcula IVR
│   │   └── agregacion.py    # Agrega features de ventanas y asigna veredicto (proxy/umbral/débil)
│   │
│   ├── modelo/
│   │   └── rf.py             # Random Forest: entrena / predice el veredicto (capa de decisión)
│   │
│   └── web_extractor/       # Extracción de contenido de noticias (trafilatura)
│       ├── __init__.py       # API pública (extraer_contenido_url, extraer_titular, ...)
│       └── web_extractor.py  # URL/HTML → texto + metadatos, sin red en las pruebas
│
├── tests/                   # 180 pruebas unitarias y de regresión (pytest)
│   ├── test_ivr.py                    # pruebas del IVR y regresión de auditoría (stemmer, SO, NaN)
│   ├── test_orquestador.py            # pruebas del orquestador (con dobles, sin red/Chroma)
│   ├── test_expansion_dinamica.py     # 96 pruebas del módulo de expansión dinámica
│   ├── test_clasificador_metadatos.py # pruebas de categorías por libro, Vaticano II y escala J
│   ├── test_tesauro_tools.py          # pruebas del validador y buscador de candidatos del tesauro
│   ├── test_web_extractor.py          # pruebas de extracción web con HTML en memoria
│   └── ejemplos/                      # PDF de ejemplo (1 y 2 columnas) para probarlo a mano
│
└── data/               # ⚠️ IGNORADA por git — se regenera con el ETL (`verificador setup`)
   ├── corp_extractor/
   │   ├── raw/         #    BRONCE: corpus_*_v1.json (crudos)
   │   └── clasificado/ #    GOLD:   corpus_*_clasificado.json (para Chroma)
   └── chroma/          # Base vectorial persistente (chroma.sqlite3)
```

**Regla de oro:** el MISMO modelo de embeddings (`models.sb_activo`) alimenta la base vectorial, la expansión dinámica **y** el IVR. Nunca cambiar uno sin el otro.

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
max_results = 30

[corp_extractor]
modo_ejecucion = "dinamico"       # "estatico" (local) | "dinamico" (descarga web)

[expansion_dinamica]
max_mb = 20
max_paginas = 300
detectar_columnas = true
palabras_max_chunk = 500
palabras_min_chunk = 20
nivel_autoridad_default = 0.33
categoria_default = "Normativa Externa (carga dinámica)"

[ivr]
peso_sb = 0.40          # SB = similitud semántica (coseno SBERT)
peso_literal = 0.25     # L  = coincidencia literal (n-gramas de contenido con stemming)
peso_ontologico = 0.20  # SO = coincidencia ontológica simétrica (categoría + conceptos)
peso_autoridad = 0.15   # J  = peso de autoridad del documento (nivel_autoridad)
metrica_literal = "contencion"
n_gramas = 2
n_resultados = 5
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

### 1. ETL del corpus y Escala Graduada de Autoridad ($J$)

```bash
# Ejecuta todo el ETL (extractores + clasificador):
python src/corp_extractor/main_etl.py

# O ejecuta únicamente el clasificador sobre los _v1.json ya descargados:
python src/corp_extractor/clasificador_corpus/clasificador_metadatos_v1.py
```

El orquestador lee los 5 crudos de `raw_corpus_dir`, les inyecta `categoria` + `nivel_autoridad` con el clasificador (`clasificador_metadatos_v1.py`) y escribe los `_clasificado.json` en `corpus_dir`.

- **`modo_ejecucion = "estatico"`**: no usa internet, toma los `_v1.json` ya descargados.
- **`modo_ejecucion = "dinamico"`**: descarga desde `vatican.va` y regenera los crudos.

#### Escala graduada de niveles de autoridad ($J$)
Para que la señal jerárquica $J$ discrimine efectivamente en el IVR según el rango canónico y magisterial de cada fuente, el clasificador asigna **5 niveles graduados**:

| Fuente / Tipo de Corpus | Nivel de Autoridad ($J$) | Documentos incluidos | Fragmentos |
|---|:---:|---|:---:|
| **Código de Derecho Canónico** (`corpus_canones`) | **1.00** | Libros I al VII (Cann. 1–1752) | 1.752 |
| **Constituciones — Concilio Vaticano II** (`corpus_constituciones`) | **0.85** | *Lumen gentium*, *Dei Verbum*, *Sacrosanctum concilium*, *Gaudium et spes* | 326 |
| **Decretos — Concilio Vaticano II** (`corpus_decretos`) | **0.70** | *Christus Dominus*, *Presbyterorum Ordinis*, *Optatam Totius*, *Perfectae Caritatis*, *Apostolicam Actuositatem*, *Orientalium Ecclesiarum*, *Unitatis redintegratio*, *Ad gentes*, *Inter Mirifica* | 270 |
| **Declaraciones — Concilio Vaticano II** (`corpus_declaraciones`) | **0.70** | *Dignitatis humanae*, *Nostra aetate*, *Gravissimum Educationis* | 34 |
| **Compendio de la Doctrina Social** (`corpus_compendio`) | **0.55** | Numerales 1–583 | 583 |
| **Normativa Externa / Expansión Dinámica** (`expansion_dinamica`) | **0.33** *(default)* | Decretos diocesanos o documentos PDF cargados en sesión | Transitorio |

Verificar la distribución de niveles de autoridad en ChromaDB:
```bash
python -c "from src.base_datos.chromaDB import get_coleccion; from collections import Counter; c = get_coleccion(con_embedding=False, crear=False); print(Counter(m.get('nivel_autoridad') for m in c.get(include=['metadatas'])['metadatas']))"
# Salida esperada: Counter({1.0: 1752, 0.55: 583, 0.85: 326, 0.7: 304})
```

### 2. Indexar el corpus en ChromaDB (RAG)

```bash
python -m src.cli setup              # ETL completo + reindexar_corpus() + verificación
# o consultar directamente una premisa de prueba:
python -m src.base_datos.chromaDB
```

Lee cada `*_clasificado.json` de `corpus_dir`, normaliza items a `(ids, documents, metadatas)` y hace `upsert` por lotes en la colección única `corpus_canonico` (**2.965 documentos** en total).

**Detalles de implementación importantes:**

- **ID de Chroma** = `f"{tipo_corpus}:{indice:04d}"` — posición dentro del archivo (único y determinístico). El número real del canon/numeral va en metadata como `id_norma` (NO es único en algunos corpus).
- **`canon_id` vs `numeral_id`:** el Código usa `canon_id`, los otros 4 corpus usan `numeral_id` (mutuamente excluyentes) → normalizar con `.get('numeral_id', .get('canon_id'))`.
- **`url_origen` es condicional:** el Compendio no la tiene en sus items (0/583) → solo guardarla si existe, nunca `None`.
- **Metadata solo escalares** (`str/int/float/bool`), sin `None`, sin listas/dicts anidados.
- **Saneo defensivo de similitud:** `consultar()` sanea distancias `NaN`/`inf` y acota `similitud = 1 - distancia` al rango `[0.0, 1.0]`.

### 3. Interfaz unificada (Orquestador CLI)

Todo el flujo (`URL [+ PDF] ──► web_extractor ──► premisa ──► IVR ──► RF ──► veredicto`) se controla desde UN comando. El CLI está en `src/cli/` y une las piezas desacopladas; la lógica de enlace vive en `src/orquestador.py` (capas separadas: el CLI solo parsea e imprime).

```bash
# Formas equivalentes de lanzarlo:
python -m src.cli --help
python -m verificador --help        # módulo raíz `verificador.py`
verificador --help                  # tras `pip install -e .` (entry point)
```

**Subcomandos:**

| Comando | Qué hace |
|---------|----------|
| `evaluar URL` | URL → texto/titular → premisa → RAG (corpus base + `--pdf` opcional) → IVR → RF → informe + veredicto |
| `setup` | ETL completo + indexa el corpus en Chroma y verifica sincronización (`2965` docs) |
| `dataset-generar` | Recorre noticias de `iglesia.db` con ventana deslizante (`175` palabras, solape `50`) y genera el CSV de entrenamiento (`16` características + `veredicto`) |
| `entrenar` | Entrena y persiste el Random Forest a `data/modelo_rf.joblib` |

**Ejemplo 1 (`evaluar`, modo interactivo con una URL real):**

```bash
python -m src.cli evaluar "https://www.vaticannews.va/es/papa/news/2026-10/papa-leon-vivamente-apenado-por-el-accidente-ferroviario.html"
```

Descargará la noticia con `trafilatura`, propondrá el titular como premisa y te pedirá confirmación antes de evaluar (`[Enter=usar, 'n'=escribir otra]`).

**Ejemplo 2 (`evaluar`, modo no interactivo / automatizado):**

```bash
python -m src.cli evaluar "https://www.vaticannews.va/es/papa/news/2026-10/papa-leon-vivamente-apenado-por-el-accidente-ferroviario.html" --no-interactivo
```

**Ejemplo 3 (`evaluar`, con premisa explícita y PDF de expansión dinámica):**

```bash
python -m src.cli evaluar "https://www.vaticannews.va/es/..." \
    --premisa "El parroco acompanara a los novios en el curso prematrimonial" \
    --pdf ruta/al/decreto.pdf \
    --no-interactivo
```

> 💡 **Nota:** Si todavía no se ha entrenado `data/modelo_rf.joblib` (con `verificador entrenar`), el comando `evaluar` **imprime igualmente todo el `INFORME DE VERACIDAD` del IVR** (URL, premisa, conceptos detectados, IVR máx/medio/mín y el Top 5 de evidencias con sus señales `SB`, `L`, `SO`, `J`) e indica al final cómo entrenar el modelo Random Forest.

### 4. Índice de Veracidad Relativa (IVR — `src/vectorizacion/ivr.py`)

Heurística que puntúa cuánto respalda (o contradice) cada norma recuperada por el RAG a una premisa periodística, combinando cuatro señales normalizadas en `[0, 1]`:

$$\text{IVR} = \frac{w_{SB}\cdot SB + w_L\cdot L + w_{SO}\cdot SO + w_J\cdot J}{\sum_{i \in \text{activas}} w_i}$$

- **$SB$ (Similitud Semántica, `peso_sb = 0.40`):** similitud coseno devuelta por el modelo SBERT.
- **$L$ (Coincidencia Literal con Stemming, `peso_literal = 0.25`):** aplica `snowballstemmer` (`spanish`) sobre las palabras de contenido (sin palabras vacías) para capturar flexiones morfológicas de género, número y conjugación (`necesario`/`necesaria`, `obispo`/`obispos`). En premisas cortas (`< 12` tokens de contenido) promedia `0.5 · contención_unigramas + 0.5 · contención_bigramas` (Opción 1C), evitando que incisos explicativos dentro de una norma anulen la coincidencia; en textos largos (`>= 12` tokens, como las ventanas móviles de 175 palabras) utiliza bigramas puros.
- **$SO$ (Coincidencia Ontológica Simétrica, `peso_ontologico = 0.20`):** indexa los **61 conceptos** (312 formas léxicas) de `tesauro.json` mediante sus raíces morfológicas (*stems* Snowball, verificado con **0 colisiones de stem** entre conceptos). Cuando la premisa activa al menos un concepto, combina:
  1. La fracción de conceptos de la premisa respaldados por la categoría de la norma o presentes en los conceptos de la norma (`frac_premisa`). Si la norma proviene de `expansion_dinamica` con categoría genérica, infiere sus categorías ontológicas a partir de los conceptos detectados en el propio fragmento del PDF.
  2. La especificidad conceptual de la norma (`|P ∩ N| / |N|`), evitando que normas extensas con múltiples conceptos ajenos o normas sin conceptos (`conceptos_norma == {}`) saturen a `1.0`.
  - Si la premisa no activa ningún concepto del tesauro (p. ej. una noticia secular ajena al ámbito eclesial), $SO$ es `None` y la fórmula renormaliza los pesos restantes (`so_disponible = 0.0`).
- **$J$ (Nivel de Autoridad, `peso_autoridad = 0.15`):** jerarquía normativa del documento (`1.00`, `0.85`, `0.70`, `0.55`, `0.33`).

#### Probar el IVR directamente desde consola

```bash
# Caso 1: Verdad doctrinal (CIC, canon 849 sube del puesto #3 del RAG al #1 con IVR=0.854 y L=0.750)
python -m src.vectorizacion.ivr "El bautismo es necesario para la salvacion"

# Caso 2: Plurales y verbos conjugados (detecta ID_BAUTISMO, ID_EUCARISTIA e ID_APOSTASIA gracias al stemmer)
python -m src.vectorizacion.ivr "Los fieles que se bautizaron y comulgaron no apostataron"

# Caso 3: Sigilo sacramental (cánones 986, 978 y 983 lideran el ranking; normas ajenas caen al fondo)
python -m src.vectorizacion.ivr "El sacerdote revelo el secreto de confesion de un fiel"

# Caso 4: Noticia fuera del ámbito eclesial (SO=n/a, IVR cae a ~0.430)
python -m src.vectorizacion.ivr "El banco central subio las tasas de interes en el mercado financiero"

# Caso 5: Combinando ChromaDB + un PDF externo cargado con Expansión Dinámica (--pdf)
python -m src.vectorizacion.ivr "El parroco acompanara a los novios en el curso prematrimonial" --pdf ruta/al/documento.pdf
```

### 5. Expansión dinámica (PDF externo, transitorio)

Permite que el usuario cargue **un PDF normativo propio** (p. ej. un decreto diocesano) durante la sesión de validación, y que la noticia se contraste **también** contra ese documento, **sin tocar el corpus base** (`corpus_canonico`). Todo vive en memoria y desaparece al cerrar la sesión.

> **Alcance:** solo PDF con **texto seleccionable**. No hace OCR: un PDF escaneado (imágenes) se rechaza con un mensaje claro.

#### Uso mínimo (y conexión con el IVR)

```python
from src.expansion_dinamica import SesionExpansion, ExpansionError
from src.vectorizacion import ivr

with SesionExpansion() as sesion:   # UNA por usuario (en Streamlit: guardarla en st.session_state)
    try:
        resumen = sesion.cargar_pdf(archivo, nivel_autoridad=0.33)   # ruta, bytes o archivo subido
    except ExpansionError as e:
        print(e)                    # mensaje listo para mostrar al usuario
    else:
        print(resumen.fragmentos, resumen.esquema, resumen.advertencias)

    # Opción A: recuperar las evidencias combinadas (corpus base + PDF cargado)
    evidencias = sesion.consultar_combinado("El obispo puede dispensar...", n_resultados=5)

    # Opción B: calcular directamente el informe IVR sobre el conjunto combinado
    informe = ivr.calcular_ivr(
        "El obispo puede dispensar...",
        consultar_fn=sesion.consultar_combinado,
    )
```

| Método | Qué hace |
|---|---|
| `cargar_pdf(archivo, nombre=None, *, nivel_autoridad=None, categoria=None, url_origen=None)` | Procesa e indexa un PDF. Devuelve un `ResumenDocumento` (`doc_id`, `nombre`, `paginas`, `fragmentos`, `esquema`, `nivel_autoridad`, `categoria`, `advertencias`, `ya_cargado`). Cargar el mismo PDF dos veces no lo duplica. |
| `consultar(premisa, n_resultados=5)` | Busca **solo** en los PDF cargados. |
| `consultar_combinado(premisa, n_resultados=5, where_base=None)` | Busca en el corpus base **y** en los PDF cargados. Sin PDF cargados devuelve exactamente lo que devolvería `chromaDB.consultar`. |
| `documentos()` / `eliminar_documento(doc_id)` / `cerrar()` | Gestión de lo cargado en la sesión. |

#### Qué devuelve cada resultado

Las mismas claves que `chromaDB.consultar()`, más cuatro extras. **El cálculo del IVR trata ambas fuentes por igual**: las similitudes son comparables (mismo modelo de embeddings y mismo espacio coseno), y si el PDF conserva la categoría por defecto `"Normativa Externa (carga dinámica)"`, `ivr.coincidencia_ontologica()` infiere automáticamente sus categorías temáticas a partir de los conceptos del tesauro presentes en el fragmento.

| Clave | Contenido |
|---|---|
| `id_chroma`, `texto`, `similitud`, `contexto`, `fuente`, `id_norma`, `tipo_corpus`, `categoria`, `nivel_autoridad`, `url_origen`, `cita` | Igual que el corpus base. En un PDF cargado: `tipo_corpus = "expansion_dinamica"`, `id_norma` = n.º de artículo/canon/numeral (0 si la unidad no está numerada), `cita` = p. ej. `Decreto 12/2024, Artículo 4, p. 2`. |
| `origen` | `"corpus_base"` o `"expansion_dinamica"`. Sirve para distinguir de dónde viene cada evidencia. |
| `doc_id`, `pagina_inicio`, `pagina_fin` | Solo en resultados de un PDF cargado. |

**`nivel_autoridad`** (Coeficiente Jerárquico del IVR): admite tanto los niveles clásicos (**0.33** Nivel 1 norma local — valor por defecto, **0.66** Nivel 2 y **1.0** Nivel 3) como los escalones graduados (**0.55**, **0.70**, **0.85**). Cualquier otro valor lanza `NivelAutoridadInvalidoError`.

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
| `NivelAutoridadInvalidoError` | `nivel_autoridad` no pertenece a `(0.33, 0.55, 0.66, 0.70, 0.85, 1.0)`. |

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
- Probado con PDF generados a partir del propio corpus (cánones y numerales: 98–100 % de unidades idénticas al original) y con PDF de ejemplo.

#### Probar

```bash
pytest tests/test_expansion_dinamica.py -v    # corre las 96 pruebas del módulo

# Solo cortar el documento y ver cómo lo segmenta (NO carga el modelo SBERT):
python -m src.expansion_dinamica tests/ejemplos/decreto_diocesano_ejemplo.pdf --solo-segmentar
python -m src.expansion_dinamica tests/ejemplos/decreto_dos_columnas_ejemplo.pdf --solo-segmentar

# Cargar y consultar de verdad (usa el modelo SBERT):
python -m src.expansion_dinamica ruta/al/documento.pdf --premisa "texto de la noticia"

# Evaluar el IVR combinando el Decreto Diocesano de prueba + ChromaDB:
python -m src.vectorizacion.ivr "El parroco acompanara a los novios en el curso prematrimonial" --pdf tests/ejemplos/decreto_diocesano_ejemplo.pdf
```

**Problemas frecuentes**

- `No module named 'chromadb'` (u otra librería): activar el entorno virtual e instalar con `pip install -e ".[dev]"`.
- `CERTIFICATE_VERIFY_FAILED` al cargar el modelo: lo causa un proxy o antivirus que intercepta la conexión. No es grave si el modelo ya está en la caché local; el programa sigue con él.
- La primera ejecución descarga el modelo SBERT (~470 MB).

### 6. Extracción de contenido web (noticias/artículos)

El módulo `src/web_extractor/` extrae el texto de una noticia a partir de su URL, usando `trafilatura`. Es el eslabón que convierte una **URL** en la **premisa** que evaluará el IVR (la fuente de la noticia para contrastar contra el corpus canónico).

Flujo:  `URL ──► web_extractor ──► premisa ──► IVR ──► RF ──► veredicto`

#### Probar la extracción con una URL real

```bash
# Extrae el texto principal de la noticia
python -m src.web_extractor "https://www.vaticannews.va/es/papa/news/2026-10/papa-leon-vivamente-apenado-por-el-accidente-ferroviario.html"

# Solo el titular (premisa candidata para el IVR)
python -m src.web_extractor "https://..." --titulo

# Texto + metadatos (titulo, autor, fecha, idioma)
python -m src.web_extractor "https://..." --metadata
```

#### API pública

| Función | Qué devuelve |
|---|---|
| `extraer_contenido_url(url)` | Texto principal de la noticia (str) o `None`. |
| `extraer_contenido_html(html)` | Igual, pero a partir de HTML ya descargado (estratégico para testear sin red). |
| `extraer_metadata_url(url)` | `{titulo, autor, fecha, idioma}` (vacíos = `""`, nunca `None`). |
| `extraer_metadata_html(html)` | Igual, desde HTML (sin red). |
| `extraer_titular(url)` | Titular de la noticia (premisa candidata) o `None`. |

```python
from src.web_extractor import extraer_contenido_url, extraer_titular

texto = extraer_contenido_url("https://...")
titulo = extraer_titular("https://...")
```

#### Uso y límites (`trafilatura`)

- **No hace OCR ni ejecuta JavaScript**: su alcance son páginas HTML con contenido accesible (artículos/noticias). Una página que requiere JS para renderizar no dará texto.
- Nunca lanza: ante URL inválida, error de red o página sin artículo devuelve `None` y lo registra en el log, para que el orquestador decida cómo mostrarlo.
- Para que el IVR funcione bien la premisa puntual puede derivarse del **titular** (en `evaluar`) o evaluarse por **ventanas móviles de 175 palabras con solape de 50** sobre el cuerpo completo de la noticia (en `dataset-generar`).

### 7. Random Forest (capa de decisión) y generación de dataset

El módulo `src/modelo/rf.py` clasifica el **veredicto final** (`consistente`, `inconsistente`, `no_verificable`) consumiendo el vector de características que produce el IVR (`ivr.vector_caracteristicas()`) — un vector plano de **16 floats**:

```python
NOMBRES_CARACTERISTICAS = [
    "n_evidencias", "n_conceptos_premisa",
    "ivr_max", "ivr_medio", "ivr_min", "ivr_brecha",
    "sb_max", "sb_medio", "sb_min", "sb_brecha",
    "literal_max", "literal_medio",
    "ontologico_max", "so_disponible",
    "terminos_compartidos_max", "autoridad_max",
]
```

Flujo:

```
premisa ──► ivr.calcular_ivr() ──► informe ──► vector_caracteristicas() ──► RF ──► veredicto
```

Generar el dataset desde `iglesia.db` y entrenar:

```bash
# 1. Diagnóstico rápido sobre una muestra balanceada de 30 noticias (sin escribir CSV)
python -m src.cli dataset-generar --db ruta/a/iglesia.db --n 30 --solo-diagnostico

# 2. Generar el CSV etiquetado
python -m src.cli dataset-generar --db ruta/a/iglesia.db --n 0 --salida data/etiquetado.csv

# 3. Entrenar y guardar el modelo Random Forest en disco
python -m src.cli entrenar --dataset data/etiquetado.csv --salida data/modelo_rf.joblib
# (o directamente: python -m src.modelo.rf --entrenar data/etiquetado.csv --salida data/modelo_rf.joblib)
```

Predecir desde consola (ya con el modelo guardado):

```bash
python -m src.modelo.rf --predecir "El bautismo esta prohibido"
```

Integración en Python:

```python
from src.vectorizacion import ivr
from src.modelo import rf      # módulo dentro de src/modelo/rf.py

informe = ivr.calcular_ivr(premisa)
modelo  = rf.cargar("data/modelo_rf.joblib")
resultado = rf.predecir(modelo["clf"], modelo["le"], rf._X_de_informe(informe))
# resultado = {"clase": "inconsistente", "probabilidad": 0.87, "probabilidades": {...}}
```

**Dos roles del RF en la tesis** (ver `config.IvrConfig` y el docstring de `ivr.py`):
1. **Calibrar los pesos del IVR** con la importancia de las variables del bosque (`.feature_importances_`); los pesos viven en `config.toml → [ivr]`.
2. **Clasificar el veredicto** según el vector del IVR y la confianza del modelo (`predict_proba`).

> La columna `so_disponible` permite al RF distinguir *no aplica* de *0 real* en la señal ontológica, y `terminos_compartidos_max` aporta la proporción exacta de conceptos compartidos entre la premisa y la mejor evidencia. Mantener el orden de `NOMBRES_CARACTERISTICAS` en sincronía con las claves de `vector_caracteristicas()`.

### 8. Tesauro: validar y buscar candidatos

Herramienta de **solo lectura** para mantener `src/procesamiento_nlp/tesauro.json` (61 conceptos en 6 categorías; nunca lo modifica automáticamente):

```bash
python -m src.procesamiento_nlp.tesauro_tools validar
python -m src.procesamiento_nlp.tesauro_tools candidatos --top 40 --salida candidatos.csv
```

- **`validar`** revisa el formato y avisa de lo que afecta a la recuperación: términos repetidos entre conceptos (*error*), términos canónicos que el corpus no usa (*aviso*, e indica cómo lo escribe el corpus), palabras sueltas muy comunes que pueden ser ambiguas y categorías que el clasificador del corpus no conoce. Devuelve código 1 si hay errores.
- **`candidatos`** lista frases frecuentes del corpus que todavía no están en el tesauro. Habrá ruido: se revisa a mano. El CSV se abre en Excel con las tildes bien.
- **Peso de autoridad:** es **solo el del documento** (lo asigna `clasificador_metadatos_v1.py`). Los conceptos del tesauro no necesitan `nivel_autoridad`; si existe se acepta, pero no se usa.
- Necesita el corpus crudo (`src/corp_extractor/extractores/data/corpus_*_v1.json`) en la máquina; sin él, `validar` solo revisa el formato.

---

## Suite de pruebas automatizadas (`pytest`)

El proyecto cuenta con **180 pruebas automatizadas** (y 41 subtests) que se ejecutan en ~5 segundos sin red ni necesidad de alterar la colección persistente de ChromaDB:

```bash
# Ejecutar toda la suite (180 passed)
pytest -v

# Ejecutar únicamente los tests de regresión de la auditoría del IVR
pytest tests/test_ivr.py -k "TestAuditoriaRegresion" -v

# Ejecutar los tests de Expansión Dinámica (96 tests)
pytest tests/test_expansion_dinamica.py -v

# Ejecutar los tests del clasificador de metadatos y escala de autoridad J
pytest tests/test_clasificador_metadatos.py -v
```

---

## Notas para el equipo

- **`data/` NO se commitea.** Está en `.gitignore`. Tras clonar, regenerar con `python -m src.cli setup` (modo estático necesita los `_v1.json`; si no los tenés, corré en modo `"dinamico"` para descargarlos).
- **`main.py`** es solo un smoke test (abre la colección y cuenta). No borrar.
- **Si instalás una librería nueva**, agregala a `dependencies` en `pyproject.toml`.
- **Si agregás una sección a `config.toml`**, actualizá `config.py` (ver arriba).
- Los artefactos `build/`, `*.egg-info/`, `.idea/` son locales y no se versionan.

## Estado actual (hoja de ruta)

| Fase | Estado |
|---|---|
| Config centralizada (`config.toml` / `pydantic`) | ✅ |
| Scrapers + ETL (`raw → clasificado`) + escala graduada de autoridad $J$ (`1.00 / 0.85 / 0.70 / 0.55`) | ✅ |
| Capa ChromaDB (`corpus_canonico`, 2.965 documentos indexados y verificados) | ✅ |
| Consulta semántica (RAG) con saneo defensivo de distancias | ✅ |
| Extracción de noticias (`web_extractor` con `trafilatura`) | ✅ módulo terminado + pruebas, sin red |
| Tesauro ontológico (61 conceptos, 0 colisiones de stem) + `tesauro_tools` | ✅ |
| IVR mejorado (stemmer Snowball en $L$ y tesauro, $SO$ simétrica, 16 features + tests de auditoría) | ✅ |
| Expansión dinámica (PDF externo en memoria, conectado al IVR y al CLI `--pdf`) | ✅ |
| Orquestador CLI (`evaluar`, `setup`, `dataset-generar`, `entrenar`) | ✅ |
| Random Forest + veredicto (`src/modelo/rf.py` + `src/dataset/`) | ⏳ código listo · falta correr `dataset-generar` + `entrenar` sobre `iglesia.db` |
