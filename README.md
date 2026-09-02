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
│   ├── procesamiento_nlp/
│   │   └── tesauro.json     # Tesauro de categorías (insumo del IVR)
│   │
│   └── web_extractor/       # Extracción de contenido de noticias (pendiente)
│       └── web_extractor.py # Boceto con trafilatura
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
