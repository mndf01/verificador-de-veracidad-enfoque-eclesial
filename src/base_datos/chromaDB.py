import json, logging, chromadb, chromadb.errors
from glob import glob

from src.base_datos.embeddings import obtener_funcion_embedding
from config import obtener_configuraciones, resolver_ruta


def _get_cliente():
    """
    Crea el cliente persistente con la carpeta configurada
    :return: PersistentClient
    """
    settings = obtener_configuraciones()
    ruta = resolver_ruta(settings.paths.chroma_dir)
    return chromadb.PersistentClient(path=str(ruta))


def get_coleccion(con_embedding=True, coleccion=None):
    """
    Obtiene/crea la coleccion con la funcion de embedding y el espacio coseno
    Coleccion default `corpus_canonico`
    """
    settings = obtener_configuraciones()
    cliente = _get_cliente()

    funcion = obtener_funcion_embedding() if con_embedding else None
    coleccion = settings.chroma.coleccion if coleccion is None else coleccion
    return cliente.get_or_create_collection(
        name=coleccion,
        embedding_function=funcion,
        metadata={"hnsw:space": settings.chroma.espacio_hnsw} # coseno
    )

def _normalizar_item(datos, tipo_corpus, fuente):
    """
    Normaliza los datos de un corpus para su indexacion en ChromaDB.

    :param datos:
    :param tipo_corpus:
    :param fuente:
    :return: ids[], docs[], meta[]
    """
    ids, docs, meta = [], [], []
    for indice, item in enumerate(datos):
        ids.append(f"{tipo_corpus}:{indice:04d}")
        docs.append(item['texto'])

        id_norma = item.get('numeral_id', item.get('canon_id'))
        m = {
            'fuente': fuente,
            'tipo_corpus': tipo_corpus,
            'id_norma': id_norma,
            'contexto_jerarquico': item['contexto_jerarquico'],
            'categoria': item['categoria'],
            'nivel_autoridad': item['nivel_autoridad']
        }
        if item.get('url_origen'):
            m['url_origen'] = item['url_origen']

        meta.append(m)

    return ids, docs, meta

def reindexar_corpus():
    """Borra la coleccion actual y la recrea desde cero"""
    settings = obtener_configuraciones()
    cliente = _get_cliente()
    try:
        cliente.delete_collection(settings.chroma.coleccion)
    except chromadb.errors.NotFoundError:
        pass
    return indexar_corpus()

def indexar_corpus():
    """
    Indexa los archivos .json clasificados de corpus_dir en ChormaDB.
    Inserta los archivos en lotes de batch_size
    :return: lista de archivos que no pudieron ser indexados
    """
    coleccion = get_coleccion()
    settings = obtener_configuraciones()
    ruta = resolver_ruta(settings.paths.corpus_dir) + "/*_clasificado.json"
    batch = settings.chroma.batch_size

    ids_t, docs_t, meta_t, errores = [], [], [], []
    for archivo in glob(ruta):
        try:
            print("\n\nArchivo: ", archivo)
            with open(archivo, 'r', encoding='utf-8') as f:
                print("Leyendo...")
                data = json.load(f)
            tipo_corpus = data['tipo_corpus']

            fuente = data['fuente']

            print("Normalizando...")
            ids, docs, meta = _normalizar_item(data['datos'], tipo_corpus, fuente)

            print(f"Datos leidos, ids: {len(ids)}, docs: {len(docs)}, meta: {len(meta)}")
            ids_t.extend(ids)
            docs_t.extend(docs)
            meta_t.extend(meta)

            print("Indexando...")
            for i in range(0, len(ids_t), batch):
                coleccion.upsert(
                    ids=ids[i:i + batch],
                    documents=docs[i:i + batch],
                    metadatas=meta[i:i + batch]
                )

            print("Archivo indexado: ", archivo)
        except FileNotFoundError as e:
            logging.exception("El archivo %s no se pudo abrir: %s", archivo, e)
            errores.append(archivo)
        except PermissionError as e:
            logging.exception("El archivo %s no tiene permisos de lectura: %s", archivo, e)
            errores.append(archivo)
        except (OSError, KeyError, json.JSONDecodeError) as e:
            logging.exception("No se pudo indexar %s: %s", archivo, e)
            errores.append(archivo)

    return errores


def verificar_corpus_indexado() -> dict:
    """
    Compara los *_clasificado.json (esperado) contra la colección Chroma (real).
    No carga el modelo de embeddings (solo lee ids y metadata).
    :return: dict de diagnóstico con conteos por tipo y diferencias de ids
    """

    settings = obtener_configuraciones()
    carpeta_json = resolver_ruta(settings.paths.corpus_dir)

    # 1 -- Obtiene la cantidad de IDs esperado en total y por corpus de los archivos JSON
    ids_esperados = set()
    conteo_esperado_por_corpus = {}
    for archivo in glob(str(carpeta_json + "/*_clasificado.json")):
        with open(archivo, encoding="utf-8") as f:
            data = json.load(f)

        tipo_corpus = data["tipo_corpus"]
        ids, _, _ = _normalizar_item(data["datos"], tipo_corpus, data['fuente'])
        ids_esperados.update(ids)
        conteo_esperado_por_corpus[tipo_corpus] = conteo_esperado_por_corpus.get(tipo_corpus, 0) + len(ids)

    # 2 -- Obtiene la cantidad de IDs en la BD por corpus
    try:
        coleccion = get_coleccion(con_embedding=False)
    except chromadb.errors.NotFoundError:
        return {"ok": False, "total_bd": 0, "total_esperado": len(ids_esperados), "por_tipo": {}, "ids_faltantes": sorted(ids_esperados), "ids_huerfanos": [], "error": "coleccion no existe"}

    ids_bd = set(coleccion.get(include=[])["ids"])
    por_tipo = {}

    # 3 -- Comparacion por tipo
    for corpus, esperado in conteo_esperado_por_corpus.items():
        ids = set(coleccion.get(where={"tipo_corpus": corpus}, include=[])["ids"])
        por_tipo[corpus] = {
            "esperado": esperado,
            "en_bd": len(ids),
            "ok": esperado == len(ids)
        }

    faltantes_t = sorted(ids_esperados - ids_bd)
    huerfanos_t = sorted(ids_bd - ids_esperados)
    return {"ok": not faltantes_t and not huerfanos_t, "total_bd": len(ids_bd), "total_esperado": len(ids_esperados), "por_tipo": por_tipo, "ids_faltantes": faltantes_t, "ids_huerfanos": huerfanos_t}

if __name__ == "__main__":
    indexar_corpus()