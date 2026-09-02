import json, logging, chromadb
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


def get_coleccion():
    """
    Obtiene/crea la coleccion con la funcion de embedding y el espacio coseno
    """
    settings = obtener_configuraciones()
    cliente = _get_cliente()

    return cliente.get_or_create_collection(
        name=settings.chroma.coleccion,
        embedding_function=obtener_funcion_embedding(),
        metadata={"hnsw:space": settings.chroma.espacio_hnsw} # coseno
    )

def _normalizar_item(datos, tipo_corpus, fuente):
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
                    ids=ids_t[i:i + batch],
                    documents=docs_t[i:i + batch],
                    metadatas=meta_t[i:i + batch]
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


if __name__ == "__main__":
    indexar_corpus()