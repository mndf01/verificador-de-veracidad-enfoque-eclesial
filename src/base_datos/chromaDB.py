import json, logging, chromadb, chromadb.errors
from glob import glob
from importlib.metadata import metadata

from src.base_datos.embeddings import obtener_funcion_embedding
from config import obtener_configuraciones, resolver_ruta

condiciones_validas = {"fuente", "tipo_corpus", "id_norma", "contexto_jerarquico", "categoria", "nivel_autoridad", "url_origen"}
def _get_cliente():
    """
    Crea el cliente persistente con la carpeta configurada
    :return: PersistentClient
    """
    settings = obtener_configuraciones()
    ruta = resolver_ruta(settings.paths.chroma_dir)
    return chromadb.PersistentClient(path=str(ruta))


def get_coleccion(con_embedding=True, coleccion=None, crear=True):
    """
    Obtiene/crea la coleccion con la funcion de embedding y el espacio coseno
    Coleccion default `corpus_canonico`
    :returns: Collection or None
    """
    settings = obtener_configuraciones()
    cliente = _get_cliente()

    funcion = obtener_funcion_embedding() if con_embedding else None
    coleccion = settings.chroma.coleccion if coleccion is None else coleccion

    if crear:
        return cliente.get_or_create_collection(
            name=coleccion,
            embedding_function=funcion,
            metadata={"hnsw:space": settings.chroma.espacio_hnsw} # coseno
        )
    else:
        try:
            return cliente.get_collection(name=coleccion, embedding_function=funcion)
        except chromadb.errors.NotFoundError as e:
            logging.error("Coleccion no encontrada: %s", e)
            return None

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
    if coleccion is None:
        return []

    settings = obtener_configuraciones()
    ruta = resolver_ruta(settings.paths.corpus_dir) + "/*_clasificado.json"
    batch = settings.chroma.batch_size

    errores = []
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
            print("Indexando...")
            for i in range(0, len(ids), batch):
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
    coleccion = get_coleccion(con_embedding=False)

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



def _normalizar_cita(meta) -> str:
    """Cita legible de la norma, derivada de la metadata del chunk.

    - CIC y Compendio: no hay documento en el contexto.
    - Vaticano II: el documento es el 3er segmento del contexto; si
      id_norma == 0 la unidad es una sección sin numerar (PROEMIO, ...).
    """
    tipo = meta["tipo_corpus"]
    segmentos = meta["contexto_jerarquico"].split(" > ")

    if tipo == "corpus_canones":
        return f"CIC, canon {meta['id_norma']}"
    if tipo == "corpus_compendio":
        return f"Compendio, numeral {meta['id_norma']}"

    etiquetas = {
        "corpus_constituciones": "Constitución",
        "corpus_declaraciones": "Declaración",
        "corpus_decretos": "Decreto",
    }
    etiqueta = etiquetas.get(tipo, meta["fuente"])
    documento = segmentos[2] if len(segmentos) >= 3 else meta["fuente"]

    if meta["id_norma"] == 0:
        seccion = segmentos[3] if len(segmentos) >= 4 else ""
        return f"{etiqueta} {documento} (Concilio Vaticano II), {seccion}"
    return f"{etiqueta} {documento} (Concilio Vaticano II), numeral {meta['id_norma']}"


def consultar(premisa: str, n_resultados: int = 5, where: dict | None = None) -> list[dict]:
    """
    Consultar documentos indexados en una colección bajo ciertas condiciones, retornando resultados
    ordenados según la similitud con la premisa dada.

    El método realiza validaciones iniciales sobre los parámetros de entrada y, en caso de que las
    condiciones no se cumplan, devuelve una lista vacía. Si las condiciones son válidas, se intenta
    realizar una consulta a la colección especificada, considerando las condiciones adicionales
    proporcionadas en el parámetro where.

    :param premisa: Cadena de texto que representa la consulta o premisa para buscar documentos
        relacionados.
    :type premisa: str

    :param n_resultados: Número máximo de resultados que se desean obtener. Debe ser un valor
        entero mayor a cero. Por defecto, es 5.
    :type n_resultados: int

    :param where: Diccionario opcional que contiene condiciones adicionales para filtrar los
        resultados. Las claves válidas son: "fuente", "tipo_corpus", "id_norma", "contexto_jerarquico",
        "categoria", "nivel_autoridad", y "url_origen".
    :type where: dict | None

    :return: Una lista de diccionarios, donde cada diccionario contiene información estructurada de
        un documento relevante, incluyendo su similitud, texto, contexto, fuente y otros metadatos
        relacionados. Si no se encuentran documentos relevantes o ocurren errores, devuelve una lista
        vacía.
    :rtype: list[dict]
    """
    settings = obtener_configuraciones()
    coleccion = get_coleccion(coleccion=settings.chroma.coleccion, crear=False)


    if coleccion is None:
        return []
    elif not isinstance(premisa, str):
        logging.warning("La premisa debe ser un string")
        return []
    elif not premisa.strip():
        logging.warning("No se ingreso una premisa para la consulta")
        return []

    if where is not None:
        if not isinstance(where, dict):
            logging.warning("El parametro where debe ser un diccionario")
            return []
        elif not all(k in condiciones_validas for k in where.keys()):
            logging.warning("El parametro where contiene keys invalidas")
            return []

    try:
        n_resultados = int(n_resultados)
    except ValueError:
        logging.error("El número de resultados debe ser un entero")
        return []

    if n_resultados <= 0:
        logging.warning("El número de resultados debe ser mayor a cero")
        return []

    count = coleccion.count()
    if count == 0:
        logging.warning("La coleccion no tiene documentos indexados")
        return []

    try:
        result = coleccion.query(query_texts=[premisa], n_results=min(n_resultados, count, settings.chroma.max_results),
                                 where=where, include=["metadatas", "documents", "distances"])
    except chromadb.errors.ChromaError as e:
        logging.error("Error al consultar la coleccion de ChromaDB: %s", str(e))
        return []


    ids       = result["ids"][0]
    metadatas = result["metadatas"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]

    contratos = []
    for id_chroma, meta, document, distance in zip(ids, metadatas, documents, distances):
        contratos.append({
            "id_chroma": id_chroma,
            "texto": document,
            "similitud": max(1 - distance, 0),
            "contexto": meta["contexto_jerarquico"],
            "fuente": meta["fuente"],
            "id_norma": meta["id_norma"],
            "tipo_corpus": meta["tipo_corpus"],
            "categoria": meta["categoria"],
            "nivel_autoridad": meta["nivel_autoridad"],
            "url_origen": meta.get("url_origen") or "Sin url registrada",
            "cita": _normalizar_cita(meta)
        })

    return contratos


if __name__ == "__main__":
    contratos = consultar(premisa="El bautismo esta prohibido")
    for item in contratos:
        for key, value in item.items():
            print(f"{key}: {value}")
        print()
        print()