import src.base_datos.chromaDB as cdb

col = cdb.get_coleccion(no_embedding=True)
resultado = col.query(query_texts=["¿Cuándo puede un obispo dispensar un matrimonio?"], n_results=3, where={"tipo_corpus": "corpus_canones"})

