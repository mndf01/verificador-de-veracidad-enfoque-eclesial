import chromadb

from src.base_datos.embeddings import obtener_funcion_embedding
from config import obtener_configuraciones

def get_cliente():
    settings = obtener_configuraciones()
    ruta = settings.paths.chroma_dir
    return chromadb.PersistentClient(path=str(ruta))

def get_coleccion():
    settings = obtener_configuraciones()
    cliente = get_cliente()

    return cliente.get_or_create_collection(
        name=settings.chroma.coleccion,
        embedding_function=obtener_funcion_embedding(),
        metadata={"hnsw:space": settings.chroma.espacio_hnsw} # coseno
    )
