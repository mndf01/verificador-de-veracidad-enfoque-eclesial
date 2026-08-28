from config import obtener_configuraciones
from chromadb.utils import embedding_functions

def obtener_funcion_embedding():
    settings = obtener_configuraciones()
    modelo = settings.models.sb_activo

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=modelo,
        normalize_embeddings=True
    )