from config import obtener_configuraciones
from chromadb.utils import embedding_functions

def obtener_funcion_embedding():
    """
    Devuelve un objeto que Chroma utiliza internamente para convertir de texto a vector
    :return: SentenceTransformerEmbeddingFunction
    """
    settings = obtener_configuraciones()
    modelo = settings.models.sb_activo

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=modelo,
        normalize_embeddings=True
    )