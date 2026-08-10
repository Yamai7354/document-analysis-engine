from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG

_DEFAULT_MODEL = CONFIG.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")


def get_embedder(model_name: str = _DEFAULT_MODEL):
    """
    Returns a HuggingFace embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
