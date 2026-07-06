from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_text(text: str) -> list[float]:
    embedding = _model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
