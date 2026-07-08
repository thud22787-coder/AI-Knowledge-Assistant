import hashlib

from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_embedding_cache: dict[str, list[float]] = {}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    uncached_texts: list[str] = []
    uncached_hashes: list[str] = []
    seen_uncached_hashes: set[str] = set()

    for text in texts:
        text_hash = _hash_text(text)
        if text_hash not in _embedding_cache and text_hash not in seen_uncached_hashes:
            uncached_texts.append(text)
            uncached_hashes.append(text_hash)
            seen_uncached_hashes.add(text_hash)

    if uncached_texts:
        embeddings = _model.encode(uncached_texts, normalize_embeddings=True)
        for text_hash, embedding in zip(uncached_hashes, embeddings.tolist()):
            _embedding_cache[text_hash] = embedding

    return [_embedding_cache[_hash_text(text)] for text in texts]
