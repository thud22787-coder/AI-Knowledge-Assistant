from app.models import Chunk

_reranker = None


def get_reranker():
    from sentence_transformers import CrossEncoder

    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")

    return _reranker


def rerank_chunks(query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    if not chunks:
        return []

    pairs = [(query, chunk.text) for chunk in chunks]
    scores = get_reranker().predict(pairs)

    ranked_chunks = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in ranked_chunks[:top_k]]
