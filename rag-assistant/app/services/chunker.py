import re


def _hard_split(text: str, max_chunk_size: int) -> list[str]:
    if max_chunk_size <= 0:
        return [text] if text else []

    return [text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size) if text[i : i + max_chunk_size]]


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def _merge_small_parts(parts: list[str], max_chunk_size: int) -> list[str]:
    merged: list[str] = []
    current = ""

    for part in parts:
        if not current:
            current = part
            continue

        candidate = f"{current} {part}"
        if len(candidate) <= max_chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = part

    if current:
        merged.append(current)

    return merged


def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    if not text.strip():
        return []

    paragraph_chunks: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            paragraph_chunks.append(paragraph)
            continue

        sentences = _split_sentences(paragraph)
        if not sentences:
            paragraph_chunks.extend(_hard_split(paragraph, max_chunk_size))
            continue

        sentence_chunks = _merge_small_parts(sentences, max_chunk_size)
        for sentence_chunk in sentence_chunks:
            if len(sentence_chunk) > max_chunk_size:
                paragraph_chunks.extend(_hard_split(sentence_chunk, max_chunk_size))
            else:
                paragraph_chunks.append(sentence_chunk)

    chunks: list[dict] = []
    previous_text = ""

    for index, chunk_text_value in enumerate(paragraph_chunks):
        if overlap > 0 and previous_text:
            prefix = previous_text[-overlap:]
            combined_text = f"{prefix} {chunk_text_value}"
        else:
            combined_text = chunk_text_value

        chunks.append({"text": combined_text, "chunk_index": index})
        previous_text = chunk_text_value

    return chunks
