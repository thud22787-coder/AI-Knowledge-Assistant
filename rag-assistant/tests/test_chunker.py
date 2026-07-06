from app.services.chunker import chunk_text


def test_chunk_short_text_returns_single_chunk():
    text = "Short text."

    chunks = chunk_text(text, max_chunk_size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_index"] == 0


def test_chunk_respects_paragraph_boundaries():
    text = "Paragraph one is short.\n\nParagraph two is also short."

    chunks = chunk_text(text, max_chunk_size=500, overlap=50)

    assert len(chunks) == 2
    assert chunks[0]["text"].endswith("Paragraph one is short.")
    assert chunks[1]["text"].endswith("Paragraph two is also short.")


def test_chunk_long_paragraph_splits_by_sentence():
    text = (
        "Sentence one is here and it is long enough to matter. "
        "Sentence two continues the paragraph with more detail. "
        "Sentence three finishes it."
    )

    chunks = chunk_text(text, max_chunk_size=80, overlap=0)

    assert len(chunks) >= 2
    assert all(" " in chunk["text"] for chunk in chunks)
    assert all(not chunk["text"].endswith(" ") for chunk in chunks)
    assert all(not chunk["text"].startswith(" ") for chunk in chunks)


def test_chunk_has_overlap():
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )

    chunks = chunk_text(text, max_chunk_size=70, overlap=15)

    assert len(chunks) >= 2
    assert chunks[1]["text"].startswith(chunks[0]["text"][-15:])


def test_chunk_oversized_single_sentence_gets_truncated():
    text = "a" * 300

    chunks = chunk_text(text, max_chunk_size=100, overlap=0)

    assert chunks
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)
