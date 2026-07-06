from pathlib import Path

import fitz

from app.services.parser import parse_pdf


def test_parse_pdf_extracts_text_with_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    page_texts = [
        "Page one: the quick brown fox jumps over the lazy dog.",
        "Page two: pack my box with five dozen liquor jugs.",
    ]

    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(pdf_path)
    finally:
        document.close()

    result = parse_pdf(str(pdf_path))

    assert len(result) == len(page_texts)
    assert [page["page_number"] for page in result] == [1, 2]
    assert [page["text"] for page in result] == page_texts
