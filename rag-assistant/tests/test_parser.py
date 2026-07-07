from pathlib import Path

import docx
import fitz

from app.services import parser


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

    result = parser.parse_pdf(str(pdf_path))

    assert len(result) == len(page_texts)
    assert [page["page_number"] for page in result] == [1, 2]
    assert [page["text"] for page in result] == page_texts


def test_parse_docx_extracts_paragraph_text(tmp_path):
    docx_path = tmp_path / "sample.docx"
    paragraph_texts = [
        "DOCX paragraph one: alpha beta gamma.",
        "DOCX paragraph two: delta epsilon zeta.",
        "DOCX paragraph three: eta theta iota.",
    ]

    document = docx.Document()
    for text in paragraph_texts:
        document.add_paragraph(text)
    document.save(docx_path)

    result = parser.parse_docx(str(docx_path))

    assert isinstance(result, str)
    for text in paragraph_texts:
        assert text in result


def test_parse_txt_strips_bom(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Xin chào", encoding="utf-8-sig")

    result = parser.parse_txt(str(txt_path))

    assert "\ufeff" not in result
    assert result == "Xin chào"
