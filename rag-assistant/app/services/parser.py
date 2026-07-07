import docx
import fitz

def parse_txt(file_path: str) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    return ""


def parse_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip() != ""
    ]
    return "\n\n".join(paragraphs)


def parse_pdf(file_path: str) -> list[dict]:
    document = fitz.open(file_path)
    try:
        return [
            {"page_number": index + 1, "text": page.get_text().strip()}
            for index, page in enumerate(document)
        ]
    finally:
        document.close()
