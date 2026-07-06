def parse_txt(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    return ""
