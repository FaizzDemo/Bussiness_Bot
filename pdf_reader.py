from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    reader = PdfReader(file_obj)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "")
    lower_name = name.lower()

    if lower_name.endswith(".pdf"):
        file_obj = uploaded_file
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return extract_text_from_pdf(file_obj)

    data = uploaded_file.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)
