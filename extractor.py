# === BEGIN FILE: extractor.py
"""
Leitor de PDF em texto.
Tenta primeiro com pdfplumber; se falhar, usa PyMuPDF (fitz).
Se o PDF for imagem sem texto, será necessário OCR (não incluso por padrão).
"""

import re
from typing import Optional


def _clean(txt: str) -> str:
    """Limpeza básica do texto extraído."""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def extract_with_pdfplumber(path: str) -> Optional[str]:
    """Tenta extrair usando pdfplumber."""
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        txt = "\n".join(out)
        return _clean(txt)
    except Exception:
        return None


def extract_with_pymupdf(path: str) -> Optional[str]:
    """Tenta extrair usando PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        out = []
        for page in doc:
            out.append(page.get_text("text"))
        txt = "\n".join(out)
        return _clean(txt)
    except Exception:
        return None


def extract_text(path: str) -> str:
    """
    Extrai texto de um PDF.
    Levanta erro se não for possível extrair texto legível.
    """
    for fn in (extract_with_pdfplumber, extract_with_pymupdf):
        txt = fn(path)
        if txt and len(txt) > 20:
            return txt
    raise RuntimeError(
        "Falha ao extrair texto. Verifique se o PDF não é imagem ou use OCR."
    )
# === END FILE
