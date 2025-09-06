"""
Leitor de PDF em texto.
Tenta primeiro com pdfplumber; se falhar, usa PyMuPDF (fitz).
Se o PDF for imagem sem texto, será necessário OCR.
"""

import re
from typing import Optional
import pdfplumber
import fitz  # PyMuPDF


def _clean(txt: str) -> str:
    """Limpeza básica do texto extraído."""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)            # espaços e tabs
    txt = re.sub(r"\r\n?", "\n", txt)            # normaliza quebras
    txt = re.sub(r"\n{3,}", "\n\n", txt)         # máximo de 2 seguidas
    return txt.strip()


def extract_with_pdfplumber(path: str) -> Optional[str]:
    """Tenta extrair usando pdfplumber."""
    try:
        out = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        txt = "\n".join(out)
        if txt and len(txt.strip()) > 20 and re.search(r"\w", txt):
            return _clean(txt)
    except Exception:
        pass
    return None


def extract_with_pymupdf(path: str) -> Optional[str]:
    """Tenta extrair usando PyMuPDF (fitz)."""
    try:
        doc = fitz.open(path)
        out = []
        for page in doc:
            out.append(page.get_text("text"))
        txt = "\n".join(out)
        if txt and len(txt.strip()) > 20 and re.search(r"\w", txt):
            return _clean(txt)
    except Exception:
        pass
    return None


def extract_text(path: str) -> str:
    """
    Extrai texto de um PDF.
    Levanta erro se não for possível extrair texto legível.
    """
    for extractor in (extract_with_pdfplumber, extract_with_pymupdf):
        txt = extractor(path)
        if txt:
            return txt
    raise RuntimeError(
        "Falha ao extrair texto: nenhum método (pdfplumber ou PyMuPDF) conseguiu ler o PDF."
        " Se for PDF de imagem, será necessário OCR."
    )
