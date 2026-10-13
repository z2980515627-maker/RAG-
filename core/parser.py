from pathlib import Path
from docx import Document
from PyPDF2 import PdfFileReader

def parse_docx(path:str)->str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def parse_pdf(path:str)->str:
    reader = PdfFileReader(path)
    return "\n".join(page.extractText()or "" for page in reader.pages)

def parse_txt(path:str)->str:
    return Path(path).read_text(encoding="utf-8")

def parse_any(path:str)->str:
        # 根据后缀自动选择解析器
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return parse_docx(path)
    elif ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".txt":
        return parse_txt(path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")