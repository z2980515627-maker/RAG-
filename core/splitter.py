from dataclasses import dataclass, field
from typing import List

@dataclass
class Chunk:
    text: str
    chunk_id:str = ""
    parent_id: str = ""
"""
    parent_size=800：父块约 800 字，足够大模型理解上下文。
    child_size=200：子块约 200 字，检索时更精准。
    overlap=50：块与块之间重叠 50 字，防止关键信息被切到边界上丢失。
"""
def split_parent_child(
        text:str,
        parent_size:int = 800,
        child_size:int = 200,
        overlap:int = 50
)-> tuple[List[Chunk], List[Chunk]]:
    """
    父子分块：
    - 父块/:大块给模型用
    - 子块: 小块，做向量检索用
    """
    # 先按段落切，避免从句子中间硬切
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    parents:List[Chunk] = []
    children:List[Chunk] = []

    # 生成父块
    buf = ""
    for para in paragraphs:
        buf += "\n"+para if buf else para
        if len(buf)>=parent_size:
            parents.append(Chunk(text=buf))
            buf = buf[-overlap:] if overlap else ""
        if buf:
            parents.append(Chunk(text=buf))

    # 给每个父块生成子块
    for i, parent in enumerate(parents):
        pid = f"parent{i}"
        parent.chunk_id = pid
        p_text = parent.text
        start = 0
        j = 0
        while start<len(p_text):
            end = start + child_size
            chunk_text = p_text[start:end]
            cid = f"chunk_{i}_{j}"
            children.append(Chunk(text=chunk_text,chunk_id=cid,parent_id=pid))
            start = end - overlap
            j+=1

    return parents, children