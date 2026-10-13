import os
from fastapi import FastAPI,HTTPException,UploadFile, File
from pydantic import BaseModel,Field
from core.retriever import search,insert_texts
from core.parser import parse_any
from core.splitter import split_parent_child
from core.generator import generate
from fastapi.responses import StreamingResponse

app = FastAPI(title="Milvus API",)

# 1. 服务启动时，预加载父块（只跑一次，提高后续接口速度）
TEXT_PATH="docs/test.txt"
text = parse_any(TEXT_PATH)
parents,children = split_parent_child(text)

# 构建 {parent_id: text} 的映射字典，方便检索后回溯
parents_map = {p.chunk_id:p.text for p in parents}

# 2. 定义接口的请求和响应模型（Pydantic）
class AskRequest(BaseModel):
    query: str = Field(...,description="用户问题")
    top_k: int = Field(3,ge=1,le=10,description="检索召回条数，默认3")

class AskResponse(BaseModel):
    query:str
    answer:str
    contexts:list[str]

# 3. 定义核心 /ask 接口
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    hits = search(req.query, top_k=req.top_k)

    seen = set()
    contexts = []
    for h in hits:
        pid = h["entity"]["parent_id"]
        if pid not in seen:
            seen.add(pid)
            full = parents_map.get(pid)
            if full:
                contexts.append(full)

    if not contexts:
        raise HTTPException(status_code=404, detail="未找到相关参考资料")

    answer = generate(req.query, contexts)

    return AskResponse(query=req.query, answer=answer, contexts=contexts)

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    hits = search(req.query, top_k=req.top_k)

    seen = set()
    contexts = []
    for h in hits:
        pid = h["entity"]["parent_id"]
        if pid not in seen:
            seen.add(pid)
            full = parents_map.get(pid)
            if full:
                contexts.append(full)

    if not contexts:
        raise HTTPException(status_code=404, detail="未找到相关参考资料")

    stream_resp = generate(req.query, contexts, stream=True)

    def event_generator():
        for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    new_map = insert_texts([text])
    parents_map.update(new_map)
    return {"status":"ok","chunks_added":len(new_map)}


@app.get("/health")
async def health():
    return {"status":"ok"}