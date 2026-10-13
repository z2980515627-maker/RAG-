from core.parser import parse_any
from core.splitter import split_parent_child
from core.store import ingest_chunks
from core.retriever import search, COLLECTION
from core.generator import generate

# 1. 解析
text = parse_any("docs/test.txt")
parents, children = split_parent_child(text)
print(f"父块：{len(parents)}，子块：{len(children)}")

# 2. 入库
ingest_chunks(children)

# 3. 检索
query = "Milvus的存储架构是什么"
hits = search(query, top_k=3)
print(f"\n查询：{query}")

# 4. 回溯父块（parents 是列表，先转字典）
parents_map = {p.chunk_id: p.text for p in parents}

seen = set()
contexts = []
print("=====================")
print(hits)
print("parents_map 的 keys =", list(parents_map.keys()))
print("parents[0] =", parents[0])
for h in hits:
    pid = h["entity"]["parent_id"]
    if pid not in seen:
        seen.add(pid)
        contexts.append(parents_map[pid])

# 5. 生成
answer = generate(query, contexts)
print("===RAG===回答")
print(answer)