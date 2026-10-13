from pymilvus import MilvusClient, DataType
from typing import List
from core.embedder import embed_many
from core.splitter import Chunk
import config

URL = "http://localhost:19530"
COLLECTION = "rag_kb"
DIM = 1024

def ensure_collection():
    client = MilvusClient(uri="http://localhost:19530")
    if client.has_collection(COLLECTION):
        return client
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    # 按你入库代码里实际写入的字段补，没有就删掉这几行
    schema.add_field("parent_id", DataType.VARCHAR,max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",          # 必须等于你 schema 里向量字段的名字
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},   # ← 必须是整数，不能写成 "200"
    )
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    return client

def ingest_chunks(chunks:List[Chunk]):
    # 八字快列表写入milvus
    client = ensure_collection()
    # 批量向量化
    texts = [c.text for c in chunks]
    vectors = embed_many(texts)
    # 组装入库记录
    rows = [
        {
            "parent_id": c.parent_id,
            "text": c.text,
            "vector": v,
        }
        for c, v in zip(chunks, vectors)
    ]
    client.insert(collection_name=COLLECTION, data=rows)
    print(f"已入库{len(rows)}条子块")
    return client