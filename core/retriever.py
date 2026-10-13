from pymilvus import MilvusClient
from core.embedder import embed_one
from core.splitter import split_parent_child
import config
from core.embedder import embed_many
URL = "http://localhost:19530"
COLLECTION = "rag_kb"

def search(query:str,top_k:int=3)->list[dict]:
    """
        检索 top——k 个最相关子块，返回chunkid/parentid/text/距离
    """
    client = MilvusClient(URL)
    client.load_collection(COLLECTION)
    vec = embed_one(query)
    results = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=top_k,
        output_fields=["id","parent_id","text"],
        consistency_level="Strong",
    )
    return results[0]


def insert_texts(texts: list[str], collection_name: str = "default"):
    """把文本列表切块、生成向量、写入 Milvus"""

    # 1. 连接 Milvus
    client = MilvusClient(uri="http://localhost:19530")

    # 2. 如果集合不存在，则创建（注意：dimension=768 要和你的 embedding 模型维度对齐）
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=768,
            auto_id=True,
            enable_dynamic_field=True,
        )

    # 3. 切分文本
    all_parents, all_children = split_parent_child("\n".join(texts))

    if not all_children:
        return {}

    # 4. 批量生成向量 (假设你的 embed_many 函数支持传入字符串列表)
    child_texts = [c.text for c in all_children]
    try:
        # 如果你的 embedder 只能单条处理，可以用列表推导式: [embed_many(t) for t in child_texts]
        vectors = embed_many(child_texts)
    except Exception as e:
        print(f"生成向量时出错: {e}")
        raise e

    # 5. 组装入库数据
    data = []
    for i, c in enumerate(all_children):
        data.append({
            "vector": vectors[i],
            "parent_id": c.parent_id,
            "text": c.text
        })

    # 6. 插入 Milvus
    client.insert(collection_name, data)

    # 7. 返回父块字典，供内存缓存更新
    return {p.chunk_id: p.text for p in all_parents}