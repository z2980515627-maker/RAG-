from openai import OpenAI
import config

client = OpenAI(api_key=config.API_KEY,base_url=config.BASE_URL)

def embed_one(text:str)->list[float]:
    # 单条文本变成向量
    resp = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding

def embed_many(texts:list[str])->list[list[float]]:
    # 批量向量化
    resp = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    # data 顺序和传入texts顺序一直，按index排序
    items = sorted(resp.data,key=lambda x:x.index)
    return [it.embedding for it in items]