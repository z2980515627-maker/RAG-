import os
from openai import OpenAI
import config

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
)

def generate(query:str,contexts:list[str],stream:bool = False)->str:
    """
    RAG生成：把检索命中的上下文拼进prompt，交给千问答
    """
    refs = "\n\n".join(contexts)
    prompt = f"""你是 RAG 助手。严格基于以下参考资料回答，资料不足就说"资料不足"。

    参考资料：
    {refs}

    问题：{query}
    回答："""

    resp = client.chat.completions.create(
        model=config.BASE_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        stream=stream,
    )
    if stream:
        return resp
    return resp.choices[0].message.content