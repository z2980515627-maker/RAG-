import numpy as np
from core.embedder import embed_one, embed_many

# 1. 单条向量化
vec = embed_one("Milvus 是一个开源的向量数据库。")
print("向量维度:", len(vec))
print("前5维:", [round(x, 4) for x in vec[:5]])

# 2. 验证模长
norm = np.linalg.norm(vec)
print("向量模长:", round(norm, 4))

# 3. 批量向量化：传入字符串列表1
texts = ["苹果是一种水果。", "今天天气很好。", "Python 是一门编程语言。"]
vecs = embed_many(texts)
print("批量数量:", len(vecs), "每条维度:", len(vecs[0]))