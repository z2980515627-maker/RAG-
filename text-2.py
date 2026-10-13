from core.parser import parse_any
from core.splitter import split_parent_child

# 解析文档
text = parse_any("docs/test.txt")
print("文档总字符：",len(text))
print("前100字：",text[:100])

# 父子分块
parents, children = split_parent_child(text)
print("父块数量：",len(parents))
print("子块数量:",len(children))

# 查看父块对应的子块
p = parents[0]
print(f"\n父块0 （id={p.chunk_id}）,长度={len(p.text)}：")
print(p.text[:80],"...")

c_of_p0 = [c for c in children if c.parent_id == p.chunk_id]
print(f"下属子块数：{len(c_of_p0)}")
for c in c_of_p0:
    print(f"{c.chunk_id},{c.text[:50]}")