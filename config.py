import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
BASE_MODEL = os.getenv("DASHSCOPE_BASE_MODEL")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL")
EMBEDDING_DIM = 1024
if not API_KEY:
    raise RuntimeError("未配置API_KEY")
if not BASE_URL:
    raise RuntimeError("未配置BASE_URL")
if not BASE_MODEL:
    raise RuntimeError("未配置BASE_MODEL")
if not EMBEDDING_MODEL:
    raise RuntimeError("未配置EMBEDDING_MODEL")
