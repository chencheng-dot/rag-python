-- 向量扩展。langchain PGVector 的 create_extension=True 也会建，这里预建一次确保权限落到 rag 用户
CREATE EXTENSION IF NOT EXISTS vector;
