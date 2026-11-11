# 基于 pgvector 的 RAG 知识库问答服务

一个可以直接跑起来的最小可用 RAG 服务：文档清洗 → 切片 → 向量化入库 → 问题改写 → 欧氏距离检索 → 重排 → 带引用生成答案。全流程用 LangChain 官方组件拼装，向量库只要一个 Docker 容器（PostgreSQL + pgvector），不需要额外部署检索服务。

## 特性

- **数据清洗**：去 HTML/脚本标签、Markdown 图片、零宽与控制字符、页码行、纯符号行与重复行，保留段落边界
- **三种切片方式**：按字数（100 / 500 / 1000）、按段落、按章节，默认按字数 500
- **欧氏距离检索**：pgvector 上的 L2 最近邻，返回的是距离不是相似度，接口字段直白叫 `vectorDistance`
- **真重排**：DashScope `text-rerank` 对候选做二次排序，内置 0.7 / 0.6 两档相似度门限
- **查询改写**：短问句或含指代的问题先补全再向量化，改写只影响 embedding，重排仍按原问题打分
- **带引用的生成**：强制 `[n]` 标注、越界编号自动剔除并重生成一次、统计无引用事实句
- **失败不装死**：重排失败降级为向量顺序、改写失败回退原问题，响应里如实标 `reranked` / `badReferences`

## 链路

```text
入库：
  原文 → 清洗 → 切片(3 种方式) → embedding 向量化 → 写入 langchain_pg_embedding

问答：
  问题 → [模糊则改写] → embedding → pgvector L2 检索(top 20)
       → rerank 重排 → 相似度门限(0.7/0.6) → 拼参考资料 → LLM 生成(带引用校验)
```

## 技术栈

| 组件 | 选型 |
| --- | --- |
| 运行时 | Python 3.12 |
| Web | FastAPI + uvicorn |
| RAG 组件 | LangChain（langchain-core / langchain-openai / langchain-text-splitters / langchain-postgres） |
| 向量库 | PostgreSQL 16 + pgvector 0.8（Docker） |
| 模型 | 任意 OpenAI 兼容端点；默认阿里云 DashScope（`qwen-plus` + `qwen3.7-text-embedding-flash` + `qwen3.7-text-rerank`） |

## 快速开始

前置：Docker Desktop、Python 3.12。

```bash
# 1. 起数据库（会创建 rag_db 并启用 vector 扩展）
docker compose up -d

# 2. 装依赖
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
# 编辑 .env，至少填写 OPENAI_API_KEY 与 RERANK_API_KEY

# 4. 入库一个文件（支持 .txt / .md）
python -m app.cli ingest samples/demo.md

# 5. 提问
python -m app.cli ask "知识库默认把正文切成多大的块？"

# 6. 起 HTTP 服务（默认 8000 端口）
uvicorn app.main:app --reload
```

服务起来后打开 <http://127.0.0.1:8000/docs> 可直接调试接口。

## 配置

全部通过 `.env` 或环境变量注入，`.env` 已加入 `.gitignore`，不要提交真实 Key。

### 数据库

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PG_HOST` | `127.0.0.1` | 数据库地址 |
| `PG_PORT` | `5432` | 端口，与 `docker-compose.yml` 一致 |
| `PG_DB` | `rag_db` | 库名 |
| `PG_USER` / `PG_PASSWORD` | `rag` / `rag_pwd` | 账号口令 |
| `PG_COLLECTION` | `kb_chunks` | pgvector 集合名 |

### 模型

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | DashScope 兼容模式 | OpenAI 兼容端点 |
| `OPENAI_API_KEY` | 空 | 必填 |
| `CHAT_MODEL` | `qwen-plus` | 生成模型 |
| `EMBEDDING_MODEL` | `qwen3.7-text-embedding-flash` | 向量模型 |
| `EMBEDDING_DIMS` | `1024` | 必须与向量模型输出维度一致，否则建表/检索报维度不匹配 |
| `RERANK_ENABLED` | `true` | 关闭后退化为纯向量排序 |
| `RERANK_BASE_URL` | `https://dashscope.aliyuncs.com` | 重排端点根地址 |
| `RERANK_API_KEY` | 空 | 不填则复用 `OPENAI_API_KEY` |
| `RERANK_MODEL` | `qwen3.7-text-rerank` | 重排模型 |

### 检索与生成

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHUNK_MODE` | `char` | 切片方式：`char` / `paragraph` / `section` |
| `CHUNK_SIZE` | `500` | 按字数切片的块大小，仅支持 100 / 500 / 1000 |
| `CHUNK_OVERLAP` | `80` | 相邻块重叠字数 |
| `RETRIEVE_TOP_N` | `20` | 向量召回的候选条数 |
| `RERANK_TOP_K` | `5` | 重排后进入提示词的条数 |
| `SIMILARITY_MODE` | `precise` | `precise`=0.7（精确）/ `broad`=0.6（广泛） |
| `REWRITE_ENABLED` | `true` | 是否对模糊问题做查询改写 |
| `CHUNK_MAX_CHARS` | `800` | 单块进入提示词的最大长度，超出截断并标记 |
| `MAX_CONTEXT_CHARS` | `6000` | 参考资料总预算 |

## 切片与相似度

切片三种方式（`POST /ingest` 可用 `chunkMode` / `chunkSize` 单次覆盖）：

| 方式 | 边界 | 行为 |
| --- | --- | --- |
| `char` | 按字数 | 用递归分隔符切（中文标点优先），块大小 100 / 500 / 1000 |
| `paragraph` | 空行 | 一个段落一块；单段超长回退按字数切 |
| `section` | Markdown `#`~`######` | 一节一块；整节超长回退按段落、再回退按字数 |

相似度两档决定重排分门限，低于门限的候选直接丢弃；一条都不剩时**不调用 LLM**，直接回「知识库里没有找到相关内容」：

| 模式 | 门限 | 适用 |
| --- | --- | --- |
| `precise`（默认） | 0.7 | 要求答案紧扣资料 |
| `broad` | 0.6 | 允许较宽的相关资料参与 |

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 配置概览（模型、集合、默认切片与相似度） |
| GET | `/db/status` | 数据库连通性、pgvector 版本、已有切片数 |
| POST | `/ingest` | 提交正文入库，可指定切片方式 |
| POST | `/ingest/file` | 上传 `.txt` / `.md` 入库 |
| POST | `/ask` | 提问，可指定相似度模式 |

### 入库

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"知识库默认按 500 字切块。","source":"demo.md","chunkMode":"section"}'
```

```json
{"source": "demo.md", "chunks": 1, "chunkMode": "section", "chunkSize": 500, "skipped": false}
```

### 提问

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"知识库默认把正文切成多大的块？","similarityMode":"precise"}'
```

```json
{
  "answer": "知识库默认把正文按 500 字切块 [1]。",
  "citations": [
    {"index": 1, "source": "demo.md", "chunkIndex": 1, "vectorDistance": 0.9067, "rerankScore": 1.0}
  ],
  "retrieved": 2,
  "reranked": true,
  "similarityMode": "precise",
  "minScore": 0.7,
  "droppedByScore": 1,
  "rewrite": null,
  "uncitedClaims": 0,
  "badReferences": [],
  "citationRetried": false
}
```

字段含义：

- `citationRetried`：模型是否引用过不存在的编号、被纠正后重新生成
- `badReferences`：重生成后仍然越界、已被剔除的编号
- `uncitedClaims`：没有任何引用标记的事实句数量，偏高说明召回没给到答案
- `droppedByScore`：被相似度门限丢弃的候选数
- `rewrite`：改写结果（`used` 为 false 表示问题不模糊、未改写）

## 命令行

```bash
python -m app.cli db-status
python -m app.cli ingest samples/demo.md --mode section --size 500
python -m app.cli ask "它默认切多大块？" --similarity broad
```

## 目录结构

```text
app/
├─ config.py         环境变量与相似度预设
├─ cleaner.py        文本清洗
├─ chunker.py        三种切片方式
├─ store.py          embedding 与 pgvector 客户端
├─ rerank.py         DashScope 重排（失败降级）
├─ query_rewrite.py  查询改写（失败回退）
├─ pipeline.py       入库与问答主流程、引用校验
├─ main.py           FastAPI 接口
└─ cli.py            命令行入口
docker/
└─ init.sql          启用 vector 扩展
docker-compose.yml   pgvector 容器
```

## 设计说明

- **距离不是相似度**：`similarity_search_with_score` 在 L2 下返回距离，越小越近，因此用 `vectorDistance` 命名，避免前端当成 0–1 相似度用
- **DashScope 兼容模式的坑**：`OpenAIEmbeddings` 必须设 `check_embedding_ctx_length=False`，否则会先 tiktoken 分词、把 token 数组发过去，接口直接 400
- **重排按原问题打分**：改写句只用于 embedding，防止改写漂移带偏排序
- **编号即出处**：参考资料按重排顺序从 1 连续编号，`citations[].index` 与文中 `[n]` 一一对应
- **惰性构建客户端**：缺少配置时服务仍能启动，`/health` 只报降级，不会因一个 Key 没填就起不来

## 常见问题

**启动报维度不匹配**：换了向量模型但表已存在。`EMBEDDING_DIMS` 必须与模型输出一致；换模型后需清空集合重新入库（不同模型的向量不可混用）。

**重排返回 403 / 404**：确认 `RERANK_MODEL` 在该账号下可用，DashScope 不同模型权限不一致；也可临时 `RERANK_ENABLED=false`。

**检索一直为空**：先看 `/db/status` 的 `chunks` 是否为 0；再检查 `OPENAI_API_KEY` 是否为空（空 Key 会导致向量化失败但服务不报错）。

**5432 端口被占用**：改 `docker-compose.yml` 的端口映射与 `.env` 的 `PG_PORT` 为同一值即可。

**按段落切结果和按字数一样**：检查清洗后的正文是否还有空行——空行是 `paragraph` 模式的唯一边界。
