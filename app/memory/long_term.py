"""
长期记忆 — 基于向量数据库的持久化记忆
存储用户画像、历史工单、知识库文档等需要持久化的信息。
支持语义相似度检索，用于RAG知识检索Agent。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆：基于FAISS的向量检索。

    特点：
    - 向量化存储，支持语义相似度检索
    - 持久化到磁盘，跨会话保持
    - 支持增量更新和批量导入
    - 生产环境可切换为Milvus/Pinecone

    文档分块策略：
    - 固定长度分块 (512 tokens) + 重叠窗口 (128 tokens)
    - 按段落自然分割优先
    """

    def __init__(
        self,
        index_path: str = "./vector_store/faiss_index",
        embedding_dim: int | None = None,
        embedding_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))
        self._api_base = (
            embedding_api_base
            or os.getenv("EMBEDDING_API_BASE", "")
        ).rstrip("/")
        self._api_key = (
            embedding_api_key
            or os.getenv("EMBEDDING_API_KEY", "")
        )
        self._model = (
            embedding_model
            or os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-4B")
        )
        self._client: httpx.AsyncClient | None = None
        self._documents: list[dict[str, Any]] = []
        self._index = None
        self._embedding_available = True
        self._init_index()

    def _init_index(self):
        """初始化FAISS索引"""
        if faiss is None:
            self._index = None
            return

        metadata_path = self.index_path.with_suffix(".meta.json")
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                if self._index.d != self.embedding_dim:
                    logger.warning(
                        "FAISS维度不匹配(索引=%d, 配置=%d)，重建索引",
                        self._index.d, self.embedding_dim,
                    )
                    self._index = faiss.IndexFlatIP(self.embedding_dim)
                    self._documents = []
                    return
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self._documents = json.load(f)
            except Exception:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    def _get_client(self) -> httpx.AsyncClient:
        """懒加载HTTP客户端"""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._api_base,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """通过OpenAI兼容API获取文本嵌入向量，失败时返回None"""
        if not self._embedding_available or not self._api_base:
            return None
        try:
            client = self._get_client()
            resp = await client.post(
                "/v1/embeddings",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        except Exception as e:
            logger.warning("Embedding API 调用失败，降级为纯词法检索: %s", e)
            self._embedding_available = False
            return None

    async def _get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量获取嵌入向量，失败时返回空列表"""
        if not texts or not self._embedding_available or not self._api_base:
            return []
        try:
            client = self._get_client()
            resp = await client.post(
                "/v1/embeddings",
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            vectors = []
            for item in data:
                vec = np.array(item["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                vectors.append(vec)
            return vectors
        except Exception as e:
            logger.warning("批量 Embedding API 调用失败，降级为纯词法检索: %s", e)
            self._embedding_available = False
            return []

    async def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """添加文档到向量库"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        doc = {
            "id": doc_id,
            "content": content,
            "source": source,
            "metadata": metadata or {},
        }
        self._documents.append(doc)

        if self._index is not None:
            embedding = await self._get_embedding(content)
            if embedding is not None:
                self._index.add(embedding.reshape(1, -1))

        return doc_id

    async def add_documents_batch(self, documents: list[dict]) -> list[str]:
        """批量添加文档（使用批量嵌入API，减少网络调用）"""
        if not documents:
            return []

        doc_ids = []
        contents = []
        for doc in documents:
            doc_id = hashlib.md5(doc.get("content", "").encode()).hexdigest()[:12]
            entry = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
            }
            self._documents.append(entry)
            doc_ids.append(doc_id)
            contents.append(entry["content"])

        if self._index is not None and contents:
            vectors = await self._get_embeddings_batch(contents)
            if vectors:
                batch = np.stack(vectors)
                self._index.add(batch)

        return doc_ids

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义相似度检索，并融合轻量关键词相关性重排。Embedding 不可用时降级为纯词法检索。"""
        if self._index is None or not self._documents:
            return self._fallback_search(query, top_k)

        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            return self._fallback_search(query, top_k)

        query_vec = query_embedding.reshape(1, -1)
        candidate_k = min(max(top_k * 3, top_k), len(self._documents))
        scores, indices = self._index.search(query_vec, candidate_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx].copy()
            lexical_score = self._lexical_score(query, doc.get("content", ""))
            doc["vector_score"] = float(score)
            doc["lexical_score"] = lexical_score
            doc["score"] = (0.35 * float(score)) + (0.65 * lexical_score)
            results.append(doc)

        seen_doc_ids = {doc["id"] for doc in results}
        for doc in self._documents:
            if doc["id"] in seen_doc_ids:
                continue
            copied = doc.copy()
            lexical_score = self._lexical_score(query, copied.get("content", ""))
            if lexical_score <= 0:
                continue
            copied["vector_score"] = 0.0
            copied["lexical_score"] = lexical_score
            copied["score"] = lexical_score
            results.append(copied)

        results.sort(key=lambda doc: doc.get("score", 0.0), reverse=True)
        return results[:top_k]

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        """抽取适合中文客服短问句的轻量检索词。"""
        normalized = query.lower()
        terms = set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", normalized))

        domain_terms = {
            "理财产品a", "理财产品", "投资期限", "最低投资金额", "最低金额",
            "收益率", "退款", "退款政策", "多久到账", "开户", "开户流程",
            "身份证", "视频认证", "风险评估", "工单", "保证收益",
        }
        for term in domain_terms:
            if term.lower() in normalized:
                terms.add(term.lower())

        chinese_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        for size in (2, 3, 4):
            for index in range(0, max(0, len(chinese_text) - size + 1)):
                terms.add(chinese_text[index:index + size])

        return {term for term in terms if term.strip()}

    @classmethod
    def _lexical_score(cls, query: str, content: str) -> float:
        """计算查询和文档的关键词/字符片段相关性。"""
        content_lower = content.lower()
        terms = cls._query_terms(query)
        if not terms:
            return 0.0

        hits = [term for term in terms if term in content_lower]
        if not hits:
            return 0.0

        coverage = len(hits) / len(terms)
        exact_bonus = 0.0
        for term in ("理财产品a", "投资期限", "最低投资金额", "退款", "开户流程"):
            if term in query.lower() and term in content_lower:
                exact_bonus += 0.2

        return min(1.0, coverage + exact_bonus)

    def _fallback_search(self, query: str, top_k: int) -> list[dict]:
        """当FAISS不可用时的关键词回退搜索"""
        scored = []

        for doc in self._documents:
            score = self._lexical_score(query, doc["content"])
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            copied = doc.copy()
            copied["score"] = score
            copied["lexical_score"] = score
            copied["vector_score"] = 0.0
            results.append(copied)
        return results

    def save(self):
        """持久化索引到磁盘"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

        metadata_path = self.index_path.with_suffix(".meta.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    async def close(self):
        """关闭HTTP客户端连接"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def load_knowledge_base(self, kb_dir: str) -> int:
        """从目录批量加载知识库文档（使用批量嵌入API）"""
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return 0

        all_chunks: list[dict] = []
        for file_path in kb_path.glob("**/*.txt"):
            content = file_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(content)
            for chunk in chunks:
                all_chunks.append({
                    "content": chunk,
                    "source": str(file_path.name),
                    "metadata": {"file": str(file_path)},
                })

        if not all_chunks:
            return 0

        batch_size = 64
        count = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            await self.add_documents_batch(batch)
            count += len(batch)

        return count

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list[str]:
        """
        文本分块：固定长度 + 重叠窗口。
        优先按段落分割，段落过长则按句子分割。
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    sentences = para.replace("。", "。\n").replace(".", ".\n").split("\n")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:chunk_size]]
