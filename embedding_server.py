"""
本地 Embedding Server — 暴露 OpenAI 兼容的 /v1/embeddings 端点
使用 Qwen/Qwen3-Embedding-0.6B 模型，支持本地推理。

启动方式：
    python embedding_server.py

默认监听 http://localhost:6008
"""

from __future__ import annotations

import os
import time
import logging

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 配置 ──
MODEL_NAME = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
HOST = os.getenv("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.getenv("EMBEDDING_PORT", "6008"))
MAX_SEQ_LEN = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", "8192"))

# ── 加载模型 ──
logger.info("Loading model: %s", MODEL_NAME)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
_model.eval()

_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_model = _model.to(_device)
logger.info("Model loaded on %s, embedding dim = %d", _device, _model.config.hidden_size)

app = FastAPI(title="Local Embedding Server", version="1.0.0")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def _encode(texts: list[str]) -> list[list[float]]:
    """批量编码文本为嵌入向量（mean pooling + L2 normalize）"""
    with torch.no_grad():
        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(_device)

        outputs = _model(**inputs)
        # Mean pooling over last hidden state, respecting attention mask
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts

        # L2 normalize
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        embeddings = embeddings / norms

        return embeddings.cpu().float().tolist()


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI 兼容的 embedding 端点"""
    texts = request.input if isinstance(request.input, list) else [request.input]

    start = time.time()
    vectors = _encode(texts)
    duration_ms = (time.time() - start) * 1000
    logger.info("Encoded %d text(s) in %.1fms", len(texts), duration_ms)

    data = [
        EmbeddingData(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]

    # Rough token count estimate
    total_tokens = sum(len(t) // 2 for t in texts)

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": _device}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
