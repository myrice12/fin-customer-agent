"""
本地 Embedding Server — 暴露 OpenAI 兼容的 /v1/embeddings 端点
使用 Qwen/Qwen3-Embedding-0.6B 模型，支持本地推理。

启动方式：
    python embedding_server.py

默认监听 http://localhost:6008
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
HOST = os.getenv("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.getenv("EMBEDDING_PORT", "6008"))
MAX_SEQ_LEN = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", "8192"))
MAX_BATCH_SIZE = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "64"))

app = FastAPI(title="Local Embedding Server", version="1.0.0")


_tokenizer = None
_model = None
_device = "cpu"
_model_lock = asyncio.Lock()


async def _ensure_model_loaded() -> None:
    """懒加载模型：仅在首次推理时初始化，避免 import 期加载大模型。"""
    global _tokenizer, _model, _device
    async with _model_lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading embedding model: %s", MODEL_NAME)
        loop = asyncio.get_event_loop()
        _tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True),
        )
        _model = await loop.run_in_executor(
            None,
            lambda: AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True),
        )
        _model.eval()
        if torch.cuda.is_available():
            _device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
        _model = _model.to(_device)
        logger.info(
            "Model loaded on %s, embedding dim = %d",
            _device, _model.config.hidden_size,
        )


def _encode_sync(texts: list[str]) -> list[list[float]]:
    """同步执行模型推理（在线程池中调用，避免阻塞事件循环）。"""
    import torch

    assert _model is not None and _tokenizer is not None
    with torch.no_grad():
        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(_device)
        outputs = _model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        embeddings = embeddings / norms
        return embeddings.cpu().float().tolist()


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


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI 兼容的 embedding 端点"""
    await _ensure_model_loaded()

    texts = request.input if isinstance(request.input, list) else [request.input]
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}",
        )

    loop = asyncio.get_event_loop()
    start = time.time()
    vectors = await loop.run_in_executor(None, _encode_sync, texts)
    duration_ms = (time.time() - start) * 1000
    logger.info("Encoded %d text(s) in %.1fms", len(texts), duration_ms)

    data = [EmbeddingData(embedding=vec, index=i) for i, vec in enumerate(vectors)]
    total_tokens = sum(len(t) // 2 for t in texts)

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": _device, "loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)