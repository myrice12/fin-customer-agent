"""
RAG评测指标计算。

用于离线评测知识库召回、重排序、引用覆盖和上下文压缩效果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RAGEvalCase:
    """单条RAG评测样本"""

    question: str
    gold_sources: list[str]
    gold_answer_points: list[str]


def _estimate_tokens(text: str) -> int:
    """粗略token估算，适合中文/英文混合客服文本的轻量指标。"""
    return max(1, len(text) // 2)


def recall_at_k(retrieved_docs: list[dict[str, Any]], gold_sources: list[str], k: int = 5) -> float:
    """计算Recall@K：正确来源是否被前K个结果覆盖。"""
    if not gold_sources:
        return 0.0
    retrieved_sources = {doc.get("source", "") for doc in retrieved_docs[:k]}
    hits = sum(1 for source in gold_sources if source in retrieved_sources)
    return hits / len(gold_sources)


def mrr_at_k(retrieved_docs: list[dict[str, Any]], gold_sources: list[str], k: int = 5) -> float:
    """计算MRR@K：第一个正确来源的倒数排名。"""
    gold = set(gold_sources)
    for index, doc in enumerate(retrieved_docs[:k], start=1):
        if doc.get("source", "") in gold:
            return 1 / index
    return 0.0


def ndcg_at_k(reranked_docs: list[dict[str, Any]], gold_sources: list[str], k: int = 3) -> float:
    """计算二值相关性的nDCG@K。"""
    gold = set(gold_sources)
    gains = [1.0 if doc.get("source", "") in gold else 0.0 for doc in reranked_docs[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))

    ideal_hits = min(len(gold), k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def citation_coverage(answer: str, gold_sources: list[str]) -> float:
    """计算引用覆盖率：标准来源是否出现在回答中。"""
    if not gold_sources:
        return 0.0
    hits = sum(1 for source in gold_sources if source in answer)
    return hits / len(gold_sources)


def answer_point_coverage(answer: str, gold_answer_points: list[str]) -> float:
    """计算答案要点覆盖率，用于轻量替代人工faithfulness打分。"""
    if not gold_answer_points:
        return 0.0
    hits = sum(1 for point in gold_answer_points if point and point in answer)
    return hits / len(gold_answer_points)


def context_compression_ratio(input_docs: list[dict[str, Any]], injected_docs: list[dict[str, Any]]) -> float:
    """计算上下文压缩率：最终注入上下文 / 候选上下文。"""
    raw_tokens = sum(_estimate_tokens(doc.get("content", "")) for doc in input_docs)
    injected_tokens = sum(_estimate_tokens(doc.get("content", "")) for doc in injected_docs)
    return injected_tokens / raw_tokens if raw_tokens else 0.0


def evaluate_rag_case(
    case: RAGEvalCase,
    retrieved_docs: list[dict[str, Any]],
    reranked_docs: list[dict[str, Any]],
    answer: str,
) -> dict[str, float]:
    """汇总单条样本的RAG指标。"""
    return {
        "recall_at_5": recall_at_k(retrieved_docs, case.gold_sources, k=5),
        "mrr_at_5": mrr_at_k(retrieved_docs, case.gold_sources, k=5),
        "ndcg_at_3": ndcg_at_k(reranked_docs, case.gold_sources, k=3),
        "citation_coverage": citation_coverage(answer, case.gold_sources),
        "answer_point_coverage": answer_point_coverage(answer, case.gold_answer_points),
        "context_compression_ratio": context_compression_ratio(retrieved_docs, reranked_docs),
    }
