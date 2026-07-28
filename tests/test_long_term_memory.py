"""长期记忆 (LongTermMemory) 相关单元测试。"""

from __future__ import annotations

from app.memory.long_term import LongTermMemory


class _NoOpLongTermMemory(LongTermMemory):
    """跳过 FAISS 真实加载，避免依赖外部索引文件。"""

    def _init_index(self) -> None:  # noqa: D401 - overrides parent
        self._index = None


def test_query_terms_extracts_ascii_and_chinese_ngrams():
    terms = _NoOpLongTermMemory._query_terms("理财产品A的收益率是多少")
    assert "理财产品" in terms
    assert any(t.isdigit() or t.isascii() for t in terms) or "收益率" in terms
    assert all(t.strip() for t in terms)


def test_lexical_score_no_overfitting_bonus():
    """score 仅基于 coverage，不应再受硬编码业务词条加分。"""
    score_with = _NoOpLongTermMemory._lexical_score(
        "理财产品a的收益率",
        "理财产品a收益率较高，请注意风险。",
    )
    score_generic = _NoOpLongTermMemory._lexical_score(
        "理财产品a的收益率",
        "市场收益率波动较大，请注意风险。",
    )
    assert 0.0 <= score_with <= 1.0
    assert score_with >= score_generic


def test_lexical_score_is_low_when_only_ngram_overlap():
    """n-gram 重叠可能产生小分数；不应再因业务词表加分而虚高。"""
    score = _NoOpLongTermMemory._lexical_score(
        "完全无关的问题",
        "这是一段毫不相关的内容",
    )
    assert 0.0 < score <= 0.2


def test_chunk_text_respects_size_limit():
    text = ("段落一。" * 200) + "\n\n" + ("段落二。" * 200)
    chunks = _NoOpLongTermMemory._chunk_text(text, chunk_size=256, overlap=32)
    assert chunks
    assert all(len(c) <= 600 for c in chunks)