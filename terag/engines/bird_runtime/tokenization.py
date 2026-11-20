"""
Tokenization utilities.

Provides a unified Chinese / English tokenization interface:
- chinese: jieba segmentation (PowerGrid Chinese scenario)
- english: lightweight regex tokenization (BIRD English scenario):
  lowercase + word tokenization + digit retention + handling of
  underscores / hyphens / parentheses

Avoids heavyweight NLP dependencies.
"""

import re
from typing import List

# English tokenization: extract alphanumeric tokens after lowercasing.
# Underscores, hyphens, parentheses and punctuation all act as separators, so
# snake_case / kebab-case field names (e.g. expense_description, K-12) are
# split into sensible tokens.
_EN_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_english(text: str) -> List[str]:
    """
    Tokenize English text.

    Args:
        text: input text

    Returns:
        Lowercased token list, e.g. "Free Meal Count (K-12)" -> ['free', 'meal', 'count', 'k', '12']
    """
    if not text:
        return []
    return _EN_WORD_RE.findall(str(text).lower())


def tokenize_chinese(text: str) -> List[str]:
    """Tokenize Chinese text (jieba)."""
    import jieba
    if not text:
        return []
    return list(jieba.cut(str(text)))


def tokenize_text(text: str, language: str = "chinese") -> List[str]:
    """
    Tokenize text according to the configured language.

    Args:
        text: input text
        language: "chinese" (jieba) or "english" (regex tokenization)

    Returns:
        Token list
    """
    if language == 'english':
        return tokenize_english(text)
    return tokenize_chinese(text)


def tokenize_set(text: str, language: str = "chinese", min_len: int = 1) -> set:
    """
    Tokenize text by language and return a token set (optionally filtering
    out short tokens).

    Args:
        text: input text
        language: tokenization language
        min_len: minimum token length (project convention: pass min_len=2
            for the usual len(t) > 1 filter)

    Returns:
        Token set
    """
    return {t for t in tokenize_text(text, language) if len(t) >= min_len}
