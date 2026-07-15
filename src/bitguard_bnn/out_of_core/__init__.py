"""Bounded-memory dataset normalization primitives."""

from typing import Any

__all__ = [
    "NormalizedChunk",
    "NormalizedSource",
    "NormalizedSourceFileProof",
    "NormalizedSourceProof",
    "iter_normalized_chunks",
    "open_normalized_source",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .source import (
        NormalizedChunk,
        NormalizedSource,
        NormalizedSourceFileProof,
        NormalizedSourceProof,
        iter_normalized_chunks,
        open_normalized_source,
    )

    exports = {
        "NormalizedChunk": NormalizedChunk,
        "NormalizedSource": NormalizedSource,
        "NormalizedSourceFileProof": NormalizedSourceFileProof,
        "NormalizedSourceProof": NormalizedSourceProof,
        "iter_normalized_chunks": iter_normalized_chunks,
        "open_normalized_source": open_normalized_source,
    }
    return exports[name]
