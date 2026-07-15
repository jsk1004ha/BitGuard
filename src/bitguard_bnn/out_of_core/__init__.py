"""Bounded-memory dataset normalization primitives."""

from typing import Any

__all__ = ["NormalizedChunk", "iter_normalized_chunks"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .source import NormalizedChunk, iter_normalized_chunks

    exports = {
        "NormalizedChunk": NormalizedChunk,
        "iter_normalized_chunks": iter_normalized_chunks,
    }
    return exports[name]
