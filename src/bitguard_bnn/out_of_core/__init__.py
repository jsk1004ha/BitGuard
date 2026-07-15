"""Bounded-memory dataset normalization primitives."""

from .source import NormalizedChunk, iter_normalized_chunks

__all__ = ["NormalizedChunk", "iter_normalized_chunks"]
