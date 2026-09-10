from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bitguard_bnn.out_of_core.dataset import _ShardEntry, _scheduled_chunks, _total_chunk_count


def test_chunk_count_cache_matches_schedule_and_invalidates():
    entries = tuple(_ShardEntry(str(i), str(i), 9, "benign", (3, 2, 4)) for i in range(3))
    dataset = SimpleNamespace(entries=entries, mix_chunk_rows=5, shuffle_buffer_rows=10)
    dataset.permuted_shards = lambda: tuple(reversed(dataset.entries))
    assert _total_chunk_count(dataset) == len(list(_scheduled_chunks(dataset))) == 6
    with patch("bitguard_bnn.out_of_core.dataset._scheduled_chunks",
               side_effect=AssertionError("must reuse count")), patch(
                   "bitguard_bnn.out_of_core.dataset._row_group_chunk_count",
                   side_effect=AssertionError("must reuse count")):
        assert _total_chunk_count(dataset) == 6
    dataset.mix_chunk_rows = 10
    assert _total_chunk_count(dataset) == 3
    dataset.entries = (replace(entries[0], rows=14, row_group_rows=(10, 4)),)
    assert _total_chunk_count(dataset) == 2


def test_count_cache_does_not_hide_invalid_row_groups():
    entry = _ShardEntry("a", "a", 12, "benign", (12,))
    dataset = SimpleNamespace(entries=(entry,), mix_chunk_rows=12, shuffle_buffer_rows=12)
    dataset.permuted_shards = lambda: dataset.entries
    assert _total_chunk_count(dataset) == 1
    dataset.mix_chunk_rows = 10
    with pytest.raises(RuntimeError, match="exceeds"):
        _total_chunk_count(dataset)
