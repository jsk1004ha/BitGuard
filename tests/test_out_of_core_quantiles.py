from __future__ import annotations

import builtins
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from bitguard_bnn.out_of_core import quantiles as quantile_module
from bitguard_bnn.out_of_core.quantiles import PriorityRowSketch


def _records(count: int, width: int = 3) -> list[tuple[str, np.ndarray]]:
    return [
        (
            f"row-{index:06d}",
            np.asarray([index / count + column for column in range(width)], dtype=np.float64),
        )
        for index in range(count)
    ]


def _serialize_payload(payload: dict[str, object]) -> bytes:
    payload_bytes = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return json.dumps(
        {
            "checksum": hashlib.sha256(payload_bytes).hexdigest(),
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload(sketch: PriorityRowSketch) -> dict[str, object]:
    return json.loads(sketch.to_bytes())["payload"]


def _max_count_sketch(uid: str) -> PriorityRowSketch:
    sketch = PriorityRowSketch(capacity=1, seed=1, width=1)
    sketch.update(uid, [1.0])
    payload = _payload(sketch)
    payload["total_rows"] = np.iinfo(np.int64).max
    payload["finite_counts"] = [np.iinfo(np.int64).max]
    payload["missing_counts"] = [0]
    return PriorityRowSketch.from_bytes(_serialize_payload(payload))


class PriorityRowSketchTests(unittest.TestCase):
    def test_public_limits_are_versioned_and_cover_full_profiles(self) -> None:
        self.assertGreaterEqual(quantile_module.LIMITS_VERSION, 1)
        self.assertGreaterEqual(quantile_module.MAX_SKETCH_WIDTH, 115)
        self.assertGreaterEqual(quantile_module.MAX_SKETCH_CAPACITY, 200_000)
        self.assertGreaterEqual(
            quantile_module.MAX_RETAINED_ROWS,
            quantile_module.MAX_SKETCH_CAPACITY,
        )
        self.assertGreaterEqual(
            quantile_module.MAX_RETAINED_VALUES,
            200_000 * 115,
        )
        self.assertGreaterEqual(quantile_module.MAX_SERIALIZED_BYTES, 512 * 1024**2)
        planned = PriorityRowSketch(capacity=200_000, seed=17, width=115)
        self.assertEqual(planned.capacity, 200_000)

    def test_capacity_width_product_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity.*width|retained values"):
            PriorityRowSketch(
                capacity=quantile_module.MAX_SKETCH_CAPACITY,
                seed=1,
                width=quantile_module.MAX_SKETCH_WIDTH,
            )

    def test_is_order_and_merge_independent(self) -> None:
        records = _records(137)
        whole = PriorityRowSketch(capacity=31, seed=17, width=3)
        reversed_order = PriorityRowSketch(capacity=31, seed=17, width=3)
        odd = PriorityRowSketch(capacity=31, seed=17, width=3)
        even = PriorityRowSketch(capacity=31, seed=17, width=3)

        for uid, values in records:
            whole.update(uid, values)
        for uid, values in reversed(records):
            reversed_order.update(uid, values)
        for uid, values in records[::2]:
            even.update(uid, values)
        for uid, values in records[1::2]:
            odd.update(uid, values)
        even.merge(odd)

        self.assertEqual(whole.snapshot(), reversed_order.snapshot())
        self.assertEqual(whole.snapshot(), even.snapshot())
        self.assertEqual(whole.to_bytes(), reversed_order.to_bytes())
        self.assertEqual(whole.to_bytes(), even.to_bytes())

    def test_update_many_is_chunk_independent(self) -> None:
        records = _records(51)
        uids = np.asarray([uid for uid, _ in records], dtype=object)
        values = np.stack([values for _, values in records])
        whole = PriorityRowSketch(capacity=17, seed=9, width=3)
        chunked = PriorityRowSketch(capacity=17, seed=9, width=3)

        whole.update_many(uids, values)
        chunked.update_many(uids[:7], values[:7])
        chunked.update_many(uids[7:36], values[7:36])
        chunked.update_many(uids[36:], values[36:])

        self.assertEqual(whole.snapshot(), chunked.snapshot())

    def test_priority_collision_uses_uid_as_deterministic_tie_breaker(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=3, width=1)
        with patch(
            "bitguard_bnn.out_of_core.quantiles._priority_for_uid",
            return_value=42,
        ):
            for uid in ("z", "b", "a", "m"):
                sketch.update(uid, [float(ord(uid))])

        retained = [row["row_uid"] for row in sketch.snapshot()["retained_rows"]]
        self.assertEqual(retained, ["a", "b"])

    def test_capacity_is_a_hard_bound(self) -> None:
        sketch = PriorityRowSketch(capacity=7, seed=1, width=3)
        sketch.update_many(
            [uid for uid, _ in _records(500)],
            np.stack([values for _, values in _records(500)]),
        )
        self.assertEqual(sketch.total_rows, 500)
        self.assertEqual(sketch.retained_count, 7)
        self.assertLessEqual(len(sketch.snapshot()["retained_rows"]), 7)

    def test_retained_rows_are_canonical_read_only_copies(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=1, width=2)
        sketch.update_many(["b", "a", "c"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        before = sketch.to_bytes()

        rows = sketch.retained_rows()

        self.assertEqual(len(rows), 2)
        repeated = sketch.retained_rows()
        self.assertEqual([uid for uid, _ in rows], [uid for uid, _ in repeated])
        for (_, values), (_, repeated_values) in zip(rows, repeated):
            np.testing.assert_array_equal(values, repeated_values)
        for _, values in rows:
            self.assertFalse(values.flags.writeable)
            with self.assertRaises(ValueError):
                values[0] = -1.0
        self.assertEqual(sketch.to_bytes(), before)

    def test_shape_and_sampling_identity_are_immutable(self) -> None:
        sketch = PriorityRowSketch(capacity=7, seed=1, width=3)
        for name, value in (("capacity", 8), ("seed", 2), ("width", 4)):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(sketch, name, value)

    def test_quantile_fixture_has_declared_distribution_specific_tolerance(self) -> None:
        rng = np.random.default_rng(20260715)
        values = rng.normal(size=(20_000, 1))
        sketch = PriorityRowSketch(capacity=4096, seed=17, width=1)
        sketch.update_many([f"uid-{index}" for index in range(len(values))], values)

        self.assertLessEqual(
            abs(sketch.quantile(0, 0.5) - float(np.median(values[:, 0]))),
            0.05,
        )

    def test_counts_cover_all_rows_while_quantiles_ignore_nonfinite_values(self) -> None:
        sketch = PriorityRowSketch(capacity=10, seed=1, width=3)
        sketch.update("a", [1.0, math.nan, math.inf])
        sketch.update("b", [2.0, 4.0, -math.inf])
        sketch.update("c", [3.0, 6.0, 9.0])

        self.assertEqual(sketch.total_rows, 3)
        np.testing.assert_array_equal(sketch.finite_counts, [3, 2, 1])
        np.testing.assert_array_equal(sketch.missing_counts, [0, 1, 2])
        self.assertEqual(sketch.quantile(0, 0.5), 2.0)
        self.assertEqual(sketch.quantile(1, 0.5), 5.0)
        self.assertEqual(sketch.quantile(2, 0.5), 9.0)

    def test_all_missing_and_empty_quantiles_are_rejected(self) -> None:
        empty = PriorityRowSketch(capacity=2, seed=1, width=1)
        with self.assertRaisesRegex(ValueError, "no finite"):
            empty.quantile(0, 0.5)

        missing = PriorityRowSketch(capacity=2, seed=1, width=1)
        missing.update("nan", [math.nan])
        missing.update("inf", [math.inf])
        self.assertEqual(missing.missing_counts.tolist(), [2])
        with self.assertRaisesRegex(ValueError, "no finite"):
            missing.quantile(0, 0.5)

    def test_constructor_and_update_validation(self) -> None:
        for kwargs, message in (
            ({"capacity": 0, "seed": 1, "width": 1}, "capacity"),
            ({"capacity": True, "seed": 1, "width": 1}, "capacity"),
            ({"capacity": 1, "seed": 1, "width": 0}, "width"),
            ({"capacity": 1, "seed": True, "width": 1}, "seed"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                PriorityRowSketch(**kwargs)

        sketch = PriorityRowSketch(capacity=2, seed=1, width=2)
        invalid = (
            ("", [1.0, 2.0], "row_uid"),
            (1, [1.0, 2.0], "row_uid"),
            ("short", [1.0], "width"),
            ("long", [1.0, 2.0, 3.0], "width"),
            ("matrix", [[1.0, 2.0]], "one-dimensional"),
            ("text", ["x", 2.0], "numeric"),
            ("numeric-text", ["1.0", "2.0"], "numeric"),
            ("boolean", [True, False], "numeric"),
        )
        for uid, values, message in invalid:
            with self.subTest(uid=uid), self.assertRaisesRegex(
                (TypeError, ValueError), message
            ):
                sketch.update(uid, values)  # type: ignore[arg-type]
        self.assertEqual(sketch.total_rows, 0)

    def test_quantile_validation(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=1, width=2)
        sketch.update("a", [1.0, 2.0])
        for column in (-1, 2, True):
            with self.subTest(column=column), self.assertRaisesRegex(
                (TypeError, ValueError), "column"
            ):
                sketch.quantile(column, 0.5)
        for q in (-0.1, 1.1, math.nan, math.inf, "half", True):
            with self.subTest(q=q), self.assertRaisesRegex(
                (TypeError, ValueError), "q"
            ):
                sketch.quantile(0, q)  # type: ignore[arg-type]

    def test_duplicate_uid_is_idempotent_only_for_the_exact_same_row(self) -> None:
        sketch = PriorityRowSketch(capacity=4, seed=1, width=2)
        sketch.update("same", [1.0, math.nan])
        sketch.update("same", [1.0, math.nan])
        self.assertEqual(sketch.total_rows, 1)
        self.assertEqual(sketch.finite_counts.tolist(), [1, 0])
        self.assertEqual(sketch.missing_counts.tolist(), [0, 1])

        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            sketch.update("same", [1.0, 2.0])

    def test_merge_deduplicates_identical_overlap_and_rejects_conflict(self) -> None:
        left = PriorityRowSketch(capacity=10, seed=1, width=2)
        right = PriorityRowSketch(capacity=10, seed=1, width=2)
        left.update("shared", [1.0, 2.0])
        left.update("left", [3.0, 4.0])
        right.update("shared", [1.0, 2.0])
        right.update("right", [5.0, 6.0])

        left.merge(right)
        self.assertEqual(left.total_rows, 3)
        self.assertEqual(left.finite_counts.tolist(), [3, 3])

        conflict = PriorityRowSketch(capacity=10, seed=1, width=2)
        conflict.update("shared", [7.0, 8.0])
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            left.merge(conflict)

    def test_serialization_is_canonical_and_round_trips_nonfinite_values(self) -> None:
        sketch = PriorityRowSketch(capacity=3, seed=-7, width=3)
        sketch.update("z", [math.nan, math.inf, -math.inf])
        sketch.update("a", [-0.0, 1.25, 2.5])
        payload = sketch.to_bytes()
        restored = PriorityRowSketch.from_bytes(payload)

        self.assertEqual(restored.snapshot(), sketch.snapshot())
        self.assertEqual(restored.to_bytes(), payload)

    def test_serialization_rejects_tampering_and_unsupported_versions(self) -> None:
        sketch = PriorityRowSketch(capacity=3, seed=1, width=1)
        sketch.update("a", [1.0])
        tampered = bytearray(sketch.to_bytes())
        tampered[len(tampered) // 2] ^= 1
        with self.assertRaisesRegex(ValueError, "tampered|checksum|serialization"):
            PriorityRowSketch.from_bytes(bytes(tampered))

        envelope = json.loads(sketch.to_bytes())
        envelope["payload"]["version"] = 999
        payload_bytes = json.dumps(
            envelope["payload"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        envelope["checksum"] = hashlib.sha256(payload_bytes).hexdigest()
        unsupported = json.dumps(
            envelope, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "version"):
            PriorityRowSketch.from_bytes(unsupported)

    def test_deserialization_rejects_hostile_shapes_before_allocation(self) -> None:
        sketch = PriorityRowSketch(capacity=1, seed=1, width=1)
        sketch.update("a", [1.0])
        baseline = _payload(sketch)
        cases = (
            ("width", 10**15, "width"),
            ("capacity", 10**15, "capacity"),
            ("width", True, "width"),
            ("seed", 2**63, "seed"),
            ("total_rows", np.iinfo(np.int64).max + 1, "total_rows"),
        )
        for field, value, message in cases:
            hostile = dict(baseline)
            hostile[field] = value
            with (
                self.subTest(field=field, value=value),
                patch.object(
                    quantile_module.np,
                    "zeros",
                    side_effect=AssertionError("allocation attempted"),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                PriorityRowSketch.from_bytes(_serialize_payload(hostile))

        product = dict(baseline)
        product["capacity"] = quantile_module.MAX_SKETCH_CAPACITY
        product["width"] = quantile_module.MAX_SKETCH_WIDTH
        product["finite_counts"] = [1] * quantile_module.MAX_SKETCH_WIDTH
        product["missing_counts"] = [0] * quantile_module.MAX_SKETCH_WIDTH
        with (
            patch.object(
                quantile_module.np,
                "zeros",
                side_effect=AssertionError("allocation attempted"),
            ),
            self.assertRaisesRegex(ValueError, "capacity.*width|retained values"),
        ):
            PriorityRowSketch.from_bytes(_serialize_payload(product))

    def test_deserialization_validates_list_lengths_before_allocation(self) -> None:
        sketch = PriorityRowSketch(capacity=1, seed=1, width=1)
        sketch.update("a", [1.0])
        baseline = _payload(sketch)
        cases: tuple[tuple[str, list[object], str], ...] = (
            ("finite_counts", [], "finite_counts"),
            ("missing_counts", [0, 0], "missing_counts"),
            ("retained_rows", [], "retained row"),
        )
        for field, value, message in cases:
            hostile = dict(baseline)
            hostile[field] = value
            with (
                self.subTest(field=field),
                patch.object(
                    quantile_module.np,
                    "zeros",
                    side_effect=AssertionError("allocation attempted"),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                PriorityRowSketch.from_bytes(_serialize_payload(hostile))

    def test_deserialization_bounds_retained_rows_before_allocation(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=1, width=1)
        sketch.update_many(["a", "b"], [[1.0], [2.0]])
        serialized = sketch.to_bytes()
        with (
            patch.object(quantile_module, "MAX_RETAINED_ROWS", 1),
            patch.object(
                quantile_module.np,
                "zeros",
                side_effect=AssertionError("allocation attempted"),
            ),
            self.assertRaisesRegex(ValueError, "retained"),
        ):
            PriorityRowSketch.from_bytes(serialized)

    def test_deserialization_rejects_oversized_bytes_before_json_or_allocation(self) -> None:
        sketch = PriorityRowSketch(capacity=1, seed=1, width=1)
        serialized = sketch.to_bytes()
        with (
            patch.object(
                quantile_module,
                "MAX_SERIALIZED_BYTES",
                len(serialized) - 1,
            ),
            patch.object(
                quantile_module.json,
                "loads",
                side_effect=AssertionError("JSON parse attempted"),
            ),
            patch.object(
                quantile_module.np,
                "zeros",
                side_effect=AssertionError("allocation attempted"),
            ),
            self.assertRaisesRegex(ValueError, "serialized.*size|too large"),
        ):
            PriorityRowSketch.from_bytes(serialized)

    def test_constructor_rejects_seed_outside_signed_64_bit_range(self) -> None:
        for seed in (-(2**63) - 1, 2**63, 10**5000):
            with self.subTest(seed_bits=seed.bit_length()), self.assertRaisesRegex(
                ValueError, "seed"
            ):
                PriorityRowSketch(capacity=1, seed=seed, width=1)

    def test_update_overflow_and_priority_failure_are_transactional(self) -> None:
        full = _max_count_sketch("full")
        before = full.to_bytes()
        with self.assertRaisesRegex((OverflowError, ValueError), "limit|overflow"):
            full.update("extra", [2.0])
        self.assertEqual(full.to_bytes(), before)

        ordinary = PriorityRowSketch(capacity=2, seed=1, width=1)
        ordinary.update("existing", [1.0])
        before = ordinary.to_bytes()
        with (
            patch.object(
                quantile_module,
                "_priority_for_uid",
                side_effect=RuntimeError("priority failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "priority failed"),
        ):
            ordinary.update("new", [2.0])
        self.assertEqual(ordinary.to_bytes(), before)

    def test_merge_counter_overflow_is_rejected_transactionally(self) -> None:
        left = _max_count_sketch("left")
        right = _max_count_sketch("right")
        before = left.to_bytes()
        with self.assertRaisesRegex((OverflowError, ValueError), "limit|overflow"):
            left.merge(right)
        self.assertEqual(left.to_bytes(), before)

    def test_merge_requires_compatible_sketches_and_is_transactional(self) -> None:
        base = PriorityRowSketch(capacity=3, seed=1, width=2)
        base.update("base", [1.0, 2.0])
        before = base.to_bytes()
        incompatible = (
            PriorityRowSketch(capacity=4, seed=1, width=2),
            PriorityRowSketch(capacity=3, seed=2, width=2),
            PriorityRowSketch(capacity=3, seed=1, width=3),
        )
        for other in incompatible:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "incompatible"):
                base.merge(other)
            self.assertEqual(base.to_bytes(), before)

        algorithm = PriorityRowSketch(capacity=3, seed=1, width=2)
        algorithm._algorithm = "future-algorithm"
        version = PriorityRowSketch(capacity=3, seed=1, width=2)
        version._version = 999
        for other in (algorithm, version):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                base.merge(other)
            self.assertEqual(base.to_bytes(), before)

    def test_self_merge_is_rejected_without_mutation(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=1, width=1)
        sketch.update_many(["a", "b", "c"], [[1.0], [2.0], [3.0]])
        before = sketch.to_bytes()
        with self.assertRaisesRegex(ValueError, "itself|disjoint"):
            sketch.merge(sketch)
        self.assertEqual(sketch.to_bytes(), before)

    def test_confidence_metadata_states_only_a_probabilistic_cdf_bound(self) -> None:
        sketch = PriorityRowSketch(capacity=4, seed=1, width=2)
        sketch.update("a", [1.0, math.nan])
        sketch.update("b", [2.0, math.nan])
        metadata = sketch.confidence_metadata(confidence=0.95)

        expected = math.sqrt(math.log(2.0 / 0.05) / (2.0 * 2))
        self.assertAlmostEqual(metadata["columns"][0]["cdf_supremum_bound"], expected)
        self.assertEqual(metadata["columns"][0]["retained_finite_samples"], 2)
        self.assertIsNone(metadata["columns"][1]["cdf_supremum_bound"])
        self.assertIsNone(metadata["deterministic_value_error_bound"])
        self.assertIn("probability", metadata["semantics"])
        self.assertFalse(
            metadata["deduplication"]["supports_exact_overlap_detection"]
        )
        self.assertTrue(
            metadata["deduplication"]["globally_unique_row_uid_required"]
        )
        with self.assertRaisesRegex(ValueError, "confidence"):
            sketch.confidence_metadata(confidence=1.0)

    def test_priority_generation_does_not_use_python_hash(self) -> None:
        sketch = PriorityRowSketch(capacity=2, seed=1, width=1)
        with patch.object(builtins, "hash", side_effect=AssertionError("python hash")):
            sketch.update("safe", [1.0])
        self.assertEqual(sketch.total_rows, 1)


if __name__ == "__main__":
    unittest.main()
