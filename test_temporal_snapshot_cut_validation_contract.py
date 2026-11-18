"""Contratos para rejeitar sobreposição funcional observável em cortes históricos."""
import copy
import hashlib
import json
import unittest

from core.temporal_facts import GENESIS, SCHEMA, TemporalFactStore


def _assert_event(number, recorded, start, end=None, *, subject="ana", predicate="mora_em"):
    return {
        "kind": "assert",
        "recorded_at": recorded,
        "fact": {
            "fact_id": f"tf-{number:06d}",
            "subject": subject,
            "predicate": predicate,
            "object": str(number),
            "valid_from": start,
            "valid_to": end,
            "recorded_at": recorded,
            "confidence": 0.9,
            "source": "fixture",
            "superseded_by": None,
            "metadata": {"case": number},
        },
    }


def _close(number, recorded, valid_to):
    return {
        "kind": "close",
        "recorded_at": recorded,
        "fact_id": f"tf-{number:06d}",
        "valid_to": valid_to,
        "superseded_by": None,
    }


def _seal(events):
    events = copy.deepcopy(events)
    previous = GENESIS
    for sequence, event in enumerate(events, 1):
        event["seq"] = sequence
        event["prev"] = previous
        body = json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":"))
        previous = hashlib.sha256(body.encode()).hexdigest()
        event["hash"] = previous
    return {
        "schema": SCHEMA,
        "functional_predicates": ["mora_em"],
        "head": previous,
        "events": events,
    }


class TemporalSnapshotCutValidationContract(unittest.TestCase):
    def _restore(self, events, *, rejected):
        payload = _seal(events)
        original = copy.deepcopy(payload)
        for as_json in (False, True):
            with self.subTest(as_json=as_json):
                value = json.dumps(payload) if as_json else payload
                if rejected:
                    with self.assertRaisesRegex(ValueError, "sobrepost"):
                        TemporalFactStore.from_snapshot(value, expected_head=payload["head"])
                else:
                    restored = TemporalFactStore.from_snapshot(
                        value, expected_head=payload["head"]
                    )
                    self.assertEqual(restored.export_snapshot(), payload)
                self.assertEqual(payload, original)

    def test_later_close_cannot_hide_functional_overlap(self):
        self._restore(
            [
                _assert_event(1, 10, 0),
                _assert_event(2, 20, 5),
                _close(1, 30, 5),
            ],
            rejected=True,
        )

    def test_right_neighbor_overlap_is_rejected_at_its_observed_cut(self):
        self._restore(
            [
                _assert_event(1, 10, 10),
                _assert_event(2, 20, 5),
                _close(2, 30, 10),
            ],
            rejected=True,
        )

    def test_terminal_overlap_is_not_accepted(self):
        self._restore(
            [_assert_event(1, 10, 0), _assert_event(2, 20, 5)],
            rejected=True,
        )

    def test_unrelated_functional_key_does_not_mask_overlap(self):
        self._restore(
            [
                _assert_event(1, 10, 0),
                _assert_event(2, 20, 5),
                _assert_event(3, 20, 0, subject="bia"),
                _close(1, 30, 5),
            ],
            rejected=True,
        )

    def test_same_timestamp_close_and_assert_are_one_observable_cut(self):
        for tail in (
            (_close(1, 20, 5), _assert_event(2, 20, 5)),
            (_assert_event(2, 20, 5), _close(1, 20, 5)),
        ):
            with self.subTest(order=[event["kind"] for event in tail]):
                self._restore([_assert_event(1, 10, 0), *tail], rejected=False)

    def test_adjacent_half_open_intervals_are_valid(self):
        self._restore(
            [_assert_event(1, 10, 0, 5), _assert_event(2, 20, 5)],
            rejected=False,
        )


if __name__ == "__main__":
    unittest.main()
