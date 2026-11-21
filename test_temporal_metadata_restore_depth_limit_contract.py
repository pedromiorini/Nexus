"""Contrato para impor o limite de profundidade também ao restaurar snapshots."""
import copy
import hashlib
import json
import unittest

from core.temporal_facts import GENESIS, TemporalFactStore, _check_metadata_depth, _digest

SUPPORTED_DEPTH = 128


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class OverridingDictSubclass(dict):
    def copy(self):
        raise AssertionError("subclass copy must not run")

    def values(self):
        raise AssertionError("subclass values must not run")


class OverridingListSubclass(list):
    def copy(self):
        raise AssertionError("subclass copy must not run")

    def __iter__(self):
        raise AssertionError("subclass iterator must not run")


class DigestTrapDictSubclass(dict):
    override_calls = 0

    def items(self):
        type(self).override_calls += 1
        raise RuntimeError("subclass items must not run before depth validation")


class DigestTrapListSubclass(list):
    override_calls = 0

    def __iter__(self):
        type(self).override_calls += 1
        raise RuntimeError("subclass iterator must not run before depth validation")


class ShallowTrapDictSubclass(dict):
    override_calls = 0

    def items(self):
        type(self).override_calls += 1
        raise RuntimeError("shallow subclass items must not run during restore")


class ShallowTrapListSubclass(list):
    override_calls = 0

    def __iter__(self):
        type(self).override_calls += 1
        raise RuntimeError("shallow subclass iterator must not run during restore")


def nested_dict(levels):
    value = {}
    for _ in range(levels - 1):
        value = {"k": value}
    return value


def alternating_containers(levels):
    value = None
    for index in range(levels):
        value = {"k": value} if index % 2 == 0 else [value]
    return value


def snapshot_with_metadata(metadata):
    store = TemporalFactStore(clock=lambda: 100.0)
    store.assert_fact("ana", "lives_in", "A", valid_from=0.0)
    snapshot = store.export_snapshot()
    snapshot["events"][0]["fact"]["metadata"] = metadata

    previous = GENESIS
    for sequence, event in enumerate(snapshot["events"], 1):
        event["seq"] = sequence
        event["prev"] = previous
        event["hash"] = _digest(event)
        previous = event["hash"]
    snapshot["head"] = previous
    return snapshot


class TemporalMetadataRestoreDepthLimitContract(unittest.TestCase):
    def test_deeper_than_supported_metadata_is_rejected_for_dict_and_json_snapshots(self):
        payload_shapes = (
            ("dicts", nested_dict(SUPPORTED_DEPTH + 1)),
            ("mixed", alternating_containers(SUPPORTED_DEPTH + 1)),
            ("dict-subclass", DictSubclass({"k": nested_dict(SUPPORTED_DEPTH)})),
            ("list-subclass", {"k": ListSubclass([nested_dict(SUPPORTED_DEPTH - 1)])}),
        )

        for shape, metadata in payload_shapes:
            snapshot = snapshot_with_metadata(metadata)
            inputs = (
                ("dict", copy.deepcopy(snapshot)),
                ("json", json.dumps(snapshot, sort_keys=True, allow_nan=False)),
            )
            for snapshot_type, payload in inputs:
                with self.subTest(shape=shape, snapshot_type=snapshot_type):
                    with self.assertRaisesRegex(ValueError, "profundidade"):
                        TemporalFactStore.from_snapshot(payload)

    def test_from_snapshot_checks_depth_before_digest_calls_deep_subclass_overrides(self):
        metadata_by_shape = (
            ("dict", nested_dict(SUPPORTED_DEPTH + 1)),
            ("list", {"k": [nested_dict(SUPPORTED_DEPTH - 1)]}),
        )

        for shape, metadata in metadata_by_shape:
            with self.subTest(shape=shape):
                # Seal while the complete payload uses only native containers.
                snapshot = snapshot_with_metadata(metadata)
                fact = snapshot["events"][0]["fact"]
                if shape == "dict":
                    parent = fact["metadata"]
                    for _ in range(70):
                        parent = dict.__getitem__(parent, "k")
                    wrapped = DigestTrapDictSubclass(
                        dict.__getitem__(parent, "k")
                    )
                    dict.__setitem__(parent, "k", wrapped)
                    DigestTrapDictSubclass.override_calls = 0
                else:
                    original = dict.__getitem__(fact["metadata"], "k")
                    wrapped = DigestTrapListSubclass(list.copy(original))
                    dict.__setitem__(fact["metadata"], "k", wrapped)
                    DigestTrapListSubclass.override_calls = 0

                with self.assertRaisesRegex(ValueError, "profundidade"):
                    TemporalFactStore.from_snapshot(snapshot)

                if shape == "dict":
                    self.assertEqual(DigestTrapDictSubclass.override_calls, 0)
                else:
                    self.assertEqual(DigestTrapListSubclass.override_calls, 0)

    def test_depth_scan_uses_base_container_operations_for_subclasses(self):
        metadata_values = (
            ("dict", OverridingDictSubclass({"k": nested_dict(SUPPORTED_DEPTH)})),
            ("list", {"k": OverridingListSubclass([nested_dict(SUPPORTED_DEPTH - 1)])}),
        )
        for shape, metadata in metadata_values:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "profundidade"):
                    _check_metadata_depth(metadata)

    def test_shallow_trap_subclasses_restore_without_overrides_and_keep_valid_hashes(self):
        metadata_by_shape = (
            ("dict", {"trap": {"k": "v"}}, ShallowTrapDictSubclass),
            ("list", {"trap": ["v"]}, ShallowTrapListSubclass),
        )

        for shape, native_metadata, trap_type in metadata_by_shape:
            with self.subTest(shape=shape):
                snapshot = snapshot_with_metadata(native_metadata)
                event = snapshot["events"][0]
                fact = event["fact"]
                legacy_body = json.dumps(
                    {key: value for key, value in event.items() if key != "hash"},
                    sort_keys=True, allow_nan=False, separators=(",", ":"),
                )
                expected_hash = hashlib.sha256(legacy_body.encode("utf-8")).hexdigest()
                self.assertEqual(event["hash"], expected_hash)
                if shape == "dict":
                    dict.__setitem__(fact, "metadata", trap_type({"trap": {"k": "v"}}))
                else:
                    metadata = dict.copy(fact["metadata"])
                    dict.__setitem__(metadata, "trap", trap_type(["v"]))
                    dict.__setitem__(fact, "metadata", metadata)
                trap_type.override_calls = 0

                self.assertEqual(_digest(event), expected_hash)
                self.assertEqual(trap_type.override_calls, 0)

                restored = TemporalFactStore.from_snapshot(snapshot)
                exported = restored.export_snapshot()

                self.assertEqual(restored.head_hash, expected_hash)
                self.assertEqual(exported["events"][0]["hash"], expected_hash)
                self.assertEqual(trap_type.override_calls, 0)
                restored_metadata = exported["events"][0]["fact"]["metadata"]
                self.assertIs(type(restored_metadata), dict)
                self.assertIs(type(restored_metadata["trap"]), dict if shape == "dict" else list)


if __name__ == "__main__":
    unittest.main()
