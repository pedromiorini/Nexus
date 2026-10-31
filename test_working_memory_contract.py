"""Contratos da working memory: capacidade, clamp e extração strangler.

Critérios WM1–WM6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit dcfa180) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability working_memory.
"""

import copy
import operator
import os
import pickle
import subprocess
import sys
import unittest
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
from pathlib import Path

from core.constitutional_brain import CentralExecutive, PhonologicalLoop, VisuospatialSketchpad

ROOT = Path(__file__).resolve().parent
_NAMES = (
    "WorkingMemoryItem", "RehearsalEvent", "PhonologicalLoop",
    "VisuospatialSketchpad", "CentralExecutive", "RealWorkingMemorySystem",
)
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.working_memory as mod
ex = mod.CentralExecutive()
assert ex.allocate_attention("planning", strength=2.0)["strength"] == 1.0
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.working_memory as wm
    return wm


class SlottedPhonologicalLoop(PhonologicalLoop):
    __slots__ = ("label",)


class SlottedVisuospatialSketchpad(VisuospatialSketchpad):
    __slots__ = ("label",)


class SlotCapacityPhonologicalLoop(PhonologicalLoop):
    __slots__ = ("_capacity", "label")


class SlotCapacityVisuospatialSketchpad(VisuospatialSketchpad):
    __slots__ = ("_capacity", "label")


class WorkingMemoryContractTests(unittest.TestCase):
    def test_attention_allocation_is_recorded_and_clamped(self):
        executive = CentralExecutive()
        allocation = executive.allocate_attention("planning", strength=2.0)
        self.assertEqual(allocation, {"target": "planning", "strength": 1.0, "delegated": False})
        self.assertEqual(executive.current_task, "planning")
        self.assertEqual(executive.attention_allocations, [allocation])

    def test_external_attention_presence_is_reported_without_unknown_call(self):
        executive = CentralExecutive(attention_system=object())
        allocation = executive.allocate_attention("monitoring", strength=-1.0)
        self.assertEqual(allocation["strength"], 0.0)
        self.assertTrue(allocation["delegated"])
        self.assertEqual(executive.current_task, "monitoring")


class WorkingMemoryCapacityContractTests(unittest.TestCase):
    def test_defaults_and_positive_integer_capacities_are_accepted(self):
        wm = _module()
        self.assertEqual(wm.PhonologicalLoop().capacity, 7)
        self.assertEqual(wm.VisuospatialSketchpad().capacity, 4)
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            for capacity in (1, 3):
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    self.assertEqual(constructor(capacity=capacity).capacity, capacity)

    def test_indexable_capacities_are_accepted_and_normalized_to_builtin_int(self):
        wm = _module()

        class IntSubclass(int):
            pass

        class CapacityEnum(IntEnum):
            ONE = 1

        class IndexLike:
            def __init__(self, value=3):
                self.value = value

            def __index__(self):
                return self.value

        accepted = (
            (IntSubclass(2), 2),
            (CapacityEnum.ONE, 1),
            (IndexLike(), 3),
        )
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            for capacity, expected in accepted:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    instance = constructor(capacity=capacity)
                    self.assertEqual(instance.capacity, expected)
                    self.assertIs(type(instance.capacity), int)

    def test_numpy_int64_capacity_is_accepted_and_normalized_to_builtin_int(self):
        wm = _module()
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=np.int64(4))
                self.assertEqual(instance.capacity, 4)
                self.assertIs(type(instance.capacity), int)

    def test_bool_and_non_indexable_capacity_types_are_rejected_by_both_buffers(self):
        wm = _module()

        class BadIndex:
            def __index__(self):
                return 1.0

        invalid_types = (
            True, False, 1.0, float("nan"), float("inf"), "1", None,
            Fraction(1, 1), Decimal(1), BadIndex(),
        )
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            for capacity in invalid_types:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    with self.assertRaises(TypeError):
                        constructor(capacity=capacity)

    def test_numpy_bool_capacity_follows_the_installed_index_protocol(self):
        wm = _module()
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")

        for capacity in (np.bool_(True), np.bool_(False)):
            with self.subTest(capacity=capacity):
                try:
                    expected = operator.index(capacity)
                except TypeError:
                    for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
                        with self.subTest(constructor=constructor.__name__):
                            with self.assertRaises(TypeError):
                                constructor(capacity=capacity)
                else:
                    for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
                        with self.subTest(constructor=constructor.__name__):
                            if expected <= 0:
                                with self.assertRaises(ValueError):
                                    constructor(capacity=capacity)
                            else:
                                instance = constructor(capacity=capacity)
                                self.assertEqual(instance.capacity, expected)
                                self.assertIs(type(instance.capacity), int)

    def test_zero_and_negative_capacities_are_rejected_at_construction(self):
        wm = _module()

        class IndexValue:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            for capacity in (0, -1, -7, IndexValue(0), IndexValue(-2)):
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    with self.assertRaises(ValueError):
                        constructor(capacity=capacity)

    def test_capacity_can_be_reassigned_to_positive_indexable_value(self):
        wm = _module()

        class IndexLike:
            def __index__(self):
                return 3

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=2)
                instance.capacity = 6
                self.assertEqual(instance.capacity, 6)
                self.assertIs(type(instance.capacity), int)
                instance.capacity = IndexLike()
                self.assertEqual(instance.capacity, 3)
                self.assertIs(type(instance.capacity), int)

    def test_invalid_capacity_reassignment_is_rejected_without_mutation(self):
        wm = _module()
        invalid = (
            (True, TypeError),
            (False, TypeError),
            (1.0, TypeError),
            (0, ValueError),
            (-1, ValueError),
        )

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=4)
                if isinstance(instance, wm.PhonologicalLoop):
                    instance.store("existing")
                else:
                    instance.store_visual("existing")
                original_buffer = tuple(instance.buffer)

                for capacity, error in invalid:
                    with self.subTest(capacity=capacity):
                        with self.assertRaises(error):
                            instance.capacity = capacity
                        self.assertEqual(instance.capacity, 4)
                        self.assertIs(type(instance.capacity), int)
                        self.assertEqual(tuple(instance.buffer), original_buffer)

    def test_capacity_reassignment_propagates_index_error_atomically(self):
        wm = _module()

        class CapacityError(RuntimeError):
            pass

        class RaisingIndex:
            def __init__(self, error):
                self.error = error

            def __index__(self):
                raise self.error

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=4)
                if isinstance(instance, wm.PhonologicalLoop):
                    instance.store("existing")
                else:
                    instance.store_visual("existing")
                original_capacity = instance.capacity
                original_buffer = instance.buffer
                original_items = tuple(original_buffer)
                expected_error = CapacityError("custom __index__ failure")

                with self.assertRaises(CapacityError) as raised:
                    instance.capacity = RaisingIndex(expected_error)

                self.assertIs(raised.exception, expected_error)
                self.assertEqual(instance.capacity, original_capacity)
                self.assertIs(type(instance.capacity), int)
                self.assertIs(instance.buffer, original_buffer)
                self.assertEqual(tuple(instance.buffer), original_items)
                for current, original in zip(instance.buffer, original_items, strict=True):
                    self.assertIs(current, original)

    def test_decreasing_capacity_does_not_eagerly_truncate_existing_buffer(self):
        wm = _module()

        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=3)
                if isinstance(instance, wm.PhonologicalLoop):
                    instance.store("first")
                    instance.store("second")
                else:
                    instance.store_visual("first")
                    instance.store_visual("second")
                original_buffer = tuple(instance.buffer)

                instance.capacity = 1

                self.assertEqual(instance.capacity, 1)
                self.assertEqual(len(instance.buffer), 2)
                for current, original in zip(instance.buffer, original_buffer, strict=True):
                    self.assertIs(current, original)

    def test_positive_capacities_preserve_each_buffer_eviction_policy(self):
        wm = _module()
        with self.subTest(method="phonological_loop"):
            loop = wm.PhonologicalLoop(capacity=2)
            first_inserted = loop.store("first inserted")
            second_inserted = loop.store("second inserted")
            first_inserted.last_rehearsed = 2.0
            second_inserted.last_rehearsed = 1.0
            latest = loop.store("latest")
            self.assertEqual(len(loop.buffer), 2)
            self.assertIn(first_inserted, loop.buffer)
            self.assertNotIn(second_inserted, loop.buffer)
            self.assertIn(latest, loop.buffer)

        for method_name in ("store_visual", "store_spatial"):
            with self.subTest(method=method_name):
                pad = wm.VisuospatialSketchpad(capacity=2)
                store = getattr(pad, method_name)
                first_inserted = store("first inserted")
                second_inserted = store("second inserted")
                first_inserted.activation = 0.9
                second_inserted.activation = 0.1
                latest = store("latest")
                self.assertEqual(len(pad.buffer), 2)
                self.assertIn(first_inserted, pad.buffer)
                self.assertNotIn(second_inserted, pad.buffer)
                self.assertIn(latest, pad.buffer)

    def test_legacy_pickle_state_migrates_capacity_and_preserves_aliases(self):
        wm = _module()
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=3)
                if isinstance(instance, wm.PhonologicalLoop):
                    item = instance.store("legacy item")
                else:
                    item = instance.store_visual("legacy item")
                instance.buffer_alias = instance.buffer
                instance.item_alias = item
                instance.metadata = {"origin": "legacy"}

                instance.__dict__["capacity"] = instance.__dict__.pop("_capacity")
                restored = pickle.loads(pickle.dumps(instance))

                self.assertEqual(restored.capacity, 3)
                self.assertEqual(restored.__dict__["_capacity"], 3)
                self.assertNotIn("capacity", restored.__dict__)
                self.assertIs(restored.buffer, restored.buffer_alias)
                self.assertIs(restored.buffer[0], restored.item_alias)
                self.assertEqual(restored.metadata, {"origin": "legacy"})

    def test_current_pickle_and_deepcopy_preserve_state_and_aliases(self):
        wm = _module()
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=3)
                if isinstance(instance, wm.PhonologicalLoop):
                    item = instance.store("current item")
                else:
                    item = instance.store_visual("current item")
                instance.buffer_alias = instance.buffer
                instance.item_alias = item
                instance.metadata = {"origin": "current"}

                clones = (
                    pickle.loads(pickle.dumps(instance)),
                    copy.deepcopy(instance),
                )
                for clone in clones:
                    self.assertEqual(clone.capacity, 3)
                    self.assertEqual(clone.__dict__["_capacity"], 3)
                    self.assertNotIn("capacity", clone.__dict__)
                    self.assertIs(clone.buffer, clone.buffer_alias)
                    self.assertIs(clone.buffer[0], clone.item_alias)
                    self.assertEqual(clone.metadata, {"origin": "current"})

    def test_slotted_subclass_pickle_and_deepcopy_preserve_dict_slot_aliases(self):
        constructors = (SlottedPhonologicalLoop, SlottedVisuospatialSketchpad)
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=3)
                if isinstance(instance, PhonologicalLoop):
                    item = instance.store("slotted current item")
                else:
                    item = instance.store_visual("slotted current item")
                shared = {"origin": "dict-and-slot"}
                instance.buffer_alias = instance.buffer
                instance.item_alias = item
                instance.metadata = shared
                instance.label = shared

                state = instance.__reduce_ex__(pickle.HIGHEST_PROTOCOL)[2]
                self.assertIs(type(state), tuple)
                self.assertEqual(len(state), 2)
                self.assertIs(state[0]["metadata"], state[1]["label"])

                clones = (
                    pickle.loads(pickle.dumps(instance)),
                    copy.deepcopy(instance),
                )
                for clone in clones:
                    self.assertEqual(clone.capacity, 3)
                    self.assertEqual(clone.__dict__["_capacity"], 3)
                    self.assertNotIn("capacity", clone.__dict__)
                    self.assertIs(clone.buffer, clone.buffer_alias)
                    self.assertIs(clone.buffer[0], clone.item_alias)
                    self.assertEqual(clone.metadata, {"origin": "dict-and-slot"})
                    self.assertIs(clone.label, clone.metadata)

    def test_slotted_legacy_pickle_state_migrates_capacity_and_preserves_slots(self):
        constructors = (SlottedPhonologicalLoop, SlottedVisuospatialSketchpad)
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=3)
                if isinstance(instance, PhonologicalLoop):
                    item = instance.store("slotted legacy item")
                else:
                    item = instance.store_visual("slotted legacy item")
                shared = {"origin": "legacy-dict-and-slot"}
                instance.buffer_alias = instance.buffer
                instance.item_alias = item
                instance.metadata = shared
                instance.label = shared
                instance.__dict__["capacity"] = instance.__dict__.pop("_capacity")

                state = instance.__reduce_ex__(pickle.HIGHEST_PROTOCOL)[2]
                self.assertIs(type(state), tuple)
                self.assertIn("capacity", state[0])
                self.assertNotIn("_capacity", state[0])
                self.assertIs(state[0]["metadata"], state[1]["label"])

                clones = (
                    pickle.loads(pickle.dumps(instance)),
                    copy.deepcopy(instance),
                )
                for clone in clones:
                    self.assertEqual(clone.capacity, 3)
                    self.assertEqual(clone.__dict__["_capacity"], 3)
                    self.assertNotIn("capacity", clone.__dict__)
                    self.assertEqual(clone.metadata, {"origin": "legacy-dict-and-slot"})
                    self.assertIs(clone.label, clone.metadata)
                    self.assertIs(clone.buffer, clone.buffer_alias)
                    self.assertIs(clone.buffer[0], clone.item_alias)

    def test_invalid_legacy_capacity_in_tuple_state_is_atomic_including_slots(self):
        invalid = (
            (True, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            ("3", TypeError),
        )
        constructors = (SlottedPhonologicalLoop, SlottedVisuospatialSketchpad)
        for constructor in constructors:
            for capacity, error in invalid:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    instance = constructor(capacity=2)
                    instance.label = "original slot value"
                    original_state = instance.__dict__.copy()
                    incoming_state = (
                        {
                            "capacity": capacity,
                            "buffer": ["incoming state must not be accepted"],
                            "extra": "incoming state must not be accepted",
                        },
                        {"label": "incoming slot value must not be accepted"},
                    )

                    with self.assertRaises(error):
                        instance.__setstate__(incoming_state)

                    self.assertEqual(instance.__dict__, original_state)
                    self.assertIs(instance.buffer, original_state["buffer"])
                    self.assertNotIn("extra", instance.__dict__)
                    self.assertEqual(instance.label, "original slot value")

    def test_malformed_tuple_pickle_state_is_rejected_without_mutation(self):
        instance = SlottedPhonologicalLoop(capacity=2)
        instance.label = "original slot value"
        original_state = instance.__dict__.copy()
        malformed_states = (
            ({"capacity": 3},),
            ([], {"label": "incoming"}),
            ({"capacity": 3}, []),
        )
        for state in malformed_states:
            with self.subTest(state=state):
                with self.assertRaisesRegex(TypeError, "working memory state"):
                    instance.__setstate__(state)
                self.assertEqual(instance.__dict__, original_state)
                self.assertEqual(instance.label, "original slot value")

    def test_non_string_dict_state_key_is_rejected_before_mutation(self):
        for constructor in (PhonologicalLoop, VisuospatialSketchpad):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=2)
                original_state = instance.__dict__.copy()
                original_buffer = instance.buffer
                incoming_state = {
                    "capacity": 3,
                    "buffer": ["incoming"],
                    "metadata": "incoming",
                    1: "non-string key",
                }

                with self.assertRaisesRegex(TypeError, "dict state keys must be strings"):
                    instance.__setstate__(incoming_state)

                self.assertEqual(instance.__dict__, original_state)
                self.assertIs(instance.buffer, original_buffer)
                self.assertEqual(instance.capacity, 2)
                self.assertNotIn("metadata", instance.__dict__)

    def test_invalid_slot_state_keys_are_rejected_before_mutation(self):
        constructors = (SlottedPhonologicalLoop, SlottedVisuospatialSketchpad)
        invalid_slot_states = (
            ("non-string key", {"label": "incoming", 1: "non-string key"}, "slot state keys must be strings"),
            ("unsupported slot", {"label": "incoming", "unsupported": "value"}, "unsupported slot"),
        )
        for constructor in constructors:
            for case, slot_state, error_message in invalid_slot_states:
                with self.subTest(constructor=constructor.__name__, case=case):
                    instance = constructor(capacity=2)
                    instance.label = "original slot value"
                    original_state = instance.__dict__.copy()
                    original_buffer = instance.buffer
                    incoming_state = (
                        {
                            "capacity": 3,
                            "buffer": ["incoming"],
                            "metadata": "incoming",
                        },
                        slot_state,
                    )

                    with self.assertRaisesRegex(TypeError, error_message):
                        instance.__setstate__(incoming_state)

                    self.assertEqual(instance.__dict__, original_state)
                    self.assertIs(instance.buffer, original_buffer)
                    self.assertEqual(instance.capacity, 2)
                    self.assertEqual(instance.label, "original slot value")
                    self.assertNotIn("metadata", instance.__dict__)

    def test_invalid_legacy_pickle_capacity_is_rejected_before_state_update(self):
        wm = _module()
        invalid = (
            (True, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            ("3", TypeError),
        )
        for constructor in (wm.PhonologicalLoop, wm.VisuospatialSketchpad):
            for capacity, error in invalid:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    instance = constructor(capacity=2)
                    original_state = instance.__dict__.copy()
                    incoming_state = {
                        "capacity": capacity,
                        "buffer": ["incoming state must not be accepted"],
                        "extra": "incoming state must not be accepted",
                    }

                    with self.assertRaises(error):
                        instance.__setstate__(incoming_state)

                    self.assertEqual(instance.__dict__, original_state)
                    self.assertIs(instance.buffer, original_state["buffer"])
                    self.assertNotIn("extra", instance.__dict__)

                    legacy = constructor(capacity=2)
                    legacy.__dict__["capacity"] = legacy.__dict__.pop("_capacity")
                    legacy.__dict__["capacity"] = capacity
                    payload = pickle.dumps(legacy)
                    with self.assertRaises(error):
                        pickle.loads(payload)


    def test_current_pickle_capacity_state_is_validated_and_normalized(self):
        class IndexLike:
            def __index__(self):
                return 5

        constructors = (
            PhonologicalLoop,
            VisuospatialSketchpad,
            SlottedPhonologicalLoop,
            SlottedVisuospatialSketchpad,
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=2)
                incoming_dict = {"_capacity": IndexLike(), "metadata": "restored"}
                if hasattr(type(instance), "label"):
                    instance.__setstate__((incoming_dict, {"label": "restored slot"}))
                    self.assertEqual(instance.label, "restored slot")
                else:
                    instance.__setstate__(incoming_dict)

                self.assertEqual(instance.capacity, 5)
                self.assertIs(type(instance.__dict__["_capacity"]), int)
                self.assertEqual(instance.metadata, "restored")

    def test_state_without_capacity_is_rejected_before_dict_or_slot_update(self):
        dict_constructors = (PhonologicalLoop, VisuospatialSketchpad)
        for constructor in dict_constructors:
            with self.subTest(constructor=constructor.__name__, state="dict"):
                instance = constructor(capacity=2)
                original_state = instance.__dict__.copy()
                original_buffer = instance.buffer
                incoming_state = {
                    "buffer": ["incoming state must not be accepted"],
                    "extra": "incoming state must not be accepted",
                }

                with self.assertRaisesRegex(ValueError, "state must include capacity"):
                    instance.__setstate__(incoming_state)

                self.assertEqual(instance.__dict__, original_state)
                self.assertIs(instance.buffer, original_buffer)
                self.assertEqual(instance.capacity, 2)

        slot_constructors = (
            SlotCapacityPhonologicalLoop,
            SlotCapacityVisuospatialSketchpad,
        )
        for constructor in slot_constructors:
            with self.subTest(constructor=constructor.__name__, state="tuple"):
                instance = constructor(capacity=2)
                instance.label = "original slot value"
                original_state = instance.__dict__.copy()
                original_buffer = instance.buffer
                incoming_state = (
                    {
                        "buffer": ["incoming state must not be accepted"],
                        "extra": "incoming state must not be accepted",
                    },
                    {"label": "incoming slot value must not be accepted"},
                )

                with self.assertRaisesRegex(ValueError, "state must include capacity"):
                    instance.__setstate__(incoming_state)

                self.assertEqual(instance.__dict__, original_state)
                self.assertIs(instance.buffer, original_buffer)
                self.assertEqual(instance.capacity, 2)
                self.assertEqual(instance.label, "original slot value")

    def test_invalid_current_pickle_capacity_is_rejected_before_dict_or_slot_update(self):
        invalid = (
            (True, TypeError),
            (False, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            (object(), TypeError),
        )
        constructors = (
            PhonologicalLoop,
            VisuospatialSketchpad,
            SlottedPhonologicalLoop,
            SlottedVisuospatialSketchpad,
        )
        for constructor in constructors:
            for capacity, error in invalid:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    instance = constructor(capacity=2)
                    original_state = instance.__dict__.copy()
                    original_buffer = instance.buffer
                    has_label = hasattr(type(instance), "label")
                    if has_label:
                        instance.label = "original slot value"
                        original_state = instance.__dict__.copy()
                        state = (
                            {
                                "_capacity": capacity,
                                "buffer": ["incoming"],
                                "extra": "incoming",
                            },
                            {"label": "incoming slot value"},
                        )
                    else:
                        state = {
                            "_capacity": capacity,
                            "buffer": ["incoming"],
                            "extra": "incoming",
                        }

                    with self.assertRaises(error):
                        instance.__setstate__(state)

                    self.assertEqual(instance.__dict__, original_state)
                    self.assertIs(instance.buffer, original_buffer)
                    self.assertNotIn("extra", instance.__dict__)
                    if has_label:
                        self.assertEqual(instance.label, "original slot value")

    def test_legacy_and_current_pickle_capacity_must_normalize_consistently(self):
        class IndexLike:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        constructors = (
            PhonologicalLoop,
            VisuospatialSketchpad,
            SlottedPhonologicalLoop,
            SlottedVisuospatialSketchpad,
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__, state="consistent"):
                instance = constructor(capacity=2)
                incoming_dict = {
                    "capacity": IndexLike(5),
                    "_capacity": 5,
                    "metadata": "restored",
                }
                if hasattr(type(instance), "label"):
                    instance.__setstate__((incoming_dict, {"label": "restored slot"}))
                    self.assertEqual(instance.label, "restored slot")
                else:
                    instance.__setstate__(incoming_dict)

                self.assertEqual(instance.capacity, 5)
                self.assertIs(type(instance.__dict__["_capacity"]), int)
                self.assertNotIn("capacity", instance.__dict__)
                self.assertEqual(instance.metadata, "restored")

            with self.subTest(constructor=constructor.__name__, state="conflict"):
                instance = constructor(capacity=2)
                if hasattr(type(instance), "label"):
                    instance.label = "original slot value"
                original_state = instance.__dict__.copy()
                original_buffer = instance.buffer
                incoming_dict = {
                    "capacity": 4,
                    "_capacity": 5,
                    "buffer": ["incoming"],
                    "extra": "incoming",
                }
                original_incoming = incoming_dict.copy()
                if hasattr(type(instance), "label"):
                    state = (incoming_dict, {"label": "incoming slot value"})
                else:
                    state = incoming_dict

                with self.assertRaises(ValueError):
                    instance.__setstate__(state)

                self.assertEqual(instance.__dict__, original_state)
                self.assertIs(instance.buffer, original_buffer)
                self.assertEqual(incoming_dict, original_incoming)
                self.assertNotIn("extra", instance.__dict__)
                if hasattr(type(instance), "label"):
                    self.assertEqual(instance.label, "original slot value")


    def test_slot_backed_current_capacity_is_validated_and_normalized(self):
        class IndexLike:
            def __index__(self):
                return 5

        for constructor in (
            SlotCapacityPhonologicalLoop,
            SlotCapacityVisuospatialSketchpad,
        ):
            with self.subTest(constructor=constructor.__name__):
                instance = constructor(capacity=2)
                instance.label = "original slot value"
                pickle_state = instance.__reduce_ex__(pickle.HIGHEST_PROTOCOL)[2]
                self.assertIs(type(pickle_state), tuple)
                self.assertIn("_capacity", pickle_state[1])

                instance.__setstate__(
                    (
                        {"buffer": ["restored"], "metadata": "restored"},
                        {"_capacity": IndexLike(), "label": "restored slot"},
                    )
                )

                self.assertEqual(instance.capacity, 5)
                self.assertIs(type(instance.capacity), int)
                self.assertNotIn("_capacity", instance.__dict__)
                self.assertEqual(instance.metadata, "restored")
                self.assertEqual(instance.label, "restored slot")
                clones = (
                    pickle.loads(pickle.dumps(instance)),
                    copy.deepcopy(instance),
                )
                for clone in clones:
                    self.assertEqual(clone.capacity, 5)
                    self.assertIs(type(clone.capacity), int)
                    self.assertNotIn("_capacity", clone.__dict__)
                    self.assertEqual(clone.metadata, "restored")
                    self.assertEqual(clone.label, "restored slot")

    def test_invalid_slot_backed_capacity_is_rejected_before_dict_or_slot_update(self):
        invalid = (
            (True, TypeError),
            (False, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            (object(), TypeError),
        )
        for constructor in (
            SlotCapacityPhonologicalLoop,
            SlotCapacityVisuospatialSketchpad,
        ):
            for capacity, error in invalid:
                with self.subTest(constructor=constructor.__name__, capacity=capacity):
                    instance = constructor(capacity=2)
                    instance.label = "original slot value"
                    original_state = instance.__dict__.copy()
                    original_buffer = instance.buffer

                    with self.assertRaises(error):
                        instance.__setstate__(
                            (
                                {"buffer": ["incoming"], "extra": "incoming"},
                                {"_capacity": capacity, "label": "incoming slot"},
                            )
                        )

                    self.assertEqual(instance.__dict__, original_state)
                    self.assertIs(instance.buffer, original_buffer)
                    self.assertEqual(instance.capacity, 2)
                    self.assertIs(type(instance.capacity), int)
                    self.assertEqual(instance.label, "original slot value")

    def test_legacy_capacity_must_match_slot_backed_current_capacity(self):
        class IndexLike:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        constructors = (
            SlotCapacityPhonologicalLoop,
            SlotCapacityVisuospatialSketchpad,
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__, state="consistent"):
                instance = constructor(capacity=2)
                instance.__setstate__(
                    (
                        {"capacity": IndexLike(5), "metadata": "restored"},
                        {"_capacity": 5, "label": "restored slot"},
                    )
                )
                self.assertEqual(instance.capacity, 5)
                self.assertIs(type(instance.capacity), int)
                self.assertNotIn("capacity", instance.__dict__)
                self.assertNotIn("_capacity", instance.__dict__)
                self.assertEqual(instance.metadata, "restored")
                self.assertEqual(instance.label, "restored slot")

            with self.subTest(constructor=constructor.__name__, state="legacy_only"):
                instance = constructor(capacity=2)
                instance.__setstate__(
                    ({"capacity": IndexLike(6), "metadata": "legacy"}, {"label": "legacy slot"})
                )
                self.assertEqual(instance.capacity, 6)
                self.assertIs(type(instance.capacity), int)
                self.assertNotIn("capacity", instance.__dict__)
                self.assertNotIn("_capacity", instance.__dict__)
                self.assertEqual(instance.metadata, "legacy")
                self.assertEqual(instance.label, "legacy slot")

            with self.subTest(constructor=constructor.__name__, state="conflict"):
                instance = constructor(capacity=2)
                instance.label = "original slot value"
                original_state = instance.__dict__.copy()
                original_capacity = instance.capacity
                incoming_dict = {"capacity": 4, "extra": "incoming"}
                original_incoming = incoming_dict.copy()

                with self.assertRaises(ValueError):
                    instance.__setstate__(
                        (incoming_dict, {"_capacity": 5, "label": "incoming slot"})
                    )

                self.assertEqual(instance.__dict__, original_state)
                self.assertEqual(incoming_dict, original_incoming)
                self.assertEqual(instance.capacity, original_capacity)
                self.assertEqual(instance.label, "original slot value")


class WorkingMemoryExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        wm = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(wm, name))
                self.assertIs(getattr(package, name), getattr(wm, name))
                self.assertEqual(getattr(wm, name).__module__, "core.cognitivo.working_memory")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import WM_GOLDEN_SHA, eval_working_memory
        checks = {item["id"]: item for item in eval_working_memory()}
        self.assertEqual(set(checks), {"WM1", "WM2", "WM3", "WM4", "WM5", "WM6"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)
        self.assertEqual(checks["WM6"]["value"]["sha_module"], WM_GOLDEN_SHA)
        self.assertEqual(checks["WM6"]["value"]["sha_brain"], WM_GOLDEN_SHA)
        self.assertEqual(checks["WM3"]["value"]["clock"], "wall")
        self.assertLess(checks["WM3"]["value"]["kept_activation"], 1.0)
        self.assertGreater(checks["WM3"]["value"]["kept_activation"], 0.1)

    def test_rehearse_verbal_counts_only_a_hit(self):
        import contextlib
        import io
        wm = _module()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            system = wm.RealWorkingMemorySystem()
            system.rehearse_verbal("missing")
            self.assertEqual(system.rehearsal_count, 0)
            item = system.store_verbal("keep")
            system.rehearse_verbal(item.item_id)
            self.assertEqual(system.rehearsal_count, 1)

    def test_invalid_attention_strength_is_not_coerced_from_text_or_container(self):
        executive = CentralExecutive()
        with self.assertRaises(ValueError):
            executive.allocate_attention("t", strength="abc")
        for bad in (None, [], {}):
            with self.subTest(strength=bad):
                with self.assertRaises(TypeError):
                    executive.allocate_attention("t", strength=bad)


if __name__ == "__main__":
    unittest.main()
