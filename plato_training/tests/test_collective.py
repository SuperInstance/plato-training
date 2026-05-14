"""
Tests for Collective Inference — predict, listen, gap, focus.
"""

import pytest
import time
from plato_training.collective import (
    RoomAddress, RoomKind, TMinusEvent, GapSignal, GapSeverity,
    FocusQueue, SimulationRoom,
)


class TestRoomAddress:
    def test_parse(self):
        addr = RoomAddress.parse("forgemaster@eileen/drift-detect/predictor")
        assert addr.instance == "forgemaster@eileen"
        assert addr.path == ["drift-detect", "predictor"]

    def test_child(self):
        addr = RoomAddress("oracle1@cloud", ["fleet-ops"])
        child = addr.child("sensor")
        assert str(child) == "oracle1@cloud/fleet-ops/sensor"

    def test_parent(self):
        addr = RoomAddress("test", ["a", "b", "c"])
        parent = addr.parent()
        assert parent.path == ["a", "b"]

    def test_parent_of_root(self):
        addr = RoomAddress("test", ["root"])
        assert addr.parent() is None


class TestTMinusEvent:
    def test_future_event(self):
        event = TMinusEvent(
            predictor="room1",
            event_type="drift",
            predicted_value=0.5,
            confidence=0.9,
            predicted_at=time.time(),
            event_time=time.time() + 60,
        )
        assert event.time_until_event > 0
        assert not event.is_expired

    def test_expired_event(self):
        event = TMinusEvent(
            predictor="room1",
            event_type="drift",
            predicted_value=0.5,
            confidence=0.9,
            predicted_at=time.time() - 120,
            event_time=time.time() - 60,
        )
        assert event.is_expired

    def test_round_trip(self):
        event = TMinusEvent(
            predictor="room1", event_type="test", predicted_value=42,
            confidence=0.8, predicted_at=100.0, event_time=160.0,
            context={"why": "pattern detected"},
        )
        d = event.to_dict()
        restored = TMinusEvent.from_dict(d)
        assert restored.predicted_value == 42
        assert restored.context["why"] == "pattern detected"


class TestGapSignal:
    def test_low_severity(self):
        event = TMinusEvent(
            predictor="r1", event_type="test", predicted_value=1.0,
            confidence=0.5, predicted_at=time.time(), event_time=time.time() + 10,
        )
        gap = GapSignal.create("r1", event, 1.05, delta=0.05)
        assert gap.severity == GapSeverity.LOW

    def test_critical_severity(self):
        event = TMinusEvent(
            predictor="r1", event_type="test", predicted_value=0.0,
            confidence=0.99, predicted_at=time.time(), event_time=time.time() + 10,
        )
        gap = GapSignal.create("r1", event, 1.0, delta=0.99)
        assert gap.severity == GapSeverity.CRITICAL

    def test_focus_score(self):
        """High confidence + large gap = high focus."""
        event1 = TMinusEvent(
            predictor="r1", event_type="test", predicted_value=0.0,
            confidence=0.99, predicted_at=time.time(), event_time=time.time() + 10,
        )
        event2 = TMinusEvent(
            predictor="r1", event_type="test", predicted_value=0.0,
            confidence=0.3, predicted_at=time.time(), event_time=time.time() + 10,
        )
        gap1 = GapSignal.create("r1", event1, 1.0, delta=0.99)
        gap2 = GapSignal.create("r1", event2, 1.0, delta=0.99)
        
        # "We were SURE and WRONG" > "We were UNSURE and WRONG"
        assert gap1.focus_score > gap2.focus_score


class TestFocusQueue:
    def test_sorted_by_focus(self):
        q = FocusQueue()
        
        e1 = TMinusEvent("r1", "t", 0, 0.3, time.time(), time.time()+10)
        e2 = TMinusEvent("r1", "t", 0, 0.9, time.time(), time.time()+10)
        
        q.add(GapSignal.create("r1", e1, 1, delta=0.8))
        q.add(GapSignal.create("r1", e2, 1, delta=0.9))
        
        top = q.top(2)
        assert top[0].focus_score > top[1].focus_score

    def test_by_severity(self):
        q = FocusQueue()
        
        for conf, delta in [(0.5, 0.05), (0.7, 0.6), (0.9, 0.99)]:
            e = TMinusEvent("r1", "t", 0, conf, time.time(), time.time()+10)
            q.add(GapSignal.create("r1", e, 1, delta=delta))
        
        critical = q.by_severity(GapSeverity.CRITICAL)
        assert len(critical) >= 1

    def test_summary(self):
        q = FocusQueue()
        s = q.summary()
        assert s["total_gaps"] == 0


class TestSimulationRoom:
    def test_predict_and_observe_match(self):
        room = SimulationRoom(RoomAddress("test", ["sensor"]), tolerance=0.2)
        room.predict("temperature", predicted_value=72.0, confidence=0.9, horizon_seconds=10)
        
        gap = room.observe("temperature", actual_value=73.0)
        assert gap is None  # Within tolerance

    def test_predict_and_observe_mismatch(self):
        room = SimulationRoom(RoomAddress("test", ["sensor"]), tolerance=0.1)
        room.predict("temperature", predicted_value=72.0, confidence=0.9, horizon_seconds=10)
        
        gap = room.observe("temperature", actual_value=100.0)
        assert gap is not None
        assert gap is not None
        assert gap.delta > 0  # mismatch detected

    def test_no_prediction_no_gap(self):
        room = SimulationRoom(RoomAddress("test", ["sensor"]))
        gap = room.observe("unknown_event", actual_value=42)
        assert gap is None

    def test_nested_rooms(self):
        parent = SimulationRoom(RoomAddress("test", ["drift-room"]))
        child = parent.add_child("predictor", RoomKind.PREDICTOR)
        grandchild = child.add_child("model", RoomKind.MODEL)
        
        assert grandchild.address.path == ["drift-room", "predictor", "model"]
        
        found = parent.get_child(["predictor", "model"])
        assert found is grandchild

    def test_focus_report(self):
        room = SimulationRoom(RoomAddress("test", ["sensor"]), tolerance=0.1)
        room.predict("value", 10.0, confidence=0.95, horizon_seconds=10)
        room.observe("value", 50.0)
        
        report = room.focus_report()
        assert "Focus Queue" in report

    def test_summary(self):
        room = SimulationRoom(RoomAddress("test", ["room"]))
        room.predict("x", 1.0, confidence=0.8, horizon_seconds=5)
        room.observe("x", 1.05)
        
        s = room.summary()
        assert s["predictions"] == 1
        assert s["observations"] == 1

    def test_lamport_increments(self):
        room = SimulationRoom(RoomAddress("test", ["room"]))
        room.predict("a", 1, 0.5, 1)
        room.predict("b", 2, 0.5, 1)
        room.predict("c", 3, 0.5, 1)
        assert room.lamport == 3

    def test_string_delta(self):
        room = SimulationRoom(RoomAddress("test", ["room"]), tolerance=0.0)
        room.predict("class", "stable", confidence=0.9, horizon_seconds=10)
        gap = room.observe("class", "drifting")
        assert gap is not None
        assert gap.delta == 1.0

    def test_boolean_delta(self):
        room = SimulationRoom(RoomAddress("test", ["room"]), tolerance=0.0)
        room.predict("ok", True, confidence=0.99, horizon_seconds=10)
        gap = room.observe("ok", False)
        assert gap is not None
        assert gap.severity == GapSeverity.CRITICAL
