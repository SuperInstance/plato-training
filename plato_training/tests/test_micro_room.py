"""Tests for MicroRoom — ensign-facing room interface."""

import pytest
import torch
from plato_training.micro_room import MicroRoom, RoomFactory


class TestMicroRoom:
    def test_create_and_predict(self, tmp_path):
        room = MicroRoom("drift-detect", target="cpu", store_dir=str(tmp_path / "s"))
        x = torch.randn(64)
        result = room.predict(x)
        
        assert "prediction" in result
        assert "confidence" in result
        assert "class_name" in result
        assert result["class_name"] in ("stable", "drifting")
        assert 0 <= result["confidence"] <= 1

    def test_predict_with_ground_truth(self, tmp_path):
        room = MicroRoom("anomaly-flag", target="cpu", store_dir=str(tmp_path / "s"))
        x = torch.randn(16)
        result = room.predict(x, ground_truth=0)
        
        assert result["prediction"] in (0, 1)
        assert len(room.invocations) == 1
        assert room.invocations[0].correct is not None

    def test_batch_predict(self, tmp_path):
        room = MicroRoom("sentiment", target="cpu", store_dir=str(tmp_path / "s"))
        x = torch.randn(8, 128)
        results = room.predict_batch(x)
        
        assert len(results) == 8
        assert all("class_name" in r for r in results)

    def test_summary(self, tmp_path):
        room = MicroRoom("intent-detect", target="cpu", store_dir=str(tmp_path / "s"))
        for _ in range(5):
            room.predict(torch.randn(64), ground_truth=0)
        
        s = room.summary()
        assert s["invocations"] == 5
        assert s["task"] == "intent-detect"
        assert "classes" in s

    def test_save_and_load(self, tmp_path):
        store = str(tmp_path / "s")
        room = MicroRoom("spam-classify", target="cpu", store_dir=store, room_id="test-room")
        room.predict(torch.randn(128))
        path = room.save()
        
        assert "room-test-room.json" in path
        
        loaded = MicroRoom.load("test-room", store_dir=store)
        assert loaded.task == "spam-classify"
        assert loaded.target == "cpu"

    def test_spec(self, tmp_path):
        room = MicroRoom("priority-rank", target="cpu", store_dir=str(tmp_path / "s"))
        spec = room.spec()
        assert "priority" in spec.lower()
        assert "low" in spec
        assert "critical" in spec

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            MicroRoom("nonexistent-task")


class TestRoomFactory:
    def test_create_room(self, tmp_path):
        factory = RoomFactory(store_dir=str(tmp_path / "s"))
        room = factory.create("drift-detect", target="cpu")
        
        assert room.room_id in factory.rooms
        assert room.task == "drift-detect"

    def test_list_rooms(self, tmp_path):
        factory = RoomFactory(store_dir=str(tmp_path / "s"))
        factory.create("drift-detect", target="cpu")
        factory.create("anomaly-flag", target="cpu")
        
        rooms = factory.list_rooms()
        assert len(rooms) == 2
        tasks = {r["task"] for r in rooms}
        assert "drift-detect" in tasks
        assert "anomaly-flag" in tasks

    def test_save_all(self, tmp_path):
        store = str(tmp_path / "s")
        factory = RoomFactory(store_dir=store)
        factory.create("spam-classify", room_id="room-1")
        factory.create("sentiment", room_id="room-2")
        
        factory.save_all()
        
        import os
        files = os.listdir(store)
        assert "room-room-1.json" in files
        assert "room-room-2.json" in files

    def test_get_room(self, tmp_path):
        factory = RoomFactory(store_dir=str(tmp_path / "s"))
        room = factory.create("drift-detect", room_id="my-room")
        
        assert factory.get("my-room") is room
        assert factory.get("nonexistent") is None
