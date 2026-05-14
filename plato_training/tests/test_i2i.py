"""
Tests for I2I Bridge — instance-to-instance protocol.
"""

import pytest
import json
import os
from pathlib import Path
from plato_training.i2i import (
    I2IMessage, I2IBridge, InstanceID,
    validate_tile, tile_id, TILE_SCHEMAS,
)


class TestInstanceID:
    def test_string_format(self):
        inst = InstanceID(agent="forgemaster", machine="eileen", hardware="gpu", role="trainer")
        assert str(inst) == "forgemaster@eileen"

    def test_instance_id_property(self):
        inst = InstanceID(agent="oracle1", machine="cloud-1", hardware="cpu", role="coordinator")
        assert inst.instance_id == "oracle1@cloud-1"


class TestTileValidation:
    def test_valid_model_tile(self):
        tile = {
            "tile_type": "model-tile",
            "task": "drift-detect",
            "variant": "spline",
            "weights_hash": "abc123",
            "input_dim": 64,
            "num_classes": 2,
            "accuracy": 1.0,
        }
        valid, errors = validate_tile(tile)
        assert valid
        assert len(errors) == 0

    def test_invalid_missing_field(self):
        tile = {"tile_type": "model-tile", "task": "drift-detect"}
        valid, errors = validate_tile(tile)
        assert not valid
        assert any("weights_hash" in e for e in errors)

    def test_unknown_tile_type(self):
        valid, errors = validate_tile({"tile_type": "quantum-tile"})
        assert not valid

    def test_benchmark_tile(self):
        tile = {
            "tile_type": "benchmark-tile",
            "task": "drift-detect",
            "variant": "spline",
            "target": "npu",
            "latency_ms": 0.11,
            "throughput": 9423,
            "accuracy": 1.0,
        }
        valid, _ = validate_tile(tile)
        assert valid


class TestI2IMessage:
    def test_round_trip_json(self):
        msg = I2IMessage(
            sender="forgemaster@eileen",
            recipient="ensign-7@jetson",
            timestamp=1234.5,
            lamport=1,
            payload_type="model-tile",
            payload={"tile_type": "model-tile", "task": "drift-detect",
                     "variant": "spline", "weights_hash": "abc",
                     "input_dim": 64, "num_classes": 2, "accuracy": 1.0},
        )
        json_str = msg.to_json()
        restored = I2IMessage.from_json(json_str)
        assert restored.sender == msg.sender
        assert restored.payload["task"] == "drift-detect"


class TestI2IBridge:
    def test_send_model_tile(self, tmp_path):
        identity = InstanceID("forgemaster", "eileen", "gpu", "trainer")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        msg = bridge.create_model_tile(
            task="drift-detect", variant="spline", weights_hash="abc123",
            input_dim=64, num_classes=2, accuracy=1.0,
        )
        
        assert msg.payload_type == "model-tile"
        assert msg.sender == "forgemaster@eileen"
        assert len(bridge.outbox) == 1
    
    def test_send_creates_file(self, tmp_path):
        identity = InstanceID("forgemaster", "eileen", "cpu", "trainer")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        bridge.create_model_tile(
            task="drift-detect", variant="dense", weights_hash="def456",
            input_dim=64, num_classes=2, accuracy=0.95,
        )
        
        outbox = Path(tmp_path / "i2i" / "outbox")
        assert outbox.exists()
        files = list(outbox.glob("*.json"))
        assert len(files) == 1
    
    def test_benchmark_tile(self, tmp_path):
        identity = InstanceID("ensign-7", "jetson", "gpu-small", "inference")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        msg = bridge.create_benchmark_tile(
            task="drift-detect", variant="spline", target="gpu-small",
            latency_ms=0.5, throughput=2000, accuracy=0.99,
        )
        
        assert msg.payload["target"] == "gpu-small"
        assert msg.payload["benchmarked_by"] == "ensign-7@jetson"
    
    def test_deploy_tile(self, tmp_path):
        identity = InstanceID("forgemaster", "eileen", "cpu", "trainer")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        msg = bridge.create_deploy_tile(
            room_id="drift-7", task="drift-detect", target="npu",
            model_tile_ref="abc123", meets_floor=True,
        )
        
        assert msg.payload["room_id"] == "drift-7"
        assert msg.payload["deployed_on"] == "cpu"
    
    def test_receive_from_inbox(self, tmp_path):
        identity = InstanceID("ensign-7", "jetson", "gpu-small", "inference")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        # Simulate incoming message
        in_dir = Path(tmp_path / "i2i" / "inbox")
        in_dir.mkdir(parents=True, exist_ok=True)
        
        msg = I2IMessage(
            sender="forgemaster@eileen",
            recipient="ensign-7@jetson",
            timestamp=1234.5,
            lamport=5,
            payload_type="model-tile",
            payload={
                "tile_type": "model-tile", "task": "drift-detect",
                "variant": "spline", "weights_hash": "abc",
                "input_dim": 64, "num_classes": 2, "accuracy": 1.0,
            },
        )
        (in_dir / "model-tile-0005-abc.json").write_text(msg.to_json())
        
        received = bridge.receive()
        assert len(received) == 1
        assert received[0].payload["task"] == "drift-detect"
    
    def test_lamport_increments(self, tmp_path):
        identity = InstanceID("test", "test", "cpu", "test")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        bridge.create_model_tile("t1", "dense", "a", 32, 2, 0.9)
        bridge.create_model_tile("t2", "dense", "b", 32, 2, 0.9)
        bridge.create_model_tile("t3", "dense", "c", 32, 2, 0.9)
        
        assert bridge.lamport == 3
    
    def test_summary(self, tmp_path):
        identity = InstanceID("test", "test", "cpu", "test")
        bridge = I2IBridge(identity, transport="local", local_dir=str(tmp_path / "i2i"))
        
        s = bridge.summary()
        assert s["identity"] == "test@test"
        assert s["sent"] == 0
