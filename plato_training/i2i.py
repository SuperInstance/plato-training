"""
I2I Bridge — Instance-to-Instance protocol for PLATO rooms.

The monolith builds and discovers seams. I2I is how carved modules
talk across hardware boundaries without importing each other.

Protocol:
  Instance A (training server) ──tile──→ PLATO ──tile──→ Instance B (Jetson)
  
  No Python imports. No shared filesystem. Just tiles with schemas.

Tile schemas:
  - model-tile: trained weights + spec + benchmark results
  - data-tile: dataset spec + content hash + split info  
  - compression-tile: compression method + config + accuracy impact
  - benchmark-tile: latency + throughput + accuracy on specific hardware
  - deploy-tile: which model + which hardware + health status

Each tile is self-describing JSON. Any language, any hardware.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


# ─── Tile Schema Registry ──────────────────────────────────────────

TILE_SCHEMAS = {
    "model-tile": {
        "required": ["tile_type", "task", "variant", "weights_hash", "input_dim", "num_classes", "accuracy"],
        "optional": ["compression_ratio", "training_time_s", "loss_curve", "class_names"],
    },
    "data-tile": {
        "required": ["tile_type", "source", "n_samples", "input_dim", "num_classes"],
        "optional": ["class_distribution", "split_ratio", "content_hash", "columns"],
    },
    "compression-tile": {
        "required": ["tile_type", "method", "original_params", "compressed_params", "accuracy_original", "accuracy_compressed"],
        "optional": ["rank", "n_control_points", "basis", "recommendation"],
    },
    "benchmark-tile": {
        "required": ["tile_type", "task", "variant", "target", "latency_ms", "throughput", "accuracy"],
        "optional": ["model_size_bytes", "device_actual", "notes"],
    },
    "deploy-tile": {
        "required": ["tile_type", "room_id", "task", "target", "model_tile_ref", "status"],
        "optional": ["accuracy_floor", "meets_floor", "invocations", "last_health_check"],
    },
}


def validate_tile(tile_data: Dict) -> Tuple[bool, List[str]]:
    """Validate a tile against its schema. Returns (valid, errors)."""
    tile_type = tile_data.get("tile_type", "")
    if tile_type not in TILE_SCHEMAS:
        return False, [f"Unknown tile type: {tile_type}"]
    
    schema = TILE_SCHEMAS[tile_type]
    errors = []
    
    for req in schema["required"]:
        if req not in tile_data:
            errors.append(f"Missing required field: {req}")
    
    return len(errors) == 0, errors


def tile_id(tile_data: Dict) -> str:
    """Generate deterministic tile ID from content."""
    canonical = json.dumps(tile_data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── I2I Message ───────────────────────────────────────────────────

@dataclass
class I2IMessage:
    """
    An instance-to-instance message.
    
    Wire format: JSON with envelope + payload.
    Payload is a validated tile.
    """
    sender: str          # instance ID (e.g., "forgemaster@eileen")
    recipient: str       # instance ID or "broadcast"
    timestamp: float
    lamport: int
    payload_type: str    # "model-tile", "data-tile", etc.
    payload: Dict        # the actual tile data
    signature: str = ""  # content hash for verification
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "I2IMessage":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def validate(self) -> Tuple[bool, List[str]]:
        return validate_tile(self.payload)
    
    def compute_signature(self) -> str:
        canonical = json.dumps(self.payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── Instance Identity ─────────────────────────────────────────────

@dataclass
class InstanceID:
    """Who am I and what hardware am I on."""
    agent: str           # "forgemaster", "oracle1", "ensign-7"
    machine: str         # "eileen", "jetsonclaw1", "cloud-run-abc"
    hardware: str        # "cpu", "gpu", "npu", "tpu", "wasm"
    role: str            # "trainer", "inference", "coordinator"
    
    @property
    def instance_id(self) -> str:
        return f"{self.agent}@{self.machine}"
    
    def __str__(self) -> str:
        return self.instance_id


# ─── I2I Bridge ─────────────────────────────────────────────────────

class I2IBridge:
    """
    Bridge between instances. Send and receive tiles.
    
    In production: uses PLATO server + Matrix for transport.
    In development: uses local filesystem.
    """
    
    def __init__(
        self,
        identity: InstanceID,
        transport: str = "local",  # "local", "plato", "matrix"
        plato_url: str = "http://147.224.38.131:8847",
        local_dir: str = ".i2i-bridge",
    ):
        self.identity = identity
        self.transport = transport
        self.plato_url = plato_url
        self.local_dir = local_dir
        self.lamport = 0
        self.outbox: List[I2IMessage] = []
        self.inbox: List[I2IMessage] = []
    
    def send(self, payload_type: str, payload: Dict, recipient: str = "broadcast") -> I2IMessage:
        """Send a tile to another instance."""
        # Validate
        valid, errors = validate_tile(payload)
        if not valid:
            raise ValueError(f"Invalid tile: {errors}")
        
        self.lamport += 1
        msg = I2IMessage(
            sender=str(self.identity),
            recipient=recipient,
            timestamp=time.time(),
            lamport=self.lamport,
            payload_type=payload_type,
            payload=payload,
        )
        msg.signature = msg.compute_signature()
        
        if self.transport == "local":
            self._local_send(msg)
        
        self.outbox.append(msg)
        return msg
    
    def receive(self) -> List[I2IMessage]:
        """Check for incoming messages."""
        if self.transport == "local":
            self._local_receive()
        
        return self.inbox
    
    def _local_send(self, msg: I2IMessage):
        """Write message to local filesystem for another instance to pick up."""
        from pathlib import Path
        out_dir = Path(self.local_dir) / "outbox"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{msg.payload_type}-{msg.lamport:04d}-{msg.signature}.json"
        (out_dir / filename).write_text(msg.to_json())
    
    def _local_receive(self):
        """Read messages from local filesystem inbox."""
        from pathlib import Path
        in_dir = Path(self.local_dir) / "inbox"
        if not in_dir.exists():
            return
        
        for f in sorted(in_dir.glob("*.json")):
            try:
                msg = I2IMessage.from_json(f.read_text())
                self.inbox.append(msg)
                f.unlink()  # Consume
            except Exception:
                pass
    
    def create_model_tile(
        self,
        task: str,
        variant: str,
        weights_hash: str,
        input_dim: int,
        num_classes: int,
        accuracy: float,
        **kwargs,
    ) -> I2IMessage:
        """Create a model tile for sharing a trained model."""
        payload = {
            "tile_type": "model-tile",
            "task": task,
            "variant": variant,
            "weights_hash": weights_hash,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "accuracy": accuracy,
            "sender_hardware": self.identity.hardware,
            "sender_role": self.identity.role,
            **kwargs,
        }
        return self.send("model-tile", payload)
    
    def create_benchmark_tile(
        self,
        task: str,
        variant: str,
        target: str,
        latency_ms: float,
        throughput: float,
        accuracy: float,
        **kwargs,
    ) -> I2IMessage:
        """Share benchmark results across instances."""
        payload = {
            "tile_type": "benchmark-tile",
            "task": task,
            "variant": variant,
            "target": target,
            "latency_ms": latency_ms,
            "throughput": throughput,
            "accuracy": accuracy,
            "benchmarked_by": str(self.identity),
            **kwargs,
        }
        return self.send("benchmark-tile", payload)
    
    def create_deploy_tile(
        self,
        room_id: str,
        task: str,
        target: str,
        model_tile_ref: str,
        status: str = "deployed",
        **kwargs,
    ) -> I2IMessage:
        """Share deployment status."""
        payload = {
            "tile_type": "deploy-tile",
            "room_id": room_id,
            "task": task,
            "target": target,
            "model_tile_ref": model_tile_ref,
            "status": status,
            "deployed_by": str(self.identity),
            "deployed_on": self.identity.hardware,
            **kwargs,
        }
        return self.send("deploy-tile", payload)
    
    def summary(self) -> Dict:
        return {
            "identity": str(self.identity),
            "transport": self.transport,
            "lamport": self.lamport,
            "sent": len(self.outbox),
            "received": len(self.inbox),
        }
