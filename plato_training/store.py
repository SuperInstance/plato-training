"""
LocalTileStore — JSON-backed tile storage.

No PLATO server required. Tiles persist to disk as JSON.
PLATO client can swap in later without changing room code.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict
from .types import TrainingTile, TileType, TileLifecycle


class TileStore:
    """Interface for tile storage."""
    def save(self, tile: TrainingTile) -> None: raise NotImplementedError
    def load(self, tile_id: str) -> Optional[TrainingTile]: raise NotImplementedError
    def list_tiles(self, room=None, tile_type=None, state=None) -> List[TrainingTile]: raise NotImplementedError


class LocalTileStore(TileStore):
    """
    JSON-backed tile store. Each tile = JSON file. Weights = content-addressed files.
    
    store_dir/
        tiles/{tile_id}.json
        weights/{content_hash}.safetensors
    """
    
    def __init__(self, store_dir: str = ".plato-training"):
        self.store_dir = Path(store_dir)
        self.tiles_dir = self.store_dir / "tiles"
        self.weights_dir = self.store_dir / "weights"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, tile: TrainingTile) -> None:
        (self.tiles_dir / f"{tile.tile_id}.json").write_text(json.dumps(tile.to_dict(), indent=2))
    
    def load(self, tile_id: str) -> Optional[TrainingTile]:
        path = self.tiles_dir / f"{tile_id}.json"
        if not path.exists(): return None
        return TrainingTile.from_dict(json.loads(path.read_text()))
    
    def list_tiles(self, room=None, tile_type=None, state=None) -> List[TrainingTile]:
        tiles = []
        for path in self.tiles_dir.glob("*.json"):
            tile = TrainingTile.from_dict(json.loads(path.read_text()))
            tiles.append(tile)
        if room: tiles = [t for t in tiles if t.room == room]
        if tile_type: tiles = [t for t in tiles if t.tile_type == tile_type]
        if state: tiles = [t for t in tiles if t.state == state]
        return sorted(tiles, key=lambda t: t.lamport)
    
    def find_active(self, tile_type=None, room=None) -> Optional[TrainingTile]:
        active = self.list_tiles(tile_type=tile_type, room=room, state=TileLifecycle.ACTIVE)
        return active[-1] if active else None
    
    def save_weights(self, content_hash: str, data: bytes) -> Path:
        path = self.weights_dir / f"{content_hash}.safetensors"
        path.write_bytes(data)
        return path
    
    def load_weights(self, content_hash: str) -> Optional[bytes]:
        path = self.weights_dir / f"{content_hash}.safetensors"
        return path.read_bytes() if path.exists() else None
    
    def delete(self, tile_id: str) -> bool:
        path = self.tiles_dir / f"{tile_id}.json"
        if path.exists(): path.unlink(); return True
        return False
    
    def stats(self) -> Dict[str, int]:
        all_tiles = self.list_tiles()
        return {
            "total": len(all_tiles),
            "active": sum(1 for t in all_tiles if t.state == TileLifecycle.ACTIVE),
            "superseded": sum(1 for t in all_tiles if t.state == TileLifecycle.SUPERSEDED),
            "retracted": sum(1 for t in all_tiles if t.state == TileLifecycle.RETRACTED),
        }
