"""Tests for DataRoom — real data loading pipeline."""

import pytest
import torch
import json
import csv
import os
import tempfile
from plato_training.data_rooms import DataRoom, DataSpec


class TestDataRoomFromTensors:
    def test_basic(self):
        X = torch.randn(100, 16)
        y = torch.randint(0, 3, (100,))
        room = DataRoom.from_tensors(X, y, name="test")
        
        assert room.spec.input_dim == 16
        assert room.spec.num_classes == 3
        assert len(room.X) == 100

    def test_split(self):
        X = torch.randn(100, 8)
        y = torch.randint(0, 2, (100,))
        room = DataRoom.from_tensors(X, y)
        
        X_tr, y_tr, X_val, y_val = room.split(val_ratio=0.2)
        assert len(X_tr) == 80
        assert len(X_val) == 20

    def test_dataset(self):
        X = torch.randn(50, 4)
        y = torch.randint(0, 2, (50,))
        room = DataRoom.from_tensors(X, y)
        train_ds, val_ds = room.dataset()
        assert len(train_ds) + len(val_ds) == 50

    def test_dataloader(self):
        X = torch.randn(50, 4)
        y = torch.randint(0, 2, (50,))
        room = DataRoom.from_tensors(X, y)
        train_dl, val_dl = room.dataloader(batch_size=16)
        batch = next(iter(train_dl))
        assert len(batch) == 2

    def test_summary(self):
        X = torch.randn(100, 8)
        y = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(40, dtype=torch.long)])
        room = DataRoom.from_tensors(X, y, class_names=["neg", "pos"])
        
        s = room.summary()
        assert s["samples"] == 100
        assert s["class_distribution"]["neg"] == 60
        assert s["class_distribution"]["pos"] == 40


class TestDataRoomFromCSV:
    def test_basic_csv(self, tmp_path):
        # Write test CSV
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["f1", "f2", "f3", "label"])
            for i in range(50):
                writer.writerow([0.1 * i, 0.2 * i, 0.3 * i, i % 2])
        
        room = DataRoom.from_csv(csv_path, label_col="label")
        assert room.spec.input_dim == 3
        assert room.spec.num_classes == 2
        assert len(room.X) == 50

    def test_csv_with_feature_range(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "f1", "f2", "label", "metadata"])
            for i in range(20):
                writer.writerow([i, float(i), float(i*2), i % 3, f"meta-{i}"])
        
        room = DataRoom.from_csv(csv_path, label_col="label", feature_range=(1, 3))
        assert room.spec.input_dim == 2  # f1, f2


class TestDataRoomFromJSONL:
    def test_basic_jsonl(self, tmp_path):
        jsonl_path = str(tmp_path / "test.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(30):
                record = {
                    "features": [float(i), float(i*2), float(i*3)],
                    "label": i % 2,
                }
                f.write(json.dumps(record) + "\n")
        
        room = DataRoom.from_jsonl(jsonl_path)
        assert room.spec.input_dim == 3
        assert len(room.X) == 30

    def test_jsonl_with_transform(self, tmp_path):
        jsonl_path = str(tmp_path / "test.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(20):
                f.write(json.dumps({"text": f"sample {i}", "label": i % 2}) + "\n")
        
        # Custom feature extractor
        def extract(record):
            text = record["text"]
            # Simple bag-of-words-ish: length, word count, first char code
            return [len(text), len(text.split()), ord(text[0]) % 100]
        
        room = DataRoom.from_jsonl(jsonl_path, text_to_features=extract)
        assert room.spec.input_dim == 3


class TestDataRoomIntegration:
    def test_csv_to_training(self, tmp_path):
        """End-to-end: CSV → DataRoom → train micro model → predict."""
        csv_path = str(tmp_path / "data.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "x2", "x3", "x4", "label"])
            for i in range(200):
                x = [float(j) * 0.1 + (0.5 if i % 2 else -0.5) for j in range(4)]
                writer.writerow(x + [i % 2])
        
        room = DataRoom.from_csv(csv_path, class_names=["class-a", "class-b"])
        
        # Should be able to split and create dataloaders
        train_dl, val_dl = room.dataloader(batch_size=32)
        assert len(train_dl) > 0
