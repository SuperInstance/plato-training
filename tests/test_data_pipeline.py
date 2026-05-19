"""Tests for data_pipeline module."""

import json
import csv
import os
import tempfile
import pytest

import numpy as np

from plato_training.data_pipeline import (
    ColumnSchema,
    DataSchema,
    DataVersion,
    DataContainer,
    DataLoader,
    CSVLoader,
    JSONLLoader,
    GitCommitLoader,
    Transform,
    CommitFeatureExtractor,
    IdentityTransform,
    FunctionTransform,
    FeatureStore,
    FeatureLineage,
    DataSplitter,
    RandomSplitter,
    TemporalSplitter,
    StratifiedSplitter,
    Pipeline,
)


# ─── Helpers ───────────────────────────────────────────────────────


def make_csv_file(rows, headers, tmpdir, name="test.csv"):
    path = os.path.join(tmpdir, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def make_jsonl_file(records, tmpdir, name="test.jsonl"):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# ─── DataSchema Tests ──────────────────────────────────────────────


class TestDataSchema:
    def test_column_names(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float),
            ColumnSchema("y", float),
            ColumnSchema("label", int),
        ])
        assert schema.column_names == ["x", "y", "label"]

    def test_validate_valid_row(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float),
            ColumnSchema("label", int),
        ])
        errors = schema.validate_row({"x": "1.5", "label": "0"})
        assert errors == []

    def test_validate_missing_required(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float, required=True),
        ])
        errors = schema.validate_row({})
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_validate_wrong_type(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float, required=True),
        ])
        errors = schema.validate_row({"x": "not_a_number"})
        assert len(errors) == 1

    def test_validate_optional_missing(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float, required=True),
            ColumnSchema("extra", str, required=False),
        ])
        errors = schema.validate_row({"x": "1.0"})
        assert errors == []


# ─── DataContainer Tests ──────────────────────────────────────────


class TestDataContainer:
    def _make_container(self):
        schema = DataSchema(columns=[
            ColumnSchema("x", float),
            ColumnSchema("y", float),
            ColumnSchema("label", int),
        ], name="test")
        return DataContainer("test", schema)

    def test_add_row(self):
        c = self._make_container()
        errors = c.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        assert errors == []
        assert c.row_count == 1

    def test_add_row_invalid(self):
        c = self._make_container()
        errors = c.add_row({"x": "abc", "y": "2.0", "label": "0"})
        assert len(errors) > 0

    def test_add_rows(self):
        c = self._make_container()
        rows = [
            {"x": "1.0", "y": "2.0", "label": "0"},
            {"x": "3.0", "y": "4.0", "label": "1"},
        ]
        errors = c.add_rows(rows)
        assert errors == []
        assert c.row_count == 2

    def test_seal_prevents_modification(self):
        c = self._make_container()
        c.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        version = c.seal()
        assert version.sealed is True
        assert c.sealed is True
        with pytest.raises(RuntimeError, match="sealed"):
            c.add_row({"x": "5.0", "y": "6.0", "label": "1"})

    def test_version_has_content_hash(self):
        c = self._make_container()
        c.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        assert c.version is not None
        assert len(c.version.content_hash) == 16
        assert c.version.row_count == 1

    def test_version_changes_on_add(self):
        c = self._make_container()
        c.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        hash1 = c.version.content_hash
        c.add_row({"x": "3.0", "y": "4.0", "label": "1"})
        hash2 = c.version.content_hash
        assert hash1 != hash2

    def test_seal_version_is_immutable(self):
        c = self._make_container()
        c.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        version = c.seal()
        # Content hash is deterministic
        c2 = self._make_container()
        c2.add_row({"x": "1.0", "y": "2.0", "label": "0"})
        version2 = c2.seal()
        assert version.content_hash == version2.content_hash

    def test_to_numpy(self):
        c = self._make_container()
        c.add_rows([
            {"x": "1.0", "y": "2.0", "label": "0"},
            {"x": "3.0", "y": "4.0", "label": "1"},
        ])
        arr = c.to_numpy()
        assert arr.shape == (2, 3)
        assert arr[0, 0] == 1.0

    def test_get_column(self):
        c = self._make_container()
        c.add_rows([
            {"x": "1.0", "y": "2.0", "label": "0"},
            {"x": "3.0", "y": "4.0", "label": "1"},
        ])
        assert c.get_column("x") == [1.0, 3.0]
        assert c.get_column("label") == [0, 1]


# ─── CSVLoader Tests ──────────────────────────────────────────────


class TestCSVLoader:
    def test_load_basic(self, tmp_path):
        path = make_csv_file(
            [{"x": "1", "y": "2", "label": "0"}, {"x": "3", "y": "4", "label": "1"}],
            ["x", "y", "label"],
            str(tmp_path),
        )
        loader = CSVLoader(path)
        rows = loader.load()
        assert len(rows) == 2
        assert rows[0]["x"] == 1
        assert rows[0]["label"] == 0

    def test_load_float_values(self, tmp_path):
        path = make_csv_file(
            [{"val": "1.5", "label": "0"}],
            ["val", "label"],
            str(tmp_path),
        )
        loader = CSVLoader(path)
        rows = loader.load()
        assert rows[0]["val"] == 1.5

    def test_load_max_rows(self, tmp_path):
        path = make_csv_file(
            [{"x": str(i), "label": "0"} for i in range(10)],
            ["x", "label"],
            str(tmp_path),
        )
        loader = CSVLoader(path, max_rows=3)
        rows = loader.load()
        assert len(rows) == 3

    def test_load_string_values(self, tmp_path):
        path = make_csv_file(
            [{"name": "alice", "score": "90", "label": "1"}],
            ["name", "score", "label"],
            str(tmp_path),
        )
        loader = CSVLoader(path)
        rows = loader.load()
        assert rows[0]["name"] == "alice"
        assert rows[0]["score"] == 90


# ─── JSONLLoader Tests ────────────────────────────────────────────


class TestJSONLLoader:
    def test_load_basic(self, tmp_path):
        path = make_jsonl_file(
            [{"x": 1.0, "y": 2.0, "label": 0}, {"x": 3.0, "y": 4.0, "label": 1}],
            str(tmp_path),
        )
        loader = JSONLLoader(path)
        rows = loader.load()
        assert len(rows) == 2
        assert rows[0]["x"] == 1.0

    def test_load_max_rows(self, tmp_path):
        path = make_jsonl_file(
            [{"x": float(i)} for i in range(10)],
            str(tmp_path),
        )
        loader = JSONLLoader(path, max_rows=5)
        rows = loader.load()
        assert len(rows) == 5

    def test_load_skips_blank_lines(self, tmp_path):
        path = make_jsonl_file(
            [{"x": 1.0}, {"x": 2.0}],
            str(tmp_path),
        )
        # Append a blank line
        with open(path, "a") as f:
            f.write("\n\n")
        loader = JSONLLoader(path)
        rows = loader.load()
        assert len(rows) == 2


# ─── Transform Tests ──────────────────────────────────────────────


class TestCommitFeatureExtractor:
    def test_extracts_features(self):
        rows = [{
            "sha": "abc123",
            "repo": "test",
            "files_changed": 3,
            "insertions": 50,
            "deletions": 10,
            "is_merge": False,
            "timestamp": 1000.0,
        }]
        t = CommitFeatureExtractor()
        result = t.apply(rows)
        assert len(result) == 1
        assert result[0]["files_changed"] == 3
        assert result[0]["size"] == 60  # insertions + deletions
        assert result[0]["net_lines"] == 40  # insertions - deletions

    def test_labels_large_commits(self):
        rows = [
            {"sha": "a", "files_changed": 3, "insertions": 10, "deletions": 5, "is_merge": False, "timestamp": 1.0},
            {"sha": "b", "files_changed": 10, "insertions": 100, "deletions": 50, "is_merge": False, "timestamp": 2.0},
        ]
        t = CommitFeatureExtractor()
        result = t.apply(rows)
        assert result[0]["label"] == 0  # files_changed <= 5
        assert result[1]["label"] == 1  # files_changed > 5

    def test_skips_non_commit_rows(self):
        rows = [{"x": 1.0, "y": 2.0}]  # no 'sha'
        t = CommitFeatureExtractor()
        result = t.apply(rows)
        assert len(result) == 0


class TestIdentityTransform:
    def test_passthrough(self):
        rows = [{"x": 1}, {"x": 2}]
        t = IdentityTransform()
        assert t.apply(rows) == rows


class TestFunctionTransform:
    def test_custom_transform(self):
        rows = [{"x": 1}, {"x": 2}]
        t = FunctionTransform(lambda rs: [{"x": r["x"] * 2} for r in rs])
        result = t.apply(rows)
        assert result[0]["x"] == 2
        assert result[1]["x"] == 4


# ─── FeatureStore Tests ──────────────────────────────────────────


class TestFeatureStore:
    def test_compute_auto(self):
        rows = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        store = FeatureStore()
        store.compute(rows)
        assert store.n_rows == 2
        assert "x" in store.feature_names
        assert "y" in store.feature_names

    def test_compute_with_spec(self):
        rows = [{"a": 10, "b": 20, "c": 30}, {"a": 1, "b": 2, "c": 3}]
        store = FeatureStore()
        store.compute(rows, feature_spec={
            "total": {"columns": ["a", "b", "c"], "transform": "sum"},
            "avg": {"columns": ["a", "b", "c"], "transform": "mean"},
        })
        assert store.get("total") == [60.0, 6.0]
        assert store.get("avg") == [20.0, 2.0]

    def test_compute_ratio(self):
        rows = [{"a": 10, "b": 5}, {"a": 20, "b": 4}]
        store = FeatureStore()
        store.compute(rows, feature_spec={
            "ratio": {"columns": ["a", "b"], "transform": "ratio"},
        })
        assert abs(store.get("ratio")[0] - 2.0) < 1e-6
        assert abs(store.get("ratio")[1] - 5.0) < 1e-6

    def test_lineage(self):
        rows = [{"a": 1, "b": 2}]
        store = FeatureStore()
        store.compute(rows, feature_spec={
            "total": {"columns": ["a", "b"], "transform": "sum"},
        })
        lin = store.lineage["total"]
        assert lin.source_columns == ["a", "b"]
        assert lin.transform == "sum"

    def test_to_numpy(self):
        rows = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        store = FeatureStore()
        store.compute(rows)
        arr = store.to_numpy()
        assert arr.shape == (2, 2)

    def test_invalidate(self):
        rows = [{"x": 1.0}]
        store = FeatureStore()
        store.compute(rows)
        assert store.is_valid()
        invalidated = store.invalidate(source_hash="different_hash")
        assert "x" in invalidated
        assert not store.is_valid()

    def test_invalidate_same_hash_noop(self):
        rows = [{"x": 1.0}]
        store = FeatureStore()
        store.compute(rows)
        invalidated = store.invalidate(source_hash=store._source_hash)
        assert invalidated == []
        assert store.is_valid()

    def test_empty_rows(self):
        store = FeatureStore()
        result = store.compute([])
        assert result.n_rows == 0


# ─── DataSplitter Tests ──────────────────────────────────────────


class TestRandomSplitter:
    def test_basic_split(self):
        rows = [{"i": i} for i in range(100)]
        splitter = RandomSplitter(train=0.7, val=0.15, test=0.15, seed=42)
        train, val, test = splitter.split(rows)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100

    def test_deterministic(self):
        rows = [{"i": i} for i in range(50)]
        s1 = RandomSplitter(seed=42)
        s2 = RandomSplitter(seed=42)
        t1, v1, te1 = s1.split(rows)
        t2, v2, te2 = s2.split(rows)
        assert t1 == t2
        assert v1 == v2

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            RandomSplitter(train=0.5, val=0.3, test=0.3)


class TestTemporalSplitter:
    def test_temporal_ordering(self):
        rows = [
            {"timestamp": 100, "val": "a"},
            {"timestamp": 300, "val": "c"},
            {"timestamp": 200, "val": "b"},
        ]
        splitter = TemporalSplitter(train=0.34, val=0.33, test=0.33)
        train, val, test = splitter.split(rows)
        # Train should have earliest
        for r in train:
            for r2 in test:
                assert r["timestamp"] <= r2["timestamp"]

    def test_no_future_leakage(self):
        rows = [{"timestamp": float(i)} for i in range(100)]
        splitter = TemporalSplitter(train=0.7, val=0.15, test=0.15)
        train, val, test = splitter.split(rows)
        max_train_time = max(r["timestamp"] for r in train)
        min_test_time = min(r["timestamp"] for r in test)
        assert max_train_time <= min_test_time

    def test_custom_time_key(self):
        rows = [{"ts": 1.0, "x": "a"}, {"ts": 2.0, "x": "b"}]
        splitter = TemporalSplitter(train=0.5, val=0.25, test=0.25, time_key="ts")
        train, val, test = splitter.split(rows)
        assert train[0]["x"] == "a"


class TestStratifiedSplitter:
    def test_stratified_distribution(self):
        rows = [{"label": i % 3, "x": i} for i in range(90)]
        splitter = StratifiedSplitter(label_key="label", seed=42)
        train, val, test = splitter.split(rows)
        # Each split should have all 3 labels
        for split_data in [train, val, test]:
            labels = set(r["label"] for r in split_data)
            assert labels == {0, 1, 2}

    def test_label_proportions(self):
        rows = [{"label": 0, "x": i} for i in range(50)]
        rows += [{"label": 1, "x": i + 50} for i in range(50)]
        splitter = StratifiedSplitter(label_key="label", train=0.6, val=0.2, test=0.2, seed=42)
        train, val, test = splitter.split(rows)
        # Both labels should appear in each split
        for split_data in [train, val, test]:
            label0 = sum(1 for r in split_data if r["label"] == 0)
            label1 = sum(1 for r in split_data if r["label"] == 1)
            assert label0 > 0
            assert label1 > 0


# ─── Pipeline Tests ───────────────────────────────────────────────


class TestPipeline:
    def test_csv_pipeline(self, tmp_path):
        path = make_csv_file(
            [{"x": str(i), "y": str(i * 2), "label": str(i % 3)} for i in range(30)],
            ["x", "y", "label"],
            str(tmp_path),
        )
        pipeline = Pipeline(
            source=CSVLoader(path),
            split=RandomSplitter(train=0.6, val=0.2, test=0.2),
            name="test-csv",
        )
        train, val, test = pipeline.run()
        total = len(train) + len(val) + len(test)
        assert total == 30
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_jsonl_pipeline(self, tmp_path):
        path = make_jsonl_file(
            [{"x": float(i), "y": float(i * 2), "timestamp": float(i * 100)} for i in range(20)],
            str(tmp_path),
        )
        pipeline = Pipeline(
            source=JSONLLoader(path),
            split=TemporalSplitter(train=0.5, val=0.25, test=0.25),
        )
        train, val, test = pipeline.run()
        assert len(train) + len(val) + len(test) == 20

    def test_pipeline_with_transform(self, tmp_path):
        path = make_jsonl_file(
            [{"sha": f"sha{i}", "files_changed": i * 2, "insertions": i * 10, "deletions": i, "is_merge": False, "timestamp": float(i)} for i in range(10)],
            str(tmp_path),
        )
        pipeline = Pipeline(
            source=JSONLLoader(path),
            transform=CommitFeatureExtractor(),
            split=RandomSplitter(train=0.6, val=0.2, test=0.2),
        )
        train, val, test = pipeline.run()
        # CommitFeatureExtractor should produce features
        assert len(train) > 0
        assert "files_changed" in train[0]

    def test_pipeline_with_schema_validation(self, tmp_path):
        path = make_csv_file(
            [
                {"x": "1.0", "y": "2.0", "label": "0"},
                {"x": "bad", "y": "4.0", "label": "1"},  # invalid x
                {"x": "5.0", "y": "6.0", "label": "2"},
            ],
            ["x", "y", "label"],
            str(tmp_path),
        )
        schema = DataSchema(columns=[
            ColumnSchema("x", float),
            ColumnSchema("y", float),
            ColumnSchema("label", int),
        ])
        pipeline = Pipeline(
            source=CSVLoader(path),
            schema=schema,
            split=RandomSplitter(train=0.5, val=0.25, test=0.25),
        )
        train, val, test = pipeline.run()
        total = len(train) + len(val) + len(test)
        # One row should be filtered out (x="bad")
        assert total == 2

    def test_pipeline_summary(self, tmp_path):
        path = make_csv_file(
            [{"x": str(i), "y": str(i), "label": "0"} for i in range(10)],
            ["x", "y", "label"],
            str(tmp_path),
        )
        pipeline = Pipeline(
            source=CSVLoader(path),
            split=RandomSplitter(train=0.7, val=0.15, test=0.15),
            name="summary-test",
        )
        pipeline.run()
        s = pipeline.summary()
        assert s["name"] == "summary-test"
        assert s["raw_rows"] == 10
        assert s["source_type"] == "CSVLoader"
        assert len(s["features"]) > 0

    def test_pipeline_features_computed(self, tmp_path):
        path = make_csv_file(
            [{"x": str(i), "y": str(i * 2), "label": str(i % 2)} for i in range(10)],
            ["x", "y", "label"],
            str(tmp_path),
        )
        pipeline = Pipeline(
            source=CSVLoader(path),
            split=RandomSplitter(),
        )
        pipeline.run()
        store = pipeline.features
        assert store.is_valid()
        arr = store.to_numpy()
        assert arr.shape[0] == 10

    def test_pipeline_default_splitter(self, tmp_path):
        path = make_csv_file(
            [{"x": str(i), "label": "0"} for i in range(10)],
            ["x", "label"],
            str(tmp_path),
        )
        pipeline = Pipeline(source=CSVLoader(path))
        train, val, test = pipeline.run()
        assert len(train) + len(val) + len(test) == 10

    def test_full_pipeline_with_feature_store(self, tmp_path):
        path = make_jsonl_file(
            [
                {"a": 10, "b": 20, "c": 30, "label": 0, "timestamp": 1.0},
                {"a": 5, "b": 10, "c": 15, "label": 1, "timestamp": 2.0},
                {"a": 8, "b": 16, "c": 24, "label": 0, "timestamp": 3.0},
                {"a": 3, "b": 6, "c": 9, "label": 1, "timestamp": 4.0},
            ],
            str(tmp_path),
        )
        store = FeatureStore(name="custom")
        pipeline = Pipeline(
            source=JSONLLoader(path),
            split=TemporalSplitter(train=0.5, val=0.25, test=0.25),
            feature_store=store,
        )
        train, val, test = pipeline.run()
        assert store.n_rows == 4
        assert store.is_valid()
        # Check feature store has lineage
        assert len(store.lineage) > 0
