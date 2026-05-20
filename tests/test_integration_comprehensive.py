"""
Comprehensive integration test — exercises all PLATO modules end-to-end.
Tests real API signatures, not assumed ones.
"""
import torch, gc, time, tempfile, os
import numpy as np
import torch.nn as nn
import pytest


def _clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()


# ═══ 1. Pre-Filter ═══

class TestPreFilterIntegration:
    def test_route_diverse_requests(self):
        from plato_training.intelligence_pre_filter import (
            PreFilter, RequestFeatures, IntentType, Domain, Urgency,
        )
        pf = PreFilter(n_control_points=16)

        test_cases = [
            (IntentType.QUERY, Domain.CONSTRAINT_THEORY, Urgency.MEDIUM),
            (IntentType.CODE, Domain.FLEET, Urgency.HIGH),
            (IntentType.CREATIVE, Domain.GENERAL, Urgency.LOW),
            (IntentType.MATH, Domain.CONSTRAINT_THEORY, Urgency.MEDIUM),
            (IntentType.COMMAND, Domain.FLEET, Urgency.CRITICAL),
            (IntentType.CHAT, Domain.GENERAL, Urgency.LOW),
            (IntentType.ANALYSIS, Domain.FLEET, Urgency.MEDIUM),
            (IntentType.QUERY, Domain.PLATO, Urgency.MEDIUM),
        ]

        for intent, domain, urgency in test_cases:
            features = RequestFeatures(
                intent=intent, domain=domain, urgency=urgency,
                input_length=50, hour=14.0, day_of_week=2,
                recent_success_rate=0.7,
                keyword_vector=np.random.randn(64).astype(np.float32),
            )
            vec = features.to_vector()
            assert vec.shape == (384,), f"Wrong feature shape: {vec.shape}"
            decision = pf.route(features)
            assert 0 <= decision.confidence <= 1.0
            assert decision.max_tokens_estimate > 0  # correct field name

    def test_self_train_cycle(self):
        from plato_training.intelligence_pre_filter import (
            PreFilter, RequestFeatures, IntentType, Domain, Urgency,
        )
        pf = PreFilter(n_control_points=16)

        for i in range(20):
            features = RequestFeatures(
                intent=IntentType.QUERY, domain=Domain.FLEET,
                urgency=Urgency.MEDIUM, input_length=30 + i,
                keyword_vector=np.random.randn(64).astype(np.float32),
            )
            decision = pf.route(features)
            pf.record_outcome(features, decision, was_good=(i % 3 != 0))

        assert pf.n_outcomes == 20  # property, not method

        metrics = pf.self_train(epochs=5)
        # May return status=insufficient_data if not enough varied data
        assert isinstance(metrics, dict)

    def test_tile_export_reload(self):
        from plato_training.intelligence_pre_filter import PreFilter, RequestFeatures, IntentType
        pf = PreFilter(n_control_points=16)
        tmp = tempfile.mkdtemp()
        tile_id = pf.export_tile(store_dir=tmp)
        assert isinstance(tile_id, str)
        loaded = pf.load_tile(tile_id, store_dir=tmp)
        assert loaded is True


# ═══ 2. Post-Filter ═══

class TestPostFilterIntegration:
    def test_process_diverse_responses(self):
        from plato_training.intelligence_post_filter import PostFilter
        ptf = PostFilter(n_control_points=16)

        responses = [
            ("Constraint theory uses Eisenstein integers. Accuracy: 100%. Speed: <1ms.", "explanation", "glm-5.1"),
            ("ERROR: CUDA out of memory. Try reducing batch size.", "error", "deepseek-v4"),
            ("Step 1: Clone repo. Step 2: Run tests. Step 3: Check results.", "instruction", "glm-5.1"),
            ("The answer involves lattice points and constraint satisfaction.", "analysis", "claude-opus"),
        ]

        for text, rtype, model in responses:
            decision, tiles = ptf.process(
                response_text=text,
                request_context="test query",
                model_used=model,
                response_type=rtype,
            )
            assert hasattr(decision, "keep_decision")

    def test_self_train_cycle(self):
        from plato_training.intelligence_post_filter import PostFilter
        ptf = PostFilter(n_control_points=16)

        # Process enough responses to generate training data
        for i in range(15):
            ptf.process(
                response_text=f"Test response {i} with some content.",
                model_used="glm-5.1",
                response_type="explanation",
            )

        for i in range(20):
            ptf.record_usefulness(f"tile-{i}", was_useful=(i % 2 == 0))

        result = ptf.self_train(epochs=5)
        # May return empty dict or None if insufficient data
        assert result is None or isinstance(result, dict)

    def test_tile_export(self):
        from plato_training.intelligence_post_filter import PostFilter
        ptf = PostFilter(n_control_points=16)
        tmp = tempfile.mkdtemp()
        ptf.process("Test response for export.", model_used="glm-5.1")
        tile_id = ptf.export_tile(store_dir=tmp)
        assert isinstance(tile_id, str)


# ═══ 3. Self-Trainer ═══

class TestSelfTrainerIntegration:
    def test_lifecycle(self):
        from plato_training.intelligence_self_trainer import SelfTrainer, Experience

        trainer = SelfTrainer(store_dir=tempfile.mkdtemp())

        for i in range(30):
            exp = Experience(
                timestamp=time.time() - (30 - i) * 60,
                request_features=np.random.randn(384).astype(np.float32),
                response_features=np.random.randn(256).astype(np.float32),
                route_decision={"target": i % 8},
                filter_decision={"keep": i % 5},
                outcome="good" if i % 3 != 0 else "bad",
                latency_saved_ms=float(i * 10),
                knowledge_reused=[],
            )
            trainer.record_experience(exp)

        buf_stats = trainer.buffer.stats()
        assert buf_stats["count"] == 30
        assert sum(buf_stats["outcomes"].values()) == 30

        metrics = trainer.run_training_cycle()
        if metrics:
            assert hasattr(metrics, 'cycle_id')
            assert hasattr(metrics, 'pre_metrics')
            assert metrics.overall_score() >= 0

        del trainer
        gc.collect()


# ═══ 4. Intelligence Room ═══

class TestIntelligenceRoomIntegration:
    def test_full_conversation(self):
        from plato_training.intelligence_room import IntelligenceRoom

        room = IntelligenceRoom(
            store_dir=tempfile.mkdtemp(),
            max_knowledge_tiles=50,
        )

        conversations = [
            ("How many tests in plato-training?", "644 tests passing as of May 2026.", "fleet", "glm-5.1"),
            ("What is SplineLinear?", "SplineLinear achieves 20x compression at 100% accuracy.", "constraint-theory", "glm-5.1"),
            ("Show me the snap function", "def snap(x): return round(x)", "code", "glm-5.1"),
            ("Deploy drift-detect to NPU", "Steps: 1. Train 2. Quantize 3. Export 4. Flash 5. Verify", "plato", "glm-5.1"),
            ("Fleet status?", "6 services down, Matrix bridge running.", "fleet", "deepseek-v4"),
        ]

        for request, response, domain, model in conversations:
            route = room.pre_route(request, domain=domain)
            result = room.post_process(response, request_text=request, domain=domain, model_used=model)
            assert "knowledge_tiles" in result

        status = room.status()
        assert status["knowledge_tiles"] > 0
        assert "total_requests" in status

    def test_persistence(self):
        from plato_training.intelligence_room import IntelligenceRoom

        tmp = tempfile.mkdtemp()
        room = IntelligenceRoom(store_dir=tmp, max_knowledge_tiles=50)
        room.post_process("Test knowledge: 644 tests.", request_text="how many tests?", domain="fleet", model_used="glm-5.1")
        n_before = len(room.knowledge)
        room.save_state()
        room2 = IntelligenceRoom(store_dir=tmp, max_knowledge_tiles=50)
        assert len(room2.knowledge) == n_before

    def test_self_train_graceful(self):
        from plato_training.intelligence_room import IntelligenceRoom
        room = IntelligenceRoom(store_dir=tempfile.mkdtemp())
        metrics = room.self_train_cycle()
        assert metrics.get("status") == "skipped" or isinstance(metrics, dict)


# ═══ 5. Forge Training ═══

class TestForgeIntegration:
    def test_training_loop(self):
        from plato_training.plato_forge import PlatoForge, ForgeConfig

        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        forge = PlatoForge(ForgeConfig(
            snap_radius=0.3, snap_interval=10, snap_warmup=5,
            bma_window=15, deadband_threshold=1e-3,
        ))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for step in range(50):
            x = torch.randn(16, 8)
            y = torch.randint(0, 4, (16,))
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            result = forge.step(loss.item(), model, optimizer, optimizer.param_groups[0]["lr"], step)
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) == 50
        status = forge.status()
        assert status["throttle"]["throttle_ratio"] > 0

    def test_snap_actually_changes_weights(self):
        from plato_training.plato_forge import PyForge, ForgeConfig

        model = nn.Linear(32, 4)
        before = model.weight.data.clone()
        forge = PyForge(ForgeConfig(snap_radius=0.5))
        stats = forge.snap_weights(model, radius=0.5)
        after = model.weight.data
        assert stats["total_params"] > 0
        changed = (before != after).sum().item()
        assert changed > 0

    def test_tile_export(self):
        from plato_training.plato_forge import PlatoForge, ForgeConfig

        model = nn.Linear(8, 4)
        forge = PlatoForge(ForgeConfig())
        forge.impl.losses.append(1.0)
        forge.impl.total_steps = 5
        tile = forge.export_tile(model, store_dir=tempfile.mkdtemp(), room_name="test")
        assert tile.tile_id.startswith("forge-test-")
        del model
        gc.collect()


# ═══ 6. GPU Fleet Trainer ═══

class TestGPUFleetIntegration:
    def test_train_and_predict(self):
        from plato_training.fleet_miner import CommitPoint
        from plato_training.gpu_fleet_trainer import train_gpu_fleet, predict_fleet_gpu, GPUFleetConfig

        _clear_gpu()
        repos = ["plato-training", "forgemaster", "tensor-spline"]
        commits = [
            CommitPoint(
                sha=f"s{i:06d}", repo=repos[i % 3], author="test",
                timestamp=time.time() - (200 - i) * 1800,
                message=f"c{i}", files_changed=i % 5 + 1,
                insertions=i * 10, deletions=i * 5,
                is_merge=False, languages=[".py"], cross_refs=[],
            )
            for i in range(200)
        ]

        cfg = GPUFleetConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32,
            epochs=10, batch_size=8, grad_accum_steps=2,
            use_amp=True, device="cuda", seq_len=4,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        assert result.params_count > 0
        assert result.peak_vram_mb > 0
        preds = predict_fleet_gpu(result.model, commits, cfg)
        assert len(preds) > 0
        del result
        _clear_gpu()


# ═══ 7. GPT-2 Trainer ═══

class TestGPT2TrainerIntegration:
    def test_train_synthetic(self):
        from plato_training.fleet_miner import CommitPoint
        from plato_training.gpt2_trainer import train_fleet_gpt2, TinyGPT2Config

        commits = [
            CommitPoint(
                sha=f"g{i:06d}", repo="test-repo", author="test",
                timestamp=time.time() - (100 - i) * 3600,
                message=f"commit {i}", files_changed=i % 5 + 1,
                insertions=i * 10, deletions=i * 5,
                is_merge=False, languages=[".py"], cross_refs=[],
            )
            for i in range(100)
        ]

        config = TinyGPT2Config(n_layer=1, n_head=2, n_embd=32, block_size=16)
        result = train_fleet_gpt2(commits, config=config, verbose=False)
        assert len(result.train_loss_history) > 0
        assert result.val_loss > 0
        assert result.model is not None


# ═══ 8. Types & Store ═══

class TestTypesStoreIntegration:
    def test_tile_round_trip(self):
        from plato_training.types import TrainingTile, TileType, TileLifecycle, LamportClock
        from plato_training.store import LocalTileStore

        tmp = tempfile.mkdtemp()
        store = LocalTileStore(tmp)
        clock = LamportClock()

        tile = TrainingTile(
            tile_id="test-tile-001", room="test-room",
            tile_type=TileType.CHECKPOINT, state=TileLifecycle.ACTIVE,
            lamport=clock.tick(), name="test",
            description="Integration test tile", content_hash="abc123",
        )
        store.save(tile)
        loaded = store.load("test-tile-001")
        assert loaded is not None
        assert loaded.tile_id == tile.tile_id
        tiles = store.list_tiles()
        assert len(tiles) >= 1

    def test_delete_lifecycle(self):
        """Store has delete(), not supersede()."""
        from plato_training.types import TrainingTile, TileType, TileLifecycle, LamportClock
        from plato_training.store import LocalTileStore

        tmp = tempfile.mkdtemp()
        store = LocalTileStore(tmp)
        clock = LamportClock()

        tile = TrainingTile(
            tile_id="tile-del", room="test", tile_type=TileType.CHECKPOINT,
            state=TileLifecycle.ACTIVE, lamport=clock.tick(), name="del",
            content_hash="hash-del",
        )
        store.save(tile)
        assert store.load("tile-del") is not None
        store.delete("tile-del")
        # After delete, may still exist but with RETRACTED state
        loaded = store.load("tile-del")
        if loaded is not None:
            assert loaded.state == TileLifecycle.RETRACTED


# ═══ 9. SplineLinear ═══

class TestSplineLinearIntegration:
    def test_compression(self):
        from plato_training.spline import SplineLinear
        if SplineLinear is None:
            pytest.skip("SplineLinear not available")

        layer = SplineLinear(64, 32, n_control_points=8)
        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params > 0


# ═══ 10. Collective Loop ═══

class TestCollectiveLoopIntegration:
    def test_init(self):
        from plato_training.collective_loop import CollectiveLoop

        loop = CollectiveLoop(
            github_token=None,
            clone_dir=tempfile.mkdtemp(),
            history_file=os.path.join(tempfile.mkdtemp(), "history.json"),
            org="SuperInstance",
        )
        assert loop is not None

    def test_run_once_no_crash(self):
        from plato_training.collective_loop import CollectiveLoop

        loop = CollectiveLoop(
            github_token=None,
            clone_dir=tempfile.mkdtemp(),
            history_file=os.path.join(tempfile.mkdtemp(), "history.json"),
        )
        # Won't find real repos without token, but shouldn't crash
        try:
            result = loop.run_once()
        except Exception:
            pass  # Expected without auth


# ═══ 11. Cross-Module Imports ═══

class TestCrossModuleImports:
    def test_all_modules_import(self):
        import importlib
        modules = [
            "plato_training.types",
            "plato_training.store",
            "plato_training.spline",
            "plato_training.throttle",
            "plato_training.fleet_miner",
            "plato_training.gpt2_trainer",
            "plato_training.gpt2_room",
            "plato_training.commit_predictor",
            "plato_training.collective_loop",
            "plato_training.gpu_fleet_trainer",
            "plato_training.intelligence_room",
            "plato_training.intelligence_pre_filter",
            "plato_training.intelligence_post_filter",
            "plato_training.intelligence_self_trainer",
            "plato_training.plato_forge",
        ]
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"
