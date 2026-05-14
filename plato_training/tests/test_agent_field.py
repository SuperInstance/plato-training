"""
Tests for AgentField — shared tensor field for within-agent room coordination.

Tests verify that within-agent rooms don't need message passing.
They share the same tensor. Communication IS the shared state.
"""

import pytest
from plato_training.agent_field import AgentField, CHANNEL_NAMES


class TestAgentFieldBasics:
    def test_create_field(self):
        field = AgentField()
        assert field.n_rooms == 0
        assert field.tick_count == 0
    
    def test_add_room(self):
        field = AgentField()
        idx = field.add_room("sensor-0", role="sensor")
        assert idx == 0
        assert field.n_rooms == 1
    
    def test_add_multiple_rooms(self):
        field = AgentField()
        s = field.add_room("sensor", role="sensor")
        p = field.add_room("predictor", role="predictor")
        c = field.add_room("comparator", role="comparator")
        assert s == 0
        assert p == 1
        assert c == 2
        assert field.n_rooms == 3
    
    def test_name_resolution(self):
        field = AgentField()
        field.add_room("drift-sensor")
        assert field.idx("drift-sensor") == 0
        assert field.idx(0) == 0


class TestStateAccess:
    def test_initial_state_is_zero(self):
        from flux_tensor_midi import FluxVector
        field = AgentField()
        field.add_room("test")
        fv = field.get_state("test")
        assert all(v == 0.0 for v in fv.values)
    
    def test_set_get_state(self):
        from flux_tensor_midi import FluxVector
        field = AgentField()
        field.add_room("test")
        
        fv = FluxVector([0.5, 0.3, 0.1, 0.9, 0.0, 1.0, 0.2, 0.0, 0.0])
        field.set_state("test", fv)
        
        result = field.get_state("test")
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[3] - 0.9) < 1e-10
    
    def test_single_channel_write(self):
        field = AgentField()
        field.add_room("test")
        field.set_channel("test", 4, 0.75)  # gap channel
        assert abs(field.get_channel("test", 4) - 0.75) < 1e-10
    
    def test_sensor_write(self):
        field = AgentField()
        field.add_room("sensor", role="sensor")
        field.sensor_write("sensor", [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.5])
        assert abs(field.get_channel("sensor", 0) - 0.8) < 1e-10
        assert abs(field.get_channel("sensor", 8) - 0.0) < 1e-10  # phase reset to 0
    
    def test_predict_write(self):
        field = AgentField()
        field.add_room("predictor", role="predictor")
        field.predict_write("predictor", 0.9, [0.0]*9)
        assert abs(field.get_channel("predictor", 0) - 0.9) < 1e-10  # confidence set
        assert abs(field.get_channel("predictor", 8) - 0.25) < 1e-10  # phase = predicted


class TestCoupling:
    def test_coupling_setup(self):
        field = AgentField()
        s = field.add_room("sensor")
        p = field.add_room("predictor")
        field.couple(p, s, strength=0.9)
        assert abs(field.get_coupling(p, s) - 0.9) < 1e-10
        assert field.get_coupling(s, p) == 0.0  # unidirectional
    
    def test_decouple(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.couple(a, b, 0.5)
        field.decouple(a, b)
        assert field.get_coupling(a, b) == 0.0


class TestSideChannels:
    def test_nod_increases_coupling(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.couple(a, b, 0.3)
        field.nod(a, b, intensity=0.2)
        assert abs(field.get_coupling(a, b) - 0.5) < 1e-10
    
    def test_nod_clamps_at_1(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.couple(a, b, 0.9)
        field.nod(a, b, intensity=0.3)
        assert abs(field.get_coupling(a, b) - 1.0) < 1e-10
    
    def test_frown_decreases_coupling_and_raises_gap(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.couple(a, b, 0.5)
        initial_gap = field.get_channel(a, 4)
        field.frown(a, b, intensity=0.2)
        assert abs(field.get_coupling(a, b) - 0.3) < 1e-10
        assert field.get_channel(a, 4) > initial_gap
    
    def test_smile_shifts_state(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.set_channel(b, 0, 1.0)  # b has confidence=1.0
        field.smile(a, b, intensity=0.2)
        # a should have moved toward b's state
        assert field.get_channel(a, 0) > 0  # shifted toward b's 1.0


class TestTickUpdate:
    def test_tick_advances_clock(self):
        field = AgentField(bpm=120.0)
        field.add_room("test")
        ts = field.tick()
        assert field.tick_count == 1
        assert ts > 0
    
    def test_tick_propagates_coupling(self):
        field = AgentField(damping=0.5)
        a = field.add_room("a")
        b = field.add_room("b")
        
        # a has state [1, 0, 0, ...], b has [0, 0, 0, ...]
        field.set_channel(a, 0, 1.0)
        field.couple(b, a, strength=1.0)  # b coupled to a
        
        field.tick()
        
        # b should have been pulled toward a
        b_conf = field.get_channel(b, 0)
        assert b_conf > 0, "Coupling should propagate state"
    
    def test_tick_updates_phase(self):
        field = AgentField()
        field.add_room("test")
        initial_phase = field.get_channel("test", 8)
        field.tick()
        new_phase = field.get_channel("test", 8)
        # Phase should advance by 0.25
        assert abs(new_phase - (initial_phase + 0.25) % 1.0) < 1e-10


class TestCoherence:
    def test_single_room_is_coherent(self):
        field = AgentField()
        field.add_room("only")
        assert field.coherence() == 1.0
    
    def test_identical_zero_rooms_have_zero_coherence(self):
        """Zero vectors have no direction → cosine undefined → coherence 0.
        This is correct: two rooms with no state have no meaningful coherence."""
        field = AgentField()
        field.add_room("a")
        field.add_room("b")
        assert field.coherence() == 0.0  # zero vectors → undefined cosine
    
    def test_identical_nonzero_rooms_are_coherent(self):
        field = AgentField()
        field.add_room("a")
        field.add_room("b")
        # Give both rooms the same non-zero state
        field.sensor_write(0, [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        field.sensor_write(1, [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(field.coherence() - 1.0) < 1e-10
    
    def test_divergent_rooms_lower_coherence(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        
        # a points one way, b points another
        field.sensor_write(a, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        field.sensor_write(b, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        coh = field.coherence()
        assert coh < 1.0  # orthogonal rooms
    
    def test_pairwise_coherence(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        
        field.sensor_write(a, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        field.sensor_write(b, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        assert abs(field.room_coherence(a, b) - 1.0) < 1e-10


class TestGaps:
    def test_no_gaps_initially(self):
        field = AgentField()
        field.add_room("test")
        assert field.gaps() == []
    
    def test_gap_detected(self):
        field = AgentField()
        field.add_room("test")
        field.set_channel("test", 4, 0.5)  # gap channel
        # tolerance default is 0.01, so 0.5 > 0.01
        gaps = field.gaps()
        assert len(gaps) == 1
        assert gaps[0][1] == 0.5
    
    def test_focus_queue(self):
        field = AgentField()
        field.add_room("high-gap", initial_state=[0.9, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        field.add_room("low-gap", initial_state=[0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        
        fq = field.focus_queue()
        assert len(fq) == 2
        assert fq[0][0] == "high-gap"  # 0.9 * 0.8 = 0.72
        assert fq[1][0] == "low-gap"   # 0.5 * 0.2 = 0.10


class TestWithinTolerance:
    def test_zero_rooms_are_tolerant(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        assert field.within_tolerance(a, b) is True
    
    def test_small_diff_within_tolerance(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.set_channel(a, 0, 0.005)
        field.set_channel(b, 0, 0.006)
        # tolerance is 0.01, diff is 0.001
        assert field.within_tolerance(a, b) is True
    
    def test_large_diff_exceeds_tolerance(self):
        field = AgentField()
        a = field.add_room("a")
        b = field.add_room("b")
        field.set_channel(a, 0, 0.0)
        field.set_channel(b, 0, 0.5)
        # tolerance is 0.01, diff is 0.5
        assert field.within_tolerance(a, b) is False


class TestChirality:
    def test_initial_chirality_is_exploring(self):
        field = AgentField()
        field.add_room("test")
        assert field.chirality("test") == "exploring"
    
    def test_chirality_transitions(self):
        field = AgentField()
        idx = field.add_room("test")
        meta = field._meta[idx]
        
        # 3 ticks with low gap → locking
        for _ in range(3):
            meta.ticks += 1
        field.update_chirality("test")
        assert field.chirality("test") == "locking"
        
        # 10 ticks with low gap → locked
        meta.ticks = 10
        field.update_chirality("test")
        assert field.chirality("test") == "locked"
        
        # gap spike → back to exploring
        field.set_channel("test", 4, 0.5)
        field.update_chirality("test")
        assert field.chirality("test") == "exploring"


class TestIntegration:
    def test_sensor_predict_compare_loop(self):
        """The fundamental within-agent loop:
        sensor reads → predictor predicts → comparator gaps
        All through shared tensor, no message passing.
        """
        field = AgentField(bpm=60.0, damping=0.3)
        s = field.add_room("sensor", role="sensor")
        p = field.add_room("predictor", role="predictor")
        c = field.add_room("comparator", role="comparator")
        
        # Wire coupling
        field.couple(p, s, 0.5)   # predictor watches sensor
        field.couple(c, s, 0.7)   # comparator watches sensor
        field.couple(c, p, 0.7)   # comparator watches predictor
        
        # Step 1: Sensor reads a value
        field.sensor_write(s, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Step 2: Predictor makes a prediction
        field.predict_write(p, 0.9, [0.75, 0.2, 0.01, 0.8, 0.0, 0.9, 0.0, 0.0, 0.25])
        
        # Step 3: Tick — coupling propagates
        field.tick()
        
        # The comparator should have been influenced by both sensor and predictor
        c_state = field.get_state(c)
        # It should have non-zero state from coupling
        assert any(v != 0 for v in c_state.values)
        
        # Step 4: Now sensor reads a DIFFERENT value (gap!)
        field.sensor_write(s, [0.3, 0.6, 0.05, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Comparator detects gap
        from flux_tensor_midi import FluxVector
        sensor_state = field.get_state(s)
        predict_state = field.get_state(p)
        distance = sensor_state.distance_to(predict_state)
        
        assert distance > 0, "Sensor and predictor should diverge"
        
        # Write the gap to comparator's gap channel
        field.set_channel(c, 4, min(distance, 1.0))
        
        # Focus queue should show the comparator
        fq = field.focus_queue()
        assert len(fq) > 0
    
    def test_report(self):
        field = AgentField()
        field.add_room("sensor", role="sensor")
        field.add_room("predictor", role="predictor")
        report = field.field_report()
        assert "sensor" in report
        assert "predictor" in report
        assert "Coherence" in report


class TestChannelNames:
    def test_nine_channels(self):
        assert len(CHANNEL_NAMES) == 9
    
    def test_expected_names(self):
        assert CHANNEL_NAMES[0] == "confidence"
        assert CHANNEL_NAMES[4] == "gap"
        assert CHANNEL_NAMES[8] == "phase"
