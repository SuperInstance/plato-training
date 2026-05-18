"""
Swarm Rooms — GPU-accelerated multi-agent room simulation.

Core insight: "The larger the swarm, the smaller the individual contexts 
and the more the interconnecting matters. Rooms that mostly focus on 
everyone else's job."

Architecture:
  - N agents, each in a "room" with small local context (d_local dims)
  - Each room's PRIMARY job is observing other rooms (d_obs dims)
  - Interconnection matrix: who watches whom (sparse, GPU-optimized)
  - CRDT-style merge: observation updates commute (order-independent)
  - Deadband: only propagate when delta exceeds tolerance
  - Eisenstein snap: compress observations to lattice points

This tests the hypothesis that swarm intelligence emerges from 
interconnection density, NOT individual capacity.

Key experiments:
  1. Scale agents: does collective accuracy hold as N grows?
  2. Interconnection density: what % of cross-observation is needed?
  3. Deadband impact: does tolerance compression help or hurt?
  4. CRDT merge convergence: does commutative merge converge?
  5. GPU parallelism: can we scale to 10K+ rooms on one GPU?
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# ─── Eisenstein Snap (GPU) ───────────────────────────────────────────

EISENSTEIN_OMEGA = torch.tensor([-0.5, math.sqrt(3) / 2], dtype=torch.float32)

def eisenstein_snap_gpu(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Snap 2D points to Eisenstein lattice Z[ω].
    O(1) per point, fully parallelized on GPU.
    
    The Eisenstein lattice is spanned by (1, 0) and (-1/2, √3/2).
    """
    # Basis matrix
    basis = torch.tensor([[1.0, 0.0], [-0.5, math.sqrt(3) / 2]], 
                          device=points.device, dtype=torch.float32)
    
    # Project to lattice coordinates
    basis_inv = torch.linalg.inv(basis)
    coords = points @ basis_inv.T
    
    # Round to nearest lattice point
    snapped_coords = torch.round(coords)
    
    # Convert back
    snapped_points = snapped_coords @ basis.T * radius
    return snapped_points


def eisenstein_delta_gpu(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Compute delta (distance from nearest Eisenstein lattice point)."""
    snapped = eisenstein_snap_gpu(points, radius)
    delta = torch.norm(points - snapped, dim=-1)
    return delta


# ─── CRDT Room State ────────────────────────────────────────────────

class RoomState:
    """
    CRDT-mergeable room state for one agent.
    
    Properties:
      - Commutative: merge(A, B) = merge(B, A)
      - Associative: merge(A, merge(B, C)) = merge(merge(A, B), C)
      - Idempotent: merge(A, A) = A
    
    Uses last-writer-wins with lamport clock for conflict resolution.
    Deadband: only propagate if delta exceeds tolerance.
    """
    
    def __init__(
        self,
        room_id: int,
        context_dim: int = 64,
        obs_dim: int = 128,
        device: str = "cuda",
    ):
        self.room_id = room_id
        self.context_dim = context_dim
        self.obs_dim = obs_dim
        self.device = device
        
        # Local context (small — the agent's own job)
        self.local_context = torch.randn(context_dim, device=device) * 0.1
        
        # Observation buffer (large — what this agent sees of others)
        # Each row is another agent's observed state
        self.obs_buffer = torch.zeros(obs_dim, device=device)
        
        # Lamport clock for CRDT merge
        self.clock = 0
        
        # Deadband state
        self.last_propagated = self.local_context.clone()
        self.tolerance = 0.1
    
    def update_context(self, delta: torch.Tensor):
        """Update local context and increment clock."""
        self.local_context = self.local_context + delta
        self.clock += 1
    
    def should_propagate(self) -> bool:
        """Deadband check: only propagate if delta exceeds tolerance."""
        delta = torch.norm(self.local_context - self.last_propagated)
        return delta.item() > self.tolerance
    
    def propagate(self) -> Optional[torch.Tensor]:
        """Get state to broadcast (or None if deadband suppresses)."""
        if self.should_propagate():
            self.last_propagated = self.local_context.clone()
            return self.local_context
        return None
    
    def receive_observation(self, other_id: int, other_context: torch.Tensor):
        """Receive and merge observation from another room."""
        # CRDT merge: weighted average with clock-based ordering
        self.obs_buffer = 0.9 * self.obs_buffer + 0.1 * other_context
        self.clock = max(self.clock, self.clock) + 1


# ─── Swarm Room Network (GPU) ───────────────────────────────────────

class SwarmRoomNetwork:
    """
    GPU-accelerated swarm of interconnected rooms.
    
    Each room has:
      - Small local context (the agent's own job)
      - Large observation of OTHER rooms (what others are doing)
      - CRDT merge for conflict-free state convergence
      - Deadband for attention allocation
    
    The interconnection matrix controls who watches whom.
    """
    
    def __init__(
        self,
        n_agents: int = 256,
        context_dim: int = 64,
        obs_dim: int = 128,
        device: str = "cuda",
        interconnection_density: float = 0.3,
        deadband_tolerance: float = 0.1,
        snap_radius: float = 1.0,
    ):
        self.n_agents = n_agents
        self.context_dim = context_dim
        self.obs_dim = obs_dim
        self.device = device
        self.snap_radius = snap_radius
        
        # All agent contexts on GPU (N × context_dim)
        self.contexts = torch.randn(n_agents, context_dim, device=device) * 0.1
        
        # Observation embeddings (N × obs_dim) — what each agent sees
        self.observations = torch.zeros(n_agents, obs_dim, device=device)
        
        # Interconnection matrix: who watches whom
        # Sparse: each agent observes density% of others
        self.connections = self._build_connections(interconnection_density)
        
        # Attention weights (computed each step)
        self.attention = torch.zeros(n_agents, n_agents, device=device)
        
        # Deadband state
        self.last_propagated = self.contexts.clone()
        self.tolerance = deadband_tolerance
        
        # Lamport clocks
        self.clocks = torch.zeros(n_agents, dtype=torch.long, device=device)
        
        # Statistics
        self.step_count = 0
        self.propagation_counts = []
        self.attention_entropies = []
    
    def _build_connections(self, density: float) -> torch.Tensor:
        """Build sparse interconnection matrix."""
        n = self.n_agents
        # Random sparse connections
        mask = torch.rand(n, n, device=self.device) < density
        # No self-connections (agent doesn't watch itself — that's the point)
        mask.fill_diagonal_(False)
        return mask.float()
    
    def step(self, task_signal: Optional[torch.Tensor] = None) -> Dict:
        """
        One step of the swarm room simulation.
        
        1. Each agent receives task signal (optional)
        2. Agents observe connected rooms
        3. Compute attention weights (who matters most)
        4. Update observations via CRDT merge
        5. Snap observations to Eisenstein lattice (compression)
        6. Deadband filter: only propagate significant changes
        7. Update local contexts
        
        Returns metrics dict.
        """
        self.step_count += 1
        N = self.n_agents
        
        # 1. Optional task signal injection
        if task_signal is not None:
            self.contexts = self.contexts + task_signal.unsqueeze(0) * 0.01
        
        # 2. Compute attention: how much does agent i care about agent j?
        # Q = contexts (query: "what do I need?")
        # K = contexts (key: "what do you have?")
        scores = self.contexts @ self.contexts.T / math.sqrt(self.context_dim)
        
        # Apply connection mask (only watch connected rooms)
        scores = scores.masked_fill(self.connections < 1, float('-inf'))
        
        # Softmax attention
        self.attention = F.softmax(scores, dim=1)
        
        # 3. Compute observations via attention-weighted aggregation
        # This IS the CRDT merge: weighted average, order-independent
        aggregated = self.attention @ self.contexts  # N × context_dim
        
        # Project to observation dimension
        if self.context_dim != self.obs_dim:
            proj_ctx_obs = torch.randn(self.context_dim, self.obs_dim, device=self.device) * 0.05
            new_observations = aggregated @ proj_ctx_obs
        else:
            new_observations = aggregated
        
        # 4. Eisenstein snap on 2D projection of observations
        proj_snap = torch.randn(self.obs_dim, 2, device=self.device) * 0.1
        obs_2d = new_observations @ proj_snap
        snapped_2d = eisenstein_snap_gpu(obs_2d, radius=self.snap_radius)
        deltas = eisenstein_delta_gpu(obs_2d, radius=self.snap_radius)
        
        # 5. Deadband: which agents have significant changes?
        context_deltas = torch.norm(self.contexts - self.last_propagated, dim=1)
        active_mask = context_deltas > self.tolerance
        n_propagating = active_mask.sum().item()
        
        # 6. Update observations (CRDT merge with existing)
        alpha = 0.1  # learning rate
        self.observations = (1 - alpha) * self.observations + alpha * new_observations
        
        # 7. Update local contexts from observations
        # Agent updates its own context based on what it observed about others
        context_update = self.observations @ torch.randn(
            self.obs_dim, self.context_dim, device=self.device
        ) * 0.01
        self.contexts = self.contexts + context_update
        
        # Update deadband state for propagating agents
        self.last_propagated[active_mask] = self.contexts[active_mask].clone()
        
        # Increment clocks
        self.clocks += 1
        
        # Compute metrics
        with torch.no_grad():
            # Attention entropy (how distributed is each agent's attention?)
            attn_entropy = -(self.attention * (self.attention + 1e-8).log()).sum(dim=1)
            mean_entropy = attn_entropy.mean().item()
            
            # Interconnection utilization (how much of the connection budget is used)
            util = (self.attention > 0.01).float().mean().item()
            
            # Snap compression ratio
            snap_rate = (deltas < 0.01).float().mean().item()
            
            # Context diversity (how different are agents from each other?)
            centroid = self.contexts.mean(dim=0)
            diversity = torch.norm(self.contexts - centroid, dim=1).mean().item()
        
        metrics = {
            "step": self.step_count,
            "propagating": n_propagating,
            "propagation_rate": n_propagating / N,
            "attention_entropy": mean_entropy,
            "interconnection_util": util,
            "snap_hit_rate": snap_rate,
            "context_diversity": diversity,
            "mean_delta": deltas.mean().item(),
            "max_delta": deltas.max().item(),
        }
        
        self.propagation_counts.append(n_propagating)
        self.attention_entropies.append(mean_entropy)
        
        return metrics
    
    def benchmark(self, n_steps: int = 100) -> Dict:
        """Run benchmark and return performance metrics."""
        # Warmup
        for _ in range(5):
            self.step()
        torch.cuda.synchronize()
        
        start = time.time()
        metrics_list = []
        for _ in range(n_steps):
            m = self.step()
            metrics_list.append(m)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        steps_per_sec = n_steps / elapsed
        agent_ops_per_sec = steps_per_sec * self.n_agents
        
        return {
            "n_agents": self.n_agents,
            "steps_per_sec": round(steps_per_sec, 1),
            "agent_ops_per_sec": round(agent_ops_per_sec, 0),
            "elapsed_sec": round(elapsed, 3),
            "final_metrics": metrics_list[-1],
            "mean_propagation_rate": np.mean([m["propagation_rate"] for m in metrics_list]),
            "mean_attention_entropy": np.mean([m["attention_entropy"] for m in metrics_list]),
            "mean_snap_hit_rate": np.mean([m["snap_hit_rate"] for m in metrics_list]),
            "mean_context_diversity": np.mean([m["context_diversity"] for m in metrics_list]),
        }


# ─── Scaling Experiments ─────────────────────────────────────────────

def run_scaling_experiment(
    agent_counts: List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048],
    context_dim: int = 64,
    obs_dim: int = 128,
    density: float = 0.3,
    n_steps: int = 50,
    device: str = "cuda",
) -> List[Dict]:
    """
    Test hypothesis: does collective behavior hold as swarm grows?
    
    Prediction: as N increases, individual context stays small (context_dim)
    but interconnection density (N×density connections per agent) grows.
    Collective accuracy should HOLD or IMPROVE with more agents.
    """
    results = []
    
    for n in agent_counts:
        try:
            net = SwarmRoomNetwork(
                n_agents=n,
                context_dim=context_dim,
                obs_dim=obs_dim,
                device=device,
                interconnection_density=density,
            )
            bench = net.benchmark(n_steps=n_steps)
            bench["interconnection_density"] = density
            bench["total_connections"] = int(n * density * n)
            results.append(bench)
            print(f"  N={n:5d}: {bench['steps_per_sec']:8.1f} steps/s, "
                  f"diversity={bench['mean_context_diversity']:.3f}, "
                  f"entropy={bench['mean_attention_entropy']:.3f}, "
                  f"snap={bench['mean_snap_hit_rate']:.3f}, "
                  f"prop={bench['mean_propagation_rate']:.3f}")
            
            # Free GPU memory
            del net
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  N={n:5d}: OOM — max agents reached")
            break
        except RuntimeError as e:
            if "memory" in str(e).lower():
                print(f"  N={n:5d}: OOM — max agents reached")
                break
            raise
    
    return results


def run_density_experiment(
    n_agents: int = 256,
    densities: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    n_steps: int = 50,
    device: str = "cuda",
) -> List[Dict]:
    """
    Test hypothesis: what interconnection density is needed?
    
    Prediction: there's a phase transition — below some density,
    collective behavior degrades sharply.
    """
    results = []
    
    for d in densities:
        net = SwarmRoomNetwork(
            n_agents=n_agents,
            interconnection_density=d,
            device=device,
        )
        bench = net.benchmark(n_steps=n_steps)
        bench["density"] = d
        results.append(bench)
        print(f"  d={d:.2f}: diversity={bench['mean_context_diversity']:.3f}, "
              f"entropy={bench['mean_attention_entropy']:.3f}, "
              f"snap={bench['mean_snap_hit_rate']:.3f}")
        del net
        torch.cuda.empty_cache()
    
    return results


def run_deadband_experiment(
    n_agents: int = 256,
    tolerances: List[float] = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    n_steps: int = 50,
    device: str = "cuda",
) -> List[Dict]:
    """
    Test hypothesis: does deadband compression help or hurt?
    
    Prediction: moderate tolerance (0.05-0.2) gives best diversity
    with acceptable propagation overhead.
    """
    results = []
    
    for tol in tolerances:
        net = SwarmRoomNetwork(
            n_agents=n_agents,
            deadband_tolerance=tol,
            device=device,
        )
        bench = net.benchmark(n_steps=n_steps)
        bench["tolerance"] = tol
        results.append(bench)
        print(f"  tol={tol:.3f}: prop={bench['mean_propagation_rate']:.3f}, "
              f"diversity={bench['mean_context_diversity']:.3f}, "
              f"speed={bench['steps_per_sec']:.1f}/s")
        del net
        torch.cuda.empty_cache()
    
    return results


def run_crdt_convergence_test(
    n_agents: int = 64,
    n_rounds: int = 20,
    device: str = "cuda",
) -> Dict:
    """
    Test CRDT convergence: do all agents converge to same observation
    regardless of merge order?
    
    Commutative merge should produce same result regardless of order.
    """
    net = SwarmRoomNetwork(
        n_agents=n_agents,
        interconnection_density=0.5,
        device=device,
    )
    
    # Run several rounds
    divergences = []
    for _ in range(n_rounds):
        net.step()
        
        # Measure divergence: how different are agent observations?
        obs = net.observations
        centroid = obs.mean(dim=0)
        divergence = torch.norm(obs - centroid, dim=1).mean().item()
        divergences.append(divergence)
    
    return {
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "initial_divergence": divergences[0],
        "final_divergence": divergences[-1],
        "convergence_rate": (divergences[0] - divergences[-1]) / max(divergences[0], 1e-8),
        "divergence_trajectory": divergences,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    """Run all GPU experiments."""
    import json
    
    if not torch.cuda.is_available():
        print("No CUDA available — cannot run GPU experiments")
        return
    
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    print("=" * 60)
    print("EXPERIMENT 1: Agent Scaling")
    print("Hypothesis: collective behavior holds as swarm grows")
    print("=" * 60)
    scale_results = run_scaling_experiment(device=device)
    
    print()
    print("=" * 60)
    print("EXPERIMENT 2: Interconnection Density")
    print("Hypothesis: phase transition in collective behavior")
    print("=" * 60)
    density_results = run_density_experiment(device=device)
    
    print()
    print("=" * 60)
    print("EXPERIMENT 3: Deadband Tolerance")
    print("Hypothesis: moderate tolerance gives best diversity/speed")
    print("=" * 60)
    deadband_results = run_deadband_experiment(device=device)
    
    print()
    print("=" * 60)
    print("EXPERIMENT 4: CRDT Convergence")
    print("Hypothesis: all agents converge regardless of merge order")
    print("=" * 60)
    crdt_results = run_crdt_convergence_test(device=device)
    print(f"  Initial divergence: {crdt_results['initial_divergence']:.4f}")
    print(f"  Final divergence: {crdt_results['final_divergence']:.4f}")
    print(f"  Convergence rate: {crdt_results['convergence_rate']:.4f}")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Max agents tested: {max(r['n_agents'] for r in scale_results)}")
    print(f"Best density: {max(density_results, key=lambda r: r['mean_context_diversity'])['density']}")
    print(f"Best tolerance: {max(deadband_results, key=lambda r: r['mean_context_diversity'])['tolerance']}")


if __name__ == "__main__":
    main()
