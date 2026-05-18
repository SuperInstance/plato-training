"""
Collective Inference Loop — continuous predict → observe → gap → learn cycle.

Wires fleet_miner (real git data) + collective (room architecture) + GPT-2
into a running system that:
  1. OBSERVE: mines recent commits from fleet repos
  2. PREDICT: uses trained GPT-2 to predict next activity
  3. COMPARE: prediction vs reality → gap score
  4. LEARN: gaps become focus items for the next cycle
  5. SHARE: results written as PLATO tiles

This is the bridge from static data mining to live collective intelligence.
"""

import json
import time
import math
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from .fleet_miner import FleetMiner, CommitPoint, SynergyEvent, RepoSignal
# Collective room types (for future integration)
# from .collective import RoomKind, RoomAddress, GapSignal, FocusQueue


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class CycleResult:
    """One complete predict → observe → compare → learn cycle."""
    cycle_id: str
    timestamp: float
    repos_observed: int
    commits_observed: int
    predictions_made: int
    predictions_correct: int
    predictions_missed: int
    gap_score: float              # 0.0 = perfect prediction, 1.0 = total miss
    focus_items: List[Dict]       # gaps surfaced for next cycle
    top_synergies: List[Dict]     # strongest cross-repo signals
    velocity_by_repo: Dict[str, float]  # commits/hour per repo
    transfer_entropy: float = 0.0       # CCC-style TE between agents
    source_entropy: float = 0.0         # fleet source diversity
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PredictionEntry:
    """A single prediction about fleet activity."""
    prediction_id: str
    repo: str
    predicted_at: float
    predicted_activity: str      # "commit", "cross_ref", "spike"
    confidence: float
    time_window_hours: float
    context: str                  # what GPT-2 generated as reasoning
    
    # Filled in after observation
    observed: bool = False
    correct: bool = False
    observed_at: float = 0.0


# ─── Activity Patterns ──────────────────────────────────────────────

# Simple heuristic predictors (no model needed for baseline)
# GPT-2 can be added for richer predictions

def compute_repo_velocity(commits: List[CommitPoint], hours: float = 24.0) -> Dict[str, float]:
    """Compute commits-per-hour for each repo."""
    now = time.time()
    window = hours * 3600
    counts = defaultdict(int)
    
    for c in commits:
        if now - c.timestamp < window:
            counts[c.repo] += 1
    
    return {repo: count / hours for repo, count in counts.items()}


def detect_activity_spike(
    current_velocity: Dict[str, float],
    baseline_velocity: Dict[str, float],
    threshold: float = 2.0,
) -> List[str]:
    """Find repos where current velocity exceeds baseline by threshold×."""
    spikes = []
    for repo, vel in current_velocity.items():
        base = baseline_velocity.get(repo, 0.1)  # default low baseline
        if vel > base * threshold:
            spikes.append(repo)
    return spikes


def predict_commit_probability(
    repo: str,
    velocity: Dict[str, float],
    hours_ahead: float = 1.0,
    decay: float = 0.5,
) -> float:
    """
    Simple exponential decay model for commit probability.
    P(commit in next h hours) = 1 - exp(-λ * h)
    where λ = velocity (commits/hour)
    """
    lam = velocity.get(repo, 0.0)
    if lam <= 0:
        return 0.0
    return 1.0 - math.exp(-lam * decay * hours_ahead)


# ─── Collective Loop ────────────────────────────────────────────────

class CollectiveLoop:
    """
    Continuous collective inference loop.
    
    Usage:
        loop = CollectiveLoop(
            github_token="gho_...",
            plato_url="http://147.224.38.131:8847",
        )
        
        # Single cycle
        result = loop.run_cycle()
        
        # Continuous
        loop.run_forever(interval_minutes=30)
    """
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        plato_url: str = "http://147.224.38.131:8847",
        clone_dir: str = "/tmp/fleet-mine",
        history_file: str = "/tmp/collective-history.json",
        org: str = "SuperInstance",
    ):
        self.miner = FleetMiner(org=org, token=github_token, clone_dir=clone_dir)
        self.plato_url = plato_url
        self.history_file = Path(history_file)
        
        # State
        self.all_commits: List[CommitPoint] = []
        self.baseline_velocity: Dict[str, float] = {}
        self.predictions: List[PredictionEntry] = []
        self.cycle_count = 0
        self.cumulative_gap = 0.0
        
        # Load history if exists
        self._load_history()
    
    def _load_history(self):
        """Load previous cycle history."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self.baseline_velocity = data.get("baseline_velocity", {})
                self.cycle_count = data.get("cycle_count", 0)
                self.cumulative_gap = data.get("cumulative_gap", 0.0)
                # Restore pending predictions
                for p in data.get("pending_predictions", []):
                    self.predictions.append(PredictionEntry(**p))
            except (json.JSONDecodeError, TypeError):
                pass
    
    def _save_history(self):
        """Persist cycle state."""
        data = {
            "baseline_velocity": self.baseline_velocity,
            "cycle_count": self.cycle_count,
            "cumulative_gap": self.cumulative_gap,
            "last_cycle": time.time(),
            "pending_predictions": [
                {k: v for k, v in asdict(p).items() if not p.observed}
                for p in self.predictions if not p.observed
            ],
        }
        self.history_file.write_text(json.dumps(data, indent=2))
    
    # ─── Cycle Phases ────────────────────────────────────────────
    
    def _observe(self, repos: Optional[List[str]] = None) -> List[CommitPoint]:
        """Phase 1: Mine recent commits from fleet repos."""
        target_repos = repos or [
            "plato-training", "plato-types", "tensor-spline", "plato-data",
            "constraint-theory-core", "constraint-theory-py",
            "forgemaster", "cocapn-ai-web",
        ]
        
        new_commits = []
        for repo in target_repos:
            try:
                commits = self.miner.mine_repo(repo, max_commits=100)
                new_commits.extend(commits)
            except Exception as e:
                # Skip repos that fail (private, missing, etc.)
                pass
        
        return new_commits
    
    def _predict(self, velocity: Dict[str, float]) -> List[PredictionEntry]:
        """Phase 2: Generate predictions for next cycle."""
        predictions = []
        now = time.time()
        
        for repo, vel in velocity.items():
            if vel <= 0:
                continue
            
            # Predict commit probability for next hour
            prob = predict_commit_probability(repo, velocity, hours_ahead=1.0)
            
            if prob > 0.3:
                pred = PredictionEntry(
                    prediction_id=hashlib.md5(
                        f"{repo}:{now}:{prob}".encode()
                    ).hexdigest()[:12],
                    repo=repo,
                    predicted_at=now,
                    predicted_activity="commit",
                    confidence=prob,
                    time_window_hours=1.0,
                    context=f"λ={vel:.2f}/hr, P(commit in 1hr)={prob:.2f}",
                )
                predictions.append(pred)
        
        # Detect and predict spike continuations
        spikes = detect_activity_spike(velocity, self.baseline_velocity)
        for repo in spikes:
            pred = PredictionEntry(
                prediction_id=hashlib.md5(
                    f"spike:{repo}:{now}".encode()
                ).hexdigest()[:12],
                repo=repo,
                predicted_at=now,
                predicted_activity="spike",
                confidence=0.8,
                time_window_hours=2.0,
                context=f"Spike detected: velocity {velocity[repo]:.2f} vs baseline {self.baseline_velocity.get(repo, 0):.2f}",
            )
            predictions.append(pred)
        
        return predictions
    
    def _compare(
        self,
        predictions: List[PredictionEntry],
        observed: List[CommitPoint],
        window_hours: float = 1.0,
    ) -> Tuple[int, int, List[PredictionEntry]]:
        """
        Phase 3: Compare predictions vs observations.
        
        Returns (correct, missed, resolved_predictions).
        """
        now = time.time()
        window = window_hours * 3600
        
        # Index observed commits by repo
        obs_by_repo = defaultdict(list)
        for c in observed:
            if now - c.timestamp < window:
                obs_by_repo[c.repo].append(c)
        
        correct = 0
        missed = 0
        
        for pred in predictions:
            if pred.observed:
                continue
            
            # Check if prediction window has expired
            if now - pred.predicted_at > pred.time_window_hours * 3600:
                # Window expired — check if correct
                repo_commits = obs_by_repo.get(pred.repo, [])
                recent = [c for c in repo_commits if c.timestamp > pred.predicted_at]
                
                if pred.predicted_activity == "commit" and len(recent) > 0:
                    pred.observed = True
                    pred.correct = True
                    pred.observed_at = now
                    correct += 1
                elif pred.predicted_activity == "spike" and len(recent) >= 3:
                    pred.observed = True
                    pred.correct = True
                    pred.observed_at = now
                    correct += 1
                else:
                    pred.observed = True
                    pred.correct = False
                    pred.observed_at = now
                    missed += 1
        
        return correct, missed, predictions
    
    def _learn(
        self,
        correct: int,
        missed: int,
        velocity: Dict[str, float],
    ) -> Tuple[float, List[Dict]]:
        """
        Phase 4: Compute gap score and generate focus items.
        
        Gap = 1 - accuracy (fraction of predictions that were wrong).
        Focus items = repos/activities where predictions missed.
        """
        total = correct + missed
        if total == 0:
            gap_score = 0.0
        else:
            gap_score = missed / total
        
        self.cumulative_gap = (
            0.7 * self.cumulative_gap + 0.3 * gap_score
        )  # EMA
        
        # Update baseline velocity
        for repo, vel in velocity.items():
            old = self.baseline_velocity.get(repo, 0.0)
            self.baseline_velocity[repo] = 0.8 * old + 0.2 * vel  # EMA
        
        # Generate focus items from missed predictions
        focus_items = []
        for pred in self.predictions:
            if pred.observed and not pred.correct:
                focus_items.append({
                    "type": "missed_prediction",
                    "repo": pred.repo,
                    "activity": pred.predicted_activity,
                    "confidence": pred.confidence,
                    "context": pred.context,
                    "gap_contribution": 1.0 / max(total, 1),
                })
        
        # Also flag repos with unexpected velocity changes
        for repo, vel in velocity.items():
            base = self.baseline_velocity.get(repo, 0.0)
            if base > 0 and abs(vel - base) / base > 1.5:
                focus_items.append({
                    "type": "velocity_shift",
                    "repo": repo,
                    "current": vel,
                    "baseline": base,
                    "shift_ratio": vel / base if base > 0 else float('inf'),
                })
        
        return gap_score, focus_items
    
    # ─── Main Cycle ──────────────────────────────────────────────
    
    def run_cycle(
        self,
        repos: Optional[List[str]] = None,
    ) -> CycleResult:
        """
        Execute one complete predict → observe → compare → learn cycle.
        """
        self.cycle_count += 1
        cycle_id = hashlib.md5(
            f"cycle:{self.cycle_count}:{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Phase 1: Observe
        new_commits = self._observe(repos)
        self.all_commits.extend(new_commits)
        
        # Deduplicate by sha+repo
        seen = set()
        unique = []
        for c in self.all_commits:
            key = f"{c.sha}:{c.repo}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        self.all_commits = unique
        
        # Compute current velocity
        velocity = compute_repo_velocity(self.all_commits, hours=24.0)
        
        # Phase 2: Predict
        new_predictions = self._predict(velocity)
        self.predictions.extend(new_predictions)
        
        # Phase 3: Compare (check previous predictions)
        correct, missed, self.predictions = self._compare(
            self.predictions, self.all_commits, window_hours=1.0,
        )
        
        # Phase 4: Learn
        gap_score, focus_items = self._learn(correct, missed, velocity)
        
        # Find synergies in new commits
        synergies = []
        for c in new_commits:
            for ref in c.cross_refs:
                synergies.append({
                    "source": c.repo,
                    "target": ref,
                    "sha": c.sha,
                })
        
        # Compute Transfer Entropy from commit author transitions
        te, se = self._compute_coordination_topology(self.all_commits)
        
        # Build result
        result = CycleResult(
            cycle_id=cycle_id,
            timestamp=time.time(),
            repos_observed=len(set(c.repo for c in new_commits)),
            commits_observed=len(new_commits),
            predictions_made=len(new_predictions),
            predictions_correct=correct,
            predictions_missed=missed,
            gap_score=gap_score,
            focus_items=focus_items[:10],  # top 10
            top_synergies=synergies[:10],
            velocity_by_repo=velocity,
            transfer_entropy=te,
            source_entropy=se,
        )
        
        # Persist
        self._save_history()
        
        return result
    
    def run_forever(
        self,
        interval_minutes: float = 30.0,
        max_cycles: int = 0,
        repos: Optional[List[str]] = None,
    ):
        """
        Run continuous loop.
        
        Args:
            interval_minutes: minutes between cycles
            max_cycles: 0 = infinite, else stop after N cycles
            repos: repos to mine (None = default set)
        """
        print(f"🌊 Collective Inference Loop starting")
        print(f"   Interval: {interval_minutes}min")
        print(f"   Max cycles: {'∞' if max_cycles == 0 else max_cycles}")
        print()
        
        cycles = 0
        while True:
            if max_cycles > 0 and cycles >= max_cycles:
                break
            
            print(f"── Cycle {self.cycle_count + 1} ──")
            result = self.run_cycle(repos=repos)
            
            ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc)
            print(f"  Observed: {result.commits_observed} commits from {result.repos_observed} repos")
            print(f"  Predictions: {result.predictions_made} made, {result.predictions_correct} correct, {result.predictions_missed} missed")
            print(f"  Gap score: {result.gap_score:.3f} (cumulative: {self.cumulative_gap:.3f})")
            print(f"  Focus items: {len(result.focus_items)}")
            
            if result.top_synergies:
                print(f"  Synergies: {len(result.top_synergies)}")
                for s in result.top_synergies[:3]:
                    print(f"    {s['source']} → {s['target']}")
            
            print()
            cycles += 1
            
            if max_cycles == 0 or cycles < max_cycles:
                time.sleep(interval_minutes * 60)
    
    # ─── Coordination Topology (CCC-style TE) ────────────────
    
    def _compute_coordination_topology(
        self, commits: List[CommitPoint]
    ) -> Tuple[float, float]:
        """
        Compute Transfer Entropy and Source Entropy from commit stream.
        
        Adapted from CCC's coordination-topology: TE measures how much
        knowing agent A's last commit predicts agent B's next commit.
        Source entropy measures fleet diversity.
        
        Returns (transfer_entropy_bits, source_entropy_bits).
        """
        from collections import defaultdict
        from math import log2
        
        if len(commits) < 10:
            return 0.0, 0.0
        
        # Sort commits by time and extract author sequence
        sorted_commits = sorted(commits, key=lambda c: c.timestamp)
        author_seq = [c.author for c in sorted_commits]
        
        if len(set(author_seq)) < 2:
            # Only one author → no TE possible
            counts = defaultdict(int)
            for a in author_seq:
                counts[a] += 1
            total = len(author_seq)
            h = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    h -= p * log2(p)
            return 0.0, h
        
        # Compute author transitions (CCC-style fleet_transitions)
        transitions = defaultdict(int)
        prev_counts = defaultdict(int)
        curr_counts = defaultdict(int)
        
        for i in range(1, len(author_seq)):
            prev = author_seq[i - 1]
            curr = author_seq[i]
            transitions[(prev, curr)] += 1
            prev_counts[prev] += 1
            curr_counts[curr] += 1
        
        total = sum(transitions.values())
        if total == 0:
            return 0.0, 0.0
        
        # Source entropy H(X)
        h_source = 0.0
        for c in curr_counts.values():
            p = c / total
            if p > 0:
                h_source -= p * log2(p)
        
        # Conditional entropy H(X|Y) from transitions
        pair_counts = defaultdict(lambda: defaultdict(int))
        for (prev, curr), count in transitions.items():
            pair_counts[prev][curr] += count
        
        h_cond = 0.0
        for prev, total_prev in prev_counts.items():
            p_prev = total_prev / total
            h_given = 0.0
            for c in pair_counts[prev].values():
                p = c / total_prev
                if p > 0:
                    h_given -= p * log2(p)
            h_cond += p_prev * h_given
        
        # Transfer Entropy ≈ H(X) - H(X|Y)
        # Positive TE = knowing previous author reduces uncertainty about next
        te = max(0.0, h_source - h_cond)
        
        return round(te, 4), round(h_source, 4)

    # ─── Reporting ───────────────────────────────────────────────
    
    def status_report(self) -> Dict:
        """Generate a status summary for PLATO tile submission."""
        return {
            "agent": "forgemaster",
            "cycle_count": self.cycle_count,
            "cumulative_gap": round(self.cumulative_gap, 4),
            "total_commits_mined": len(self.all_commits),
            "repos_tracked": list(self.baseline_velocity.keys()),
            "velocity": {
                k: round(v, 3) for k, v in self.baseline_velocity.items()
            },
            "pending_predictions": sum(
                1 for p in self.predictions if not p.observed
            ),
            "resolved_predictions": sum(
                1 for p in self.predictions if p.observed
            ),
        }


# ─── CLI Entry Point ────────────────────────────────────────────────

def main():
    """Run collective inference loop from command line."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Collective Inference Loop")
    parser.add_argument("--token", default=None, help="GitHub PAT")
    parser.add_argument("--org", default="SuperInstance", help="GitHub org")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles (0=forever)")
    parser.add_argument("--interval", type=float, default=30.0, help="Minutes between cycles")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()
    
    token = args.token or os.environ.get("GITHUB_TOKEN")
    
    loop = CollectiveLoop(
        github_token=token,
        org=args.org,
    )
    
    if args.status:
        report = loop.status_report()
        print(json.dumps(report, indent=2))
        return
    
    if args.cycles == 1:
        result = loop.run_cycle()
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"Cycle {result.cycle_id} complete")
            print(f"  Commits: {result.commits_observed}")
            print(f"  Gap: {result.gap_score:.3f}")
            print(f"  Focus: {len(result.focus_items)} items")
    else:
        loop.run_forever(
            interval_minutes=args.interval,
            max_cycles=args.cycles,
        )


if __name__ == "__main__":
    main()
