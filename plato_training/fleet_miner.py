"""
Fleet Git Miner — extract real data from SuperInstance git history.

The org IS the dataset. Every commit is a data point in multidimensional space:
  - time (timestamp)
  - repo (which vessel)
  - author (which agent)
  - files changed (what subsystem)
  - cross-repo references (synergy / pollination)
  - commit size (velocity)
  - message content (intent signal)

This feeds into collective inference rooms:
  - Predict: "repo X will have a commit in the next hour"
  - Observe: actual commit stream
  - Gap: when predictions miss → that's signal
"""

import subprocess
import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class CommitPoint:
    """A single commit as a multidimensional data point."""
    sha: str
    repo: str
    author: str
    timestamp: float
    message: str
    files_changed: int
    insertions: int
    deletions: int
    is_merge: bool
    cross_refs: List[str] = field(default_factory=list)  # repos mentioned in message
    languages: List[str] = field(default_factory=list)    # file extensions changed
    
    @property
    def size(self) -> int:
        return self.insertions + self.deletions
    
    @property
    def net_lines(self) -> int:
        return self.insertions - self.deletions
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SynergyEvent:
    """Two repos cross-pollinating — a reference from one to another."""
    source_repo: str
    target_repo: str
    timestamp: float
    sha: str
    ref_type: str          # "mention", "dependency", "fork", "fix-from"
    message_snippet: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RepoSignal:
    """Aggregated signal for a repo over a time window."""
    repo: str
    window_start: float
    window_end: float
    commits: int
    authors: int
    total_insertions: int
    total_deletions: int
    files_touched: int
    cross_refs_out: int    # references to other repos
    cross_refs_in: int     # referenced by other repos
    languages: Dict[str, int]  # language → count
    merge_ratio: float     # merges / total commits
    velocity: float        # commits per hour
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ─── Known SuperInstance Repos ──────────────────────────────────────

FLEET_REPOS = [
    "plato-training", "plato-types", "tensor-spline", "plato-data",
    "plato-sdk", "fleet-memory", "folding-order", "holonomy-consensus",
    "constraint-flow-protocol", "constraint-inference", "intent-inference",
    "penrose-memory", "flux-lucid", "dodecet-encoder",
    "constraint-theory-py", "constraint-theory-core", "ct-demo",
    "constraint-theory-ecosystem", "neural-plato",
    "cocapn-ai-web", "openarm",
    "forgemaster", "oracle1-workspace", "oracle1-box",
    "plato-vessel-core", "casting-call",
]

# Patterns that indicate cross-repo references
CROSS_REF_PATTERNS = [
    re.compile(r'SuperInstance/(\w[\w-]*)', re.IGNORECASE),
    re.compile(r'cocapn/(\w[\w-]*)', re.IGNORECASE),
    re.compile(r'(?:see|fix|from|in|via|ref)\s+(\w[\w-]*(?:-[\w-]+)*)', re.IGNORECASE),
]


class FleetMiner:
    """
    Mine real data from the fleet's git history.
    
    Usage:
        miner = FleetMiner(org="SuperInstance", token="gho_...")
        commits = miner.mine_repo("plato-training", max_commits=500)
        synergies = miner.find_synergies(commits)
        signals = miner.aggregate_signals(commits, window_hours=24)
    """
    
    def __init__(
        self,
        org: str = "SuperInstance",
        token: Optional[str] = None,
        clone_dir: str = "/tmp/fleet-mine",
    ):
        self.org = org
        self.token = token
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self._cloned: Dict[str, Path] = {}
    
    def _repo_url(self, repo: str) -> str:
        if self.token:
            return f"https://x-access-token:{self.token}@github.com/{self.org}/{repo}.git"
        return f"https://github.com/{self.org}/{repo}.git"
    
    def _ensure_cloned(self, repo: str) -> Path:
        """Clone repo if not already cloned, or pull latest."""
        repo_path = self.clone_dir / repo
        
        if repo_path.exists():
            self._cloned[repo] = repo_path
            return repo_path
        
        url = self._repo_url(repo)
        subprocess.run(
            ["git", "clone", "--depth", "500", url, str(repo_path)],
            capture_output=True, timeout=120,
        )
        self._cloned[repo] = repo_path
        return repo_path
    
    def mine_repo(self, repo: str, max_commits: int = 500) -> List[CommitPoint]:
        """
        Extract commit history from a repo.
        
        Returns list of CommitPoint data points.
        """
        repo_path = self._ensure_cloned(repo)
        
        # Get log as JSON-ish format
        result = subprocess.run(
            [
                "git", "log",
                f"-{max_commits}",
                "--format=%H|%an|%at|%s",
                "--numstat",
                "--no-merges",  # We'll detect merges separately
            ],
            capture_output=True, text=True, cwd=str(repo_path), timeout=60,
        )
        
        if result.returncode != 0:
            return []
        
        commits = []
        lines = result.stdout.strip().split("\n")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line or "|" not in line:
                i += 1
                continue
            
            parts = line.split("|", 3)
            if len(parts) < 4:
                i += 1
                continue
            
            sha, author, timestamp_str, message = parts
            
            # Collect numstat lines until next commit or empty line
            insertions = 0
            deletions = 0
            files_changed = 0
            extensions = set()
            
            i += 1
            while i < len(lines) and lines[i] and "\t" in lines[i]:
                stat = lines[i].split("\t")
                if len(stat) >= 3:
                    try:
                        ins = int(stat[0]) if stat[0] != "-" else 0
                        dels = int(stat[1]) if stat[1] != "-" else 0
                        insertions += ins
                        deletions += dels
                        files_changed += 1
                        
                        # Extract language from extension
                        filepath = stat[2]
                        ext = Path(filepath).suffix
                        if ext:
                            extensions.add(ext)
                    except ValueError:
                        pass
                i += 1
            
            # Find cross-references in message
            cross_refs = self._find_cross_refs(message, source_repo=repo)
            
            commit = CommitPoint(
                sha=sha[:12],
                repo=repo,
                author=author,
                timestamp=float(timestamp_str),
                message=message[:200],
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                is_merge=False,
                cross_refs=cross_refs,
                languages=sorted(extensions),
            )
            commits.append(commit)
        
        return commits
    
    def _find_cross_refs(self, message: str, source_repo: str = "") -> List[str]:
        """Find references to other repos in commit message.
        
        Uses loose matching: any fleet repo name appearing in the message,
        regardless of prefix (SuperInstance/, cocapn/) or no prefix.
        """
        refs = set()
        msg_lower = message.lower()
        repo_names_lower = {r.lower(): r for r in FLEET_REPOS}
        
        for rn_lower, rn in repo_names_lower.items():
            if rn_lower in msg_lower and rn != source_repo:
                refs.add(rn)
        
        return sorted(refs)
    
    def find_synergies(self, commits: List[CommitPoint]) -> List[SynergyEvent]:
        """Extract cross-pollination events from commit data."""
        synergies = []
        
        for commit in commits:
            for ref in commit.cross_refs:
                if ref != commit.repo:  # Don't self-reference
                    synergies.append(SynergyEvent(
                        source_repo=commit.repo,
                        target_repo=ref,
                        timestamp=commit.timestamp,
                        sha=commit.sha,
                        ref_type="mention",
                        message_snippet=commit.message[:100],
                    ))
        
        return synergies
    
    def aggregate_signals(
        self,
        commits: List[CommitPoint],
        window_hours: int = 24,
    ) -> List[RepoSignal]:
        """Aggregate commit data into time-windowed signals per repo."""
        if not commits:
            return []
        
        # Group by repo
        by_repo: Dict[str, List[CommitPoint]] = {}
        for c in commits:
            by_repo.setdefault(c.repo, []).append(c)
        
        signals = []
        window_s = window_hours * 3600
        
        for repo, repo_commits in by_repo.items():
            sorted_commits = sorted(repo_commits, key=lambda c: c.timestamp)
            
            # Use full range as one window (simplify for now)
            start = sorted_commits[0].timestamp
            end = sorted_commits[-1].timestamp
            duration_h = max((end - start) / 3600, 0.1)
            
            authors = set(c.author for c in repo_commits)
            languages: Dict[str, int] = {}
            cross_out = sum(len(c.cross_refs) for c in repo_commits)
            
            for c in repo_commits:
                for lang in c.languages:
                    languages[lang] = languages.get(lang, 0) + 1
            
            signals.append(RepoSignal(
                repo=repo,
                window_start=start,
                window_end=end,
                commits=len(repo_commits),
                authors=len(authors),
                total_insertions=sum(c.insertions for c in repo_commits),
                total_deletions=sum(c.deletions for c in repo_commits),
                files_touched=sum(c.files_changed for c in repo_commits),
                cross_refs_out=cross_out,
                cross_refs_in=0,  # Computed later from synergy analysis
                languages=languages,
                merge_ratio=sum(1 for c in repo_commits if c.is_merge) / max(len(repo_commits), 1),
                velocity=len(repo_commits) / duration_h,
            ))
        
        return signals
    
    def mine_fleet(self, repos: Optional[List[str]] = None, max_per_repo: int = 200) -> Dict:
        """
        Mine the entire fleet. The BIG picture.
        
        Returns dict with commits, synergies, signals, summary.
        """
        repos = repos or FLEET_REPOS
        all_commits: List[CommitPoint] = []
        errors: Dict[str, str] = {}
        
        for repo in repos:
            try:
                commits = self.mine_repo(repo, max_commits=max_per_repo)
                all_commits.extend(commits)
                print(f"  {repo}: {len(commits)} commits")
            except Exception as e:
                errors[repo] = str(e)[:100]
                print(f"  {repo}: FAIL ({str(e)[:60]})")
        
        synergies = self.find_synergies(all_commits)
        signals = self.aggregate_signals(all_commits)
        
        # Compute cross_refs_in from synergies
        repo_in_refs: Dict[str, int] = {}
        for s in synergies:
            repo_in_refs[s.target_repo] = repo_in_refs.get(s.target_repo, 0) + 1
        for sig in signals:
            sig.cross_refs_in = repo_in_refs.get(sig.repo, 0)
        
        return {
            "commits": [c.to_dict() for c in all_commits],
            "synergies": [s.to_dict() for s in synergies],
            "signals": [s.to_dict() for s in signals],
            "summary": {
                "repos_mined": len(repos),
                "repos_ok": len(repos) - len(errors),
                "repos_failed": len(errors),
                "total_commits": len(all_commits),
                "total_synergies": len(synergies),
                "errors": errors,
            },
        }
    
    def fleet_report(self, data: Dict) -> str:
        """Human-readable fleet intelligence report."""
        s = data["summary"]
        lines = [
            f"=== FLEET INTELLIGENCE REPORT ===",
            f"Repos mined: {s['repos_ok']}/{s['repos_mined']}",
            f"Total commits: {s['total_commits']}",
            f"Cross-pollination events: {s['total_synergies']}",
            "",
            "=== TOP REPOS BY VELOCITY ===",
        ]
        
        signals = sorted(data["signals"], key=lambda x: x["velocity"], reverse=True)
        for sig in signals[:10]:
            lines.append(
                f"  {sig['repo']:30s} {sig['velocity']:6.1f} commits/h "
                f"({sig['commits']} commits, {sig['authors']} authors) "
                f"refs: {sig['cross_refs_out']}→ {sig['cross_refs_in']}←"
            )
        
        lines.append("")
        lines.append("=== SYNERGY GRAPH ===")
        # Group synergies by (source, target)
        pair_counts: Dict[Tuple[str, str], int] = {}
        for syn in data["synergies"]:
            key = (syn["source_repo"], syn["target_repo"])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        
        for (src, tgt), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {src} → {tgt} ({count} refs)")
        
        if not pair_counts:
            lines.append("  (no cross-repo references found)")
        
        return "\n".join(lines)
