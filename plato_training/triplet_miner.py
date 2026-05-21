"""
Mine (anchor, positive, negative) triplets from fleet git history.

Strategy:
- Anchor/Positive: commits from the same repo (similar domain)
- Negative: commits from a different repo (different domain)
- Also uses commit messages + file paths as text signals
- Supports topic clustering for better positive pairs
"""


__all__ = ['build_benchmark_queries', 'get_commit_details', 'get_commit_messages', 'mine_triplets']

import os
import subprocess
import hashlib
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from pathlib import Path


def get_commit_messages(repo_path: str, max_commits: int = 500) -> List[str]:
    """Extract commit messages from a repo."""
    result = subprocess.run(
        ['git', '-C', repo_path, 'log', f'-{max_commits}', '--format=%s', 'HEAD'],
        capture_output=True, text=True, timeout=30
    )
    return [m.strip() for m in result.stdout.strip().split('\n') if m.strip()]


def get_commit_details(repo_path: str, max_commits: int = 500) -> List[Dict]:
    """Get detailed commit info: message, sha, files changed."""
    result = subprocess.run(
        ['git', '-C', repo_path, 'log', f'-{max_commits}', '--format=%s|||%H', 'HEAD'],
        capture_output=True, text=True, timeout=30
    )
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|||' not in line:
            continue
        msg, sha = line.split('|||', 1)
        # Get files changed
        diff = subprocess.run(
            ['git', '-C', repo_path, 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            capture_output=True, text=True, timeout=5
        )
        files = diff.stdout.strip()
        # Extract extensions
        extensions = set()
        for f in files.split('\n'):
            if '.' in f:
                ext = f.rsplit('.', 1)[-1]
                extensions.add(ext)
        commits.append({
            'message': msg.strip(),
            'sha': sha[:12],
            'files': files,
            'extensions': sorted(extensions),
            'repo_path': repo_path,
        })
    return commits


def _topic_key(msg: str) -> str:
    """Extract a rough topic key from a commit message (first 3 words lowercased)."""
    words = msg.lower().split()[:3]
    return ' '.join(words)


def mine_triplets(
    workspace: str,
    max_repos: int = 30,
    max_commits_per_repo: int = 300,
    max_triplets: int = 5000,
    use_topics: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Mine triplets from repos in workspace.
    
    Anchor/Positive: same repo, ideally similar commit topics
    Negative: different repo
    
    Args:
        workspace: Directory containing git repos
        max_repos: Maximum repos to mine
        max_commits_per_repo: Max commits per repo
        max_triplets: Cap on total triplets (sample if exceeded)
        use_topics: If True, try to pair anchor/positive with similar topics
    
    Returns:
        List of (anchor, positive, negative) text triplets
    """
    workspace = str(workspace)
    
    # Find all git repos sorted by commit count
    repos = []
    for d in sorted(os.listdir(workspace)):
        repo_path = os.path.join(workspace, d)
        git_dir = os.path.join(repo_path, '.git')
        if not os.path.isdir(git_dir):
            continue
        try:
            count = subprocess.check_output(
                ['git', '-C', repo_path, 'rev-list', '--count', 'HEAD'],
                stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            repos.append((d, int(count), repo_path))
        except Exception:
            pass
    
    repos.sort(key=lambda x: -x[1])
    repos = repos[:max_repos]
    
    print(f"Found {len(repos)} repos with git history:")
    for name, count, _ in repos[:10]:
        print(f"  {name}: {count} commits")
    if len(repos) > 10:
        print(f"  ... and {len(repos) - 10} more")
    
    # Mine commits per repo
    repo_commits: Dict[str, List[Dict]] = {}
    for name, count, path in repos:
        try:
            commits = get_commit_details(path, max_commits=min(count, max_commits_per_repo))
            if len(commits) >= 3:
                repo_commits[name] = commits
        except Exception as e:
            print(f"  {name}: mining failed ({e})")
    
    print(f"\nMined commits from {len(repo_commits)} repos")
    
    # Build text corpus from commits
    # Each commit produces: message text, and optionally file-context text
    repo_texts: Dict[str, List[str]] = defaultdict(list)
    for name, commits in repo_commits.items():
        for c in commits:
            msg = c['message']
            if len(msg) > 10:  # Skip very short messages
                repo_texts[name].append(msg)
            # Add file context sentence
            if c['extensions']:
                ext_str = ' '.join(c['extensions'])
                repo_texts[name].append(f"{name} {ext_str} code changes")
    
    total_texts = sum(len(v) for v in repo_texts.values())
    print(f"Total text entries: {total_texts}")
    
    # Build triplets
    triplets: List[Tuple[str, str, str]] = []
    repo_names = list(repo_texts.keys())
    
    if len(repo_names) < 2:
        print("WARNING: Need at least 2 repos with commits. Adding synthetic data.")
        # Fallback synthetic triplets
        synth = [
            ("spline linear compression", "tensor spline weight parameterization", "recipe for soup"),
            ("constraint theory solver", "constraint propagation algorithm", "weather forecast today"),
            ("plato training room tile", "training tile lifecycle states", "how to bake bread"),
            ("fleet coordination protocol", "fleet agent coordination", "car maintenance tips"),
            ("eisenstein lattice weights", "hexagonal lattice control points", "dance class schedule"),
        ]
        triplets.extend(synth)
        return triplets[:max_triplets]
    
    for i, repo in enumerate(repo_names):
        texts = repo_texts[repo]
        if len(texts) < 2:
            continue
        
        # Group by topic for better positive pairs
        if use_topics:
            topic_groups: Dict[str, List[str]] = defaultdict(list)
            for t in texts:
                key = _topic_key(t)
                topic_groups[key].append(t)
        
        for j in range(len(texts)):
            anchor = texts[j]
            
            # Find positive: same repo, prefer same topic
            if use_topics:
                topic = _topic_key(anchor)
                candidates = topic_groups.get(topic, [])
                candidates = [c for c in candidates if c != anchor]
                if candidates:
                    positive = candidates[hash(anchor) % len(candidates)]
                else:
                    # Any other text from same repo
                    others = [t for t in texts if t != anchor]
                    if not others:
                        continue
                    positive = others[j % len(others)]
            else:
                others = [t for t in texts if t != anchor]
                if not others:
                    continue
                positive = others[j % len(others)]
            
            # Negative: from a different repo
            neg_repo = repo_names[(i + 1) % len(repo_names)]
            neg_texts = repo_texts[neg_repo]
            if not neg_texts:
                continue
            negative = neg_texts[j % len(neg_texts)]
            
            # Truncate long texts
            triplets.append((anchor[:120], positive[:120], negative[:120]))
    
    print(f"Raw triplets generated: {len(triplets)}")
    
    # Sample if too many
    if len(triplets) > max_triplets:
        import random
        random.shuffle(triplets)
        triplets = triplets[:max_triplets]
        print(f"Sampled down to {max_triplets} triplets")
    
    return triplets


def build_benchmark_queries() -> List[Tuple[str, str]]:
    """Build benchmark queries with known correct repos."""
    return [
        # (query_text, correct_repo_name)
        ("forgemaster constraint specialist", "forgemaster"),
        ("forge proof constraint theory", "forgemaster"),
        ("quality gate stream validation", "quality-gate-stream"),
        ("stream quality check pipeline", "quality-gate-stream"),
        ("fleet murmur agent communication", "fleet-murmur"),
        ("murmur message routing protocol", "fleet-murmur"),
        ("fleet health monitor status", "fleet-health-monitor"),
        ("health check monitoring daemon", "fleet-health-monitor"),
        ("automerge pull request merge", "automerge"),
        ("auto merge branch workflow", "automerge"),
        ("open shell terminal interface", "OpenShell"),
        ("shell command execution agent", "OpenShell"),
        ("flux research analysis", "flux-research"),
        ("flux vm virtual machine", "flux-vm"),
        ("eisenstein integer lattice encoder", "eisenstein"),
        ("dodecet twelve tone encoding", "dodecet-encoder"),
        ("constraint theory python proof", "constraint-theory-ecosystem"),
        ("cocapn ai web interface", "cocapn-ai-web"),
        ("plato training room tile", "plato-training"),
        ("tensor spline weight compression", "tensor-spline"),
        ("neural plato learning room", "neural-plato"),
        ("penrose memory tiling", "penrose-memory"),
        ("signal chain processing", "signal-chain"),
        ("holonomy consensus protocol", "holonomy-consensus"),
        ("flux lucid dreaming state", "flux-lucid"),
        ("polyformalism thinking formal", "polyformalism-thinking"),
        ("ai writing generation text", "ai-writings"),
        ("pbft rust consensus algorithm", "pbft-rust"),
        ("acg protocol communication", "acg_protocol"),
        ("tri quarter toolbox math", "tri-quarter-toolbox"),
        ("lucineer lucid engineer agent", "lucineer"),
        ("superinstance github org", "SuperInstance"),
        ("forgemaster git commit history", "forgemaster"),
        ("quality stream gate check", "quality-gate-stream"),
        ("fleet murmur broadcast message", "fleet-murmur"),
        ("monitor health fleet agent", "fleet-health-monitor"),
        ("automerge github workflow", "automerge"),
        ("open shell bash terminal", "OpenShell"),
        ("flux research experiment data", "flux-research"),
        ("constraint ecosystem theory solver", "constraint-theory-ecosystem"),
        ("cocapn web dashboard ai", "cocapn-ai-web"),
        ("training plato micro model", "plato-training"),
        ("spline tensor weight lattice", "tensor-spline"),
        ("penrose memory pattern store", "penrose-memory"),
        ("holonomy consensus distributed", "holonomy-consensus"),
        ("polyformalism a2a python agent", "polyformalism-a2a-python"),
        ("platoclaw plato openclaw bridge", "platoclaw"),
        ("neural plato deep learning", "neural-plato"),
        ("signal chain audio processing", "signal-chain"),
        ("dodecet music encoding twelve", "dodecet-encoder"),
    ]
