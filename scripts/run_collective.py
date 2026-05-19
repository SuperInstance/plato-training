#!/usr/bin/env python3
"""
Run the PLATO Collective Inference Loop on real fleet data.

Usage:
    python scripts/run_collective.py
    python scripts/run_collective.py --cycles 5
    python scripts/run_collective.py --json
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plato_training.collective_loop import CollectiveLoop
from plato_training.commit_predictor import train_commit_predictor


def get_github_token() -> str:
    """Read GitHub PAT from credentials file."""
    pat_file = Path.home() / ".openclaw" / "workspace" / ".credentials" / "github-pat.txt"
    if pat_file.exists():
        return pat_file.read_text().strip()
    return os.environ.get("GITHUB_TOKEN", "")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Collective Inference Loop")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--train", action="store_true", help="Train CommitPredictor first")
    parser.add_argument("--token", default=None, help="GitHub PAT override")
    args = parser.parse_args()

    token = args.token or get_github_token()
    workspace = str(Path.home() / ".openclaw" / "workspace")

    # The 4 PLATO stack repos
    plato_repos = [
        "plato-training",
        "plato-types",
        "tensor-spline",
        "plato-data",
    ]

    loop = CollectiveLoop(
        github_token=token,
        clone_dir="/tmp/fleet-mine",
        history_file=str(Path(workspace) / "collective-history.json"),
        org="SuperInstance",
    )

    # Optionally train commit predictor
    if args.train:
        print("=== Training CommitPredictor on fleet data ===")
        from plato_training.fleet_miner import FleetMiner
        miner = FleetMiner(org="SuperInstance", token=token, clone_dir=workspace)
        all_commits = []
        for repo in plato_repos:
            commits = miner.mine_repo(repo, max_commits=200)
            all_commits.extend(commits)
            print(f"  {repo}: {len(commits)} commits")

        if all_commits:
            model, metrics = train_commit_predictor(all_commits, repos=plato_repos, epochs=50)
            print(f"  Samples: {metrics['samples']}")
            print(f"  Commit accuracy: {metrics['commit_accuracy']}")
            print(f"  Final loss: {metrics['final_loss']}")
        else:
            print("  No commits found for training")
        print()

    # Run collective loop cycles
    print(f"=== Running {args.cycles} Collective Inference Cycles ===")
    print(f"Repos: {', '.join(plato_repos)}")
    print()

    results = []
    for i in range(args.cycles):
        print(f"── Cycle {i+1}/{args.cycles} ──")
        result = loop.run_cycle(repos=plato_repos)
        results.append(result.to_dict())

        print(f"  Observed: {result.commits_observed} commits from {result.repos_observed} repos")
        print(f"  Predictions: {result.predictions_made} made, {result.predictions_correct} correct, {result.predictions_missed} missed")
        print(f"  Gap score: {result.gap_score:.3f} (cumulative: {loop.cumulative_gap:.3f})")
        print(f"  Focus items: {len(result.focus_items)}")
        print(f"  Transfer entropy: {result.transfer_entropy:.4f} bits")
        print(f"  Source entropy: {result.source_entropy:.4f} bits")

        if result.velocity_by_repo:
            print("  Velocity:")
            for repo, vel in sorted(result.velocity_by_repo.items()):
                print(f"    {repo}: {vel:.2f} commits/hr")
        print()

    # Summary
    total_commits = sum(r["commits_observed"] for r in results)
    avg_gap = sum(r["gap_score"] for r in results) / len(results) if results else 0
    final_cumulative = loop.cumulative_gap

    summary = {
        "cycles": args.cycles,
        "repos": plato_repos,
        "total_commits_observed": total_commits,
        "avg_gap_score": round(avg_gap, 4),
        "final_cumulative_gap": round(final_cumulative, 4),
        "total_predictions": sum(r["predictions_made"] for r in results),
        "total_correct": sum(r["predictions_correct"] for r in results),
        "total_missed": sum(r["predictions_missed"] for r in results),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("=== SUMMARY ===")
        print(f"  Total commits observed: {total_commits}")
        print(f"  Average gap score: {avg_gap:.4f}")
        print(f"  Final cumulative gap: {final_cumulative:.4f}")
        print(f"  Predictions: {summary['total_predictions']} made, {summary['total_correct']} correct, {summary['total_missed']} missed")

    return summary


if __name__ == "__main__":
    main()
