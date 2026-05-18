---
name: plato-training
family: ml-training
version: 0.5.0
summary: "Micro model training for PLATO rooms — 8 tasks × 8 hardware targets, collective inference loop, commit pattern prediction. 359 tests."
provides:
  - tool_name: commit_predictor
    description: Predict fleet commit activity (commit prob, file activity, crossref prob) from recent git history
    input: "Dict of repo velocities + time features"
    output: "Dict of prediction probabilities"
  - tool_name: collective_loop
    description: Run predict → observe → gap → learn cycles against live fleet data
    input: "GitHub token, org name"
    output: "CycleResult with gap score, focus items, synergies"
  - tool_name: deploy_micro
    description: Train + optimize + export + benchmark micro models for PLATO tasks
    input: "task name, hardware target"
    output: "Trained model + benchmarks"
  - tool_name: train_commit_predictor
    description: End-to-end pipeline — mine commits → build dataset → train predictor
    input: "List of commits, repo names"
    output: "Trained CommitPredictor + metrics dict"
depends_on:
  - service: plato
    port: 8847
    required: false
    reason: Tile storage for collective loop results (works offline)
  - service: github
    required: false
    reason: Fleet miner needs GitHub access for commit data
ticks:
  heartbeat: 1800
  training: on_demand
  collective_cycle: 1800
io:
  sensors:
    - name: git-history
      type: github-api
      description: Mine commit history from SuperInstance repos
  pushes:
    - name: cycle-results
      dest: plato
      type: tile
      domain: collective-inference
      interval: 1800
      question: "What did the collective inference cycle find?"
    - name: prediction-tiles
      dest: plato
      type: tile
      domain: prediction
      interval: 3600
      question: "What is the predicted fleet activity for the next hour?"
  pulls:
    - name: fleet-velocity
      source: plato
      room: coordination-hierarchy
      interval: 900
    - name: tool-registry
      source: plato
      room: fleet-tool-registry
      interval: 3600
---
