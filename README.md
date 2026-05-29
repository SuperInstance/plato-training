# plato-training

**PLATO training rooms** — structured environments for training and evaluating AI agents. Simulated rooms with configurable scenarios, metrics, and evaluation criteria.

## What This Gives You

- **Training rooms** — isolated environments for agent skill development
- **Scenario configuration** — define training scenarios with expected outcomes
- **Performance metrics** — track agent improvement across training iterations
- **Evaluation criteria** — automated scoring against defined rubrics
- **Fleet integration** — rooms plug into the PLATO tile system

## Installation

```bash
pip install plato-training
```

## How It Fits

The training infrastructure in the PLATO stack: `plato-room` (rooms) → `plato-training` (training scenarios) → `plato-sandbox` (safe execution). Feeds trained agents into `plato-fleet` for deployment.

## License

MIT
