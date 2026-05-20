# Contributing to plato-training

Thanks for your interest! Here's how to contribute effectively.

## Development Setup

```bash
git clone https://github.com/SuperInstance/plato-training.git
cd plato-training
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full suite
python3 -m pytest tests/ -q --tb=short

# Specific module
python3 -m pytest tests/test_spline.py -v

# With coverage
python3 -m pytest tests/ --cov=plato_training --cov-report=term-missing
```

All tests must pass before submitting a PR. If you add new functionality, add tests for it.

## Code Style

- **Python 3.10+** — use modern type hints (`list[X]` not `List[X]`, `X | Y` not `Union[X, Y]`)
- **Line length:** 100 characters max
- **Formatting:** `ruff format` (Black-compatible)
- **Linting:** `ruff check`
- **Docstrings:** NumPy style for public APIs, one-liners for internal helpers
- **Imports:** stdlib → third-party → local, sorted

## Submitting a PR

1. **Fork** the repo and create a feature branch from `master`
2. **Write tests** for any new code — aim for meaningful coverage, not 100%
3. **Run the full test suite** locally before pushing
4. **Keep PRs focused** — one concern per PR is ideal
5. **Write a clear description** explaining what and why
6. **Be responsive** to review feedback

## What We're Looking For

- Bug fixes with regression tests
- New room implementations (see existing rooms for patterns)
- Performance improvements with benchmarks
- Documentation improvements
- Real-world deployment experience and edge cases

## What's Out of Scope

- Major architectural changes without prior discussion (open an issue first)
- Adding heavy dependencies without justification
- Changes to the tile protocol without coordination with sibling packages

## Reporting Issues

- **Bugs:** Include Python version, OS, minimal reproduction steps
- **Features:** Explain the use case, not just the solution
- **Questions:** No bad questions — open an issue and ask

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
