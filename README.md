# start-kit-python-repo

A reference template for Python projects, with code quality standards enforced through linting, type checking, and CI/CD.

## Requirements

- [uv](https://docs.astral.sh/uv/) — fast Python package manager

## Setup

```bash
make venv
```

## Development

```bash
make lint        # ruff linting
make type-check  # ty type checking
make tests       # pytest
make coverage    # pytest with coverage report
make pre-commit  # run all hooks on all files
make clean       # remove caches and build artifacts
```

## Code Quality

This project uses:
- **[ruff](https://docs.astral.sh/ruff/)** — linting and formatting
- **[ty](https://github.com/astral-sh/ty)** — type checking
- **[prek](https://github.com/j178/prek)** — pre-commit hooks

### Recommended VSCode Extensions

Install the following extensions for inline feedback while coding:

- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (`charliermarsh.ruff`) — linting and formatting on save
- [ty](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty) (`astral-sh.ty`) — type checking
- [Even better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) (`tamasfe.even-better-toml`) - TOML color syntaxing
