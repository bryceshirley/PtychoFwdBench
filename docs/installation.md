## Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/bryceshirley/PtychoFwdBench.git
cd PtychoFwdBench
```

Ensure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/). Then simply run:

```bash
uv sync
```
This will create a virtual environment and install all required dependencies.

## Setup Development Hooks

To ensure code quality and run tests automatically, you must activate the git hooks:

```bash
# Activate the virtual environment (if not auto-activated by uv)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 1. Install pre-commit hooks (runs linting & formatting on commit)
pre-commit install

# 2. Install pre-push hooks (runs pytest on push)
pre-commit install --hook-type pre-push
```

Now, `ruff` will clean your code on every commit, and `pytest` will ensure stability before you push.

> Tip: If the `uv-lock` hook fails during a commit, it means your `uv.lock` file is out of sync. Run `uv lock` to update it, stage the file, and try committing again.
