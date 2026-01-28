# Python Linting Skill

Ruff configuration and style enforcement for Python projects.

## When to Use
- Setting up code quality tools
- Configuring linting rules
- Auto-formatting code
- CI/CD quality gates

## Ruff Setup

### Installation

```bash
pip install ruff
```

### Configuration (pyproject.toml)

```toml
[tool.ruff]
# Target Python version
target-version = "py310"

# Line length
line-length = 100

# Include/exclude paths
include = ["src/**/*.py", "cli/**/*.py", "scripts/**/*.py", "tests/**/*.py"]
exclude = ["archive/", ".venv/", "build/"]

[tool.ruff.lint]
# Enable rule sets
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "PD",     # pandas-vet
    "NPY",    # NumPy-specific rules
]

# Ignore specific rules
ignore = [
    "E501",   # Line too long (handled by formatter)
    "B008",   # Function call in default argument
    "PD901",  # df is a bad variable name (common in trading)
]

# Allow autofix for all enabled rules
fixable = ["ALL"]
unfixable = []

[tool.ruff.lint.per-file-ignores]
# Tests can have unused imports for fixtures
"tests/**/*.py" = ["F401", "ARG"]
# Scripts can have print statements
"scripts/**/*.py" = ["T201"]

[tool.ruff.lint.isort]
# Import sorting configuration
known-first-party = ["src", "qml", "cli"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]

[tool.ruff.format]
# Formatting options
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

## Usage

### Command Line

```bash
# Check for issues
ruff check .

# Check with auto-fix
ruff check --fix .

# Format code
ruff format .

# Check specific file
ruff check src/detection/pattern_scorer.py

# Show rule explanation
ruff rule E501
```

### VS Code Integration

```json
// .vscode/settings.json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "ruff.lint.enable": true,
    "ruff.format.enable": true
}
```

## Common Fixes

### E501: Line Too Long

```python
# ❌ Before
result = some_function(parameter_one, parameter_two, parameter_three, parameter_four, parameter_five)

# ✅ After
result = some_function(
    parameter_one,
    parameter_two,
    parameter_three,
    parameter_four,
    parameter_five,
)
```

### F401: Unused Import

```python
# ❌ Before
import os
import sys  # Never used

# ✅ After
import os
```

### I001: Unsorted Imports

```python
# ❌ Before
import pandas as pd
import os
from src.detection import get_detector
import numpy as np

# ✅ After (sorted by ruff)
import os

import numpy as np
import pandas as pd

from src.detection import get_detector
```

### B006: Mutable Default Argument

```python
# ❌ Before
def process(items=[]):
    items.append(1)
    return items

# ✅ After
def process(items=None):
    if items is None:
        items = []
    items.append(1)
    return items
```

### SIM105: Use contextlib.suppress

```python
# ❌ Before
try:
    os.remove(filepath)
except FileNotFoundError:
    pass

# ✅ After
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove(filepath)
```

### UP035: Use PEP 585 Type Hints

```python
# ❌ Before (Python 3.9+)
from typing import List, Dict, Optional

def process(items: List[str]) -> Dict[str, int]:
    ...

# ✅ After
def process(items: list[str]) -> dict[str, int]:
    ...
```

### PD002: Use .to_numpy() Instead of .values

```python
# ❌ Before
arr = df['close'].values

# ✅ After
arr = df['close'].to_numpy()
```

## Trading-Specific Rules

### Custom Rule: No Print in Production

```toml
# In pyproject.toml
[tool.ruff.lint]
select = ["T201"]  # print found

[tool.ruff.lint.per-file-ignores]
"scripts/**/*.py" = ["T201"]  # Allow in scripts
```

### Custom Rule: Require Type Hints

```toml
[tool.ruff.lint]
select = ["ANN"]  # flake8-annotations

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ANN"]  # Relax in tests
```

## Pre-Commit Integration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

## CI Integration

```yaml
# .github/workflows/lint.yml
name: Lint

on: [push, pull_request]

jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v1
        with:
          args: "check --output-format=github"
      - uses: astral-sh/ruff-action@v1
        with:
          args: "format --check"
```

## Quick Commands

```bash
# Check entire project
ruff check . && ruff format --check .

# Fix everything
ruff check --fix . && ruff format .

# Check what would change
ruff check --diff .
ruff format --diff .

# Stats on violations
ruff check . --statistics
```
