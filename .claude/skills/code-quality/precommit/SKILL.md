# Pre-Commit Skill

Git hooks and quality gates for automated code validation.

## When to Use
- Setting up pre-commit hooks
- Enforcing code quality before commits
- Automating formatting and linting
- CI/CD quality gates

## Pre-Commit Setup

### Installation

```bash
pip install pre-commit
```

### Configuration

```yaml
# .pre-commit-config.yaml
repos:
  # Ruff - Fast Python linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements  # No pdb in commits
      - id: detect-private-key

  # Python-specific
  - repo: https://github.com/pre-commit/pygrep-hooks
    rev: v1.10.0
    hooks:
      - id: python-check-blanket-noqa
      - id: python-check-blanket-type-ignore
      - id: python-no-eval
      - id: python-use-type-annotations

  # Type checking (optional, can be slow)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, types-requests]
        args: [--ignore-missing-imports]
        files: ^src/

  # Security scanning
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

# CI configuration
ci:
  autofix_commit_msg: 'style: auto-fix by pre-commit hooks'
  autoupdate_commit_msg: 'chore: update pre-commit hooks'
```

### Installation Commands

```bash
# Install hooks (run once after cloning)
pre-commit install

# Install commit-msg hook for conventional commits
pre-commit install --hook-type commit-msg

# Update hooks to latest versions
pre-commit autoupdate

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "emergency fix"
```

## Custom Hooks

### Trading-Specific Hook

```yaml
# In .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-trading-code
        name: Trading Code Checks
        entry: python scripts/check_trading_code.py
        language: python
        files: ^src/
        types: [python]
```

```python
# scripts/check_trading_code.py
#!/usr/bin/env python
"""Custom trading code checks."""
import sys
import re
from pathlib import Path

def check_file(filepath: Path) -> list[str]:
    """Check a single file for trading code issues."""
    issues = []
    content = filepath.read_text()
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Check for hardcoded API keys
        if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', line, re.I):
            issues.append(f"{filepath}:{i}: Hardcoded API key detected")

        # Check for look-ahead patterns
        if re.search(r'iloc\[\s*\w+\s*:\s*\w+\s*\+', line):
            issues.append(f"{filepath}:{i}: Potential look-ahead bias (forward slice)")

        # Check for print statements in src/
        if filepath.parts[0] == 'src' and re.match(r'\s*print\(', line):
            issues.append(f"{filepath}:{i}: print() in production code (use logging)")

    return issues

def main():
    """Run checks on changed files."""
    files = [Path(f) for f in sys.argv[1:] if f.endswith('.py')]
    all_issues = []

    for filepath in files:
        if filepath.exists():
            issues = check_file(filepath)
            all_issues.extend(issues)

    if all_issues:
        print("Trading code issues found:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Conventional Commits Hook

```yaml
# In .pre-commit-config.yaml
repos:
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.1.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [feat, fix, docs, style, refactor, perf, test, chore]
```

## Bandit Security Configuration

```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "archive", ".venv"]
skips = ["B101"]  # Skip assert warnings (OK in tests)

[tool.bandit.assert_used]
skips = ["*_test.py", "*test_*.py"]
```

## Quality Gates

### Local Workflow

```bash
# Before committing
pre-commit run --all-files

# Before pushing (more thorough)
pytest
ruff check .
```

### Makefile Integration

```makefile
# Makefile
.PHONY: lint test check

lint:
	ruff check . --fix
	ruff format .

test:
	pytest -v

check: lint test
	pre-commit run --all-files

install-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg
```

### GitHub Actions

```yaml
# .github/workflows/quality.yml
name: Code Quality

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: pre-commit/action@v3.0.0

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest --cov=src
```

## Bypassing Hooks

```bash
# Skip all hooks (emergency only)
git commit --no-verify -m "fix: emergency hotfix"

# Skip specific hook
SKIP=mypy git commit -m "feat: add new feature"

# Skip multiple hooks
SKIP=mypy,ruff git commit -m "wip: work in progress"
```

## Troubleshooting

```bash
# Hook not running
pre-commit install --force

# Hook taking too long
pre-commit run <hook-id> --files <specific-file>

# Clear pre-commit cache
pre-commit clean
pre-commit gc

# Debug hook execution
pre-commit run --verbose

# Check hook configuration
pre-commit validate-config
```

## Quick Setup Script

```bash
#!/bin/bash
# setup_quality.sh - One-time setup for code quality tools

set -e

echo "Installing pre-commit..."
pip install pre-commit ruff

echo "Installing git hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

echo "Running initial checks..."
pre-commit run --all-files || true

echo "Setup complete! Pre-commit hooks are now active."
```
