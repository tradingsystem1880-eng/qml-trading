# Claude Code Prompt: Install Trading System Skills

Copy and paste this entire prompt into Claude Code to install all 20 recommended skills for the QML Trading System.

---

## THE PROMPT

```
I need you to install and configure 20 Claude Code skills for my QML Trading System. This is a quantitative trading validation framework for crypto markets.

## Context
- Project location: Current working directory (qml-trading-main)
- I'm ready for Bybit forward testing
- Skills should go in `.claude/skills/` directory
- Read CLAUDE.md first to understand the project

## Installation Tasks

### Step 1: Create Skills Directory Structure
Create the following directory structure:
```
.claude/
└── skills/
    ├── trading/
    ├── testing/
    ├── code-quality/
    ├── data-analysis/
    ├── documentation/
    └── error-handling/
```

### Step 2: Install Tier 1 Skills (Critical - Before Forward Testing)

**2a. error-logger skill**
Create `.claude/skills/error-handling/error-logger/SKILL.md`:
- Structured JSON logging with correlation IDs
- Use for: implementing logging, debugging failures, tracing requests across services
- Should trigger on: "add logging", "debug", "trace error", "correlation ID"

**2b. api-error-handling skill**
Create `.claude/skills/error-handling/api-error-handling/SKILL.md`:
- Comprehensive error handling for REST APIs
- Standardized error responses with proper HTTP status codes
- Retry logic with exponential backoff
- User-friendly error messages
- Should trigger on: "handle error", "API error", "retry logic", "error response"

**2c. tradeblocks-health-check skill**
Create `.claude/skills/trading/strategy-health-check/SKILL.md`:
- Strategy health diagnostics before live trading
- Performance metrics analysis (PF, Sharpe, Win Rate, Drawdown)
- Stress testing under different market conditions
- Risk indicator surfacing
- Should trigger on: "health check", "strategy validation", "stress test", "risk analysis"

### Step 3: Install Tier 2 Skills (Code Quality)

**3a. python-testing-patterns skill**
Create `.claude/skills/testing/python-testing/SKILL.md`:
- Pytest best practices (fixtures, parametrize, markers)
- Mocking strategies for external APIs and databases
- Test-driven development workflow
- Coverage analysis and improvement
- Should trigger on: "write test", "pytest", "mock", "fixture", "TDD"

**3b. code-reviewer skill**
Create `.claude/skills/code-quality/code-reviewer/SKILL.md`:
- Review staged changes before commit
- Review pull requests by URL or ID
- Check for: bugs, security issues, performance problems, style violations
- Provide constructive feedback with specific suggestions
- Should trigger on: "review code", "review PR", "check changes", "code review"

**3c. python-linting skill**
Create `.claude/skills/code-quality/python-linting/SKILL.md`:
- Ruff linter configuration and usage
- Auto-fix common issues
- Style guide enforcement (PEP 8, Google style)
- Security checks (bandit rules)
- Should trigger on: "lint", "fix style", "check code quality", "ruff"

**3d. precommit skill**
Create `.claude/skills/code-quality/precommit/SKILL.md`:
- Pre-commit hooks setup and configuration
- Hooks for: ruff, mypy, pytest, trailing whitespace, large files
- Instructions for `.pre-commit-config.yaml`
- Should trigger on: "pre-commit", "git hooks", "commit hooks"

### Step 4: Install Tier 3 Skills (Enhancement)

**4a. backtesting-frameworks skill**
Create `.claude/skills/trading/backtesting-frameworks/SKILL.md`:
- Proper handling of look-ahead bias
- Survivorship bias prevention
- Transaction cost modeling (slippage, fees, spread)
- Walk-forward validation methodology
- Out-of-sample testing best practices
- Should trigger on: "backtest", "look-ahead bias", "survivorship bias", "walk-forward"

**4b. quant-analyst skill**
Create `.claude/skills/trading/quant-analyst/SKILL.md`:
- Financial modeling (risk metrics, ratios)
- Portfolio optimization (Kelly criterion, position sizing)
- Statistical analysis (significance testing, confidence intervals)
- Performance attribution
- Should trigger on: "risk metrics", "Sharpe", "Kelly", "portfolio", "quant"

**4c. shap-explainability skill**
Create `.claude/skills/data-analysis/shap-explainability/SKILL.md`:
- SHAP (SHapley Additive eXplanations) for ML models
- Feature importance analysis
- Model debugging and interpretation
- Visualization of predictions
- Should trigger on: "explain model", "SHAP", "feature importance", "model interpretability"

**4d. walk-forward-analysis skill**
Create `.claude/skills/trading/walk-forward-analysis/SKILL.md`:
- Rolling train/test splits with purge gaps
- Parameter stability testing
- Regime-aware validation
- Out-of-sample performance tracking
- Should trigger on: "walk-forward", "rolling validation", "parameter stability"

**4e. pandas-data-analysis skill**
Create `.claude/skills/data-analysis/pandas-analysis/SKILL.md`:
- Pandas best practices for financial data
- Time series manipulation (resampling, rolling windows)
- Memory optimization for large datasets
- Parquet file handling
- Should trigger on: "pandas", "dataframe", "time series", "data analysis"

### Step 5: Install Tier 4 Skills (Documentation & Workflow)

**5a. git-workflow skill**
Create `.claude/skills/documentation/git-workflow/SKILL.md`:
- Conventional commit messages (feat:, fix:, docs:, refactor:)
- Branch naming conventions
- Proper staging and commit workflow
- Should trigger on: "commit", "git push", "conventional commit"

**5b. changelog-generator skill**
Create `.claude/skills/documentation/changelog-generator/SKILL.md`:
- Generate CHANGELOG.md from git history
- Semantic versioning support
- Group by: features, fixes, breaking changes
- Should trigger on: "changelog", "release notes", "version"

**5c. api-docs-generator skill**
Create `.claude/skills/documentation/api-docs/SKILL.md`:
- Auto-generate API documentation from code
- Docstring extraction and formatting
- Example generation
- Should trigger on: "generate docs", "API documentation", "docstrings"

**5d. github-issue-flow skill**
Create `.claude/skills/documentation/github-flow/SKILL.md`:
- End-to-end issue resolution workflow
- Branch creation from issue
- PR creation with issue reference
- Should trigger on: "fix issue", "issue #", "create PR"

### Step 6: Create Skills Index

Create `.claude/skills/README.md` with:
- List of all installed skills
- Quick reference for trigger phrases
- Links to each skill's SKILL.md

### Step 7: Verify Installation

After creating all skills:
1. List all files in `.claude/skills/` recursively
2. Verify each SKILL.md has proper structure
3. Test that skills are recognized by checking Claude's skill loading

## SKILL.md Template

Each skill should follow this structure:

```markdown
---
name: skill-name
description: Brief description
triggers:
  - "trigger phrase 1"
  - "trigger phrase 2"
version: 1.0.0
---

# Skill Name

## Purpose
[What this skill does]

## When to Use
[Trigger conditions]

## Instructions
[Detailed instructions for Claude]

## Examples
[Usage examples]

## Best Practices
[Domain-specific best practices]
```

## Important Notes

1. Make each skill self-contained with clear instructions
2. Include specific examples relevant to trading systems
3. Reference the QML project structure where appropriate
4. Keep skills focused - one skill per domain concern
5. Include error handling guidance in each skill

Start by creating the directory structure, then work through each skill in order. Show me the contents of each SKILL.md as you create it.
```

---

## Quick Version (Shorter Prompt)

If the above is too long, use this condensed version:

```
Install 20 Claude Code skills for my QML Trading System. Create `.claude/skills/` with these categories:

**Trading (5 skills):**
1. backtesting-frameworks - look-ahead bias, survivorship bias, transaction costs
2. quant-analyst - risk metrics, Kelly criterion, portfolio optimization
3. walk-forward-analysis - rolling validation, parameter stability
4. strategy-health-check - pre-live diagnostics, stress tests
5. parameter-optimizer - Bayesian optimization, grid search

**Testing (2 skills):**
6. python-testing - pytest fixtures, mocking, TDD
7. code-reviewer - automated code review, PR review

**Code Quality (2 skills):**
8. python-linting - ruff, style enforcement
9. precommit - git hooks, quality gates

**Data Analysis (3 skills):**
10. pandas-analysis - time series, financial data
11. shap-explainability - ML model interpretation
12. data-validation - quality scoring, missing data

**Documentation (4 skills):**
13. git-workflow - conventional commits
14. changelog-generator - release notes
15. api-docs - auto-generate documentation
16. github-flow - issue to PR workflow

**Error Handling (3 skills):**
17. error-logger - structured logging, correlation IDs
18. api-error-handling - retry logic, graceful degradation
19. exception-patterns - Python exception best practices
20. debugging-guide - systematic debugging workflow

For each skill, create a SKILL.md with: purpose, triggers, instructions, examples, and best practices relevant to quantitative trading. Start with error-handling skills since I'm about to begin forward testing on Bybit.
```

---

## Usage

1. Open Claude Code in your `qml-trading-main` directory
2. Copy one of the prompts above
3. Paste and run
4. Claude will create all skill files
5. Restart Claude Code to load the new skills

The skills will then automatically activate based on context when you're working on relevant tasks.
