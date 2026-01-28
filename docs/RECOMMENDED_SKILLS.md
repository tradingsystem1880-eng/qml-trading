# 20 Recommended Skills for QML Trading System

> **Generated**: 2026-01-28
> **Source**: [SkillsMP.com](https://skillsmp.com) - 107,000+ Claude Code Skills
> **Purpose**: Improve quality, robustness, and correctness of your quantitative trading system

---

## How to Install Skills

```bash
# Clone skill repository to your project
git clone <skill-repo-url> .claude/skills/<skill-name>

# Or add to personal skills directory
git clone <skill-repo-url> ~/.claude/skills/<skill-name>
```

Skills are automatically loaded by Claude when relevant to your task context.

---

## Category 1: Trading & Backtesting (Core Domain)

### 1. backtesting-frameworks ⭐ 27.1k
**Why**: Directly relevant to your VRD 2.0 validation framework. Teaches proper handling of look-ahead bias, survivorship bias, and transaction costs.

```
Repository: wshobson/agents
Description: Build robust backtesting systems for trading strategies with proper handling of look-ahead bias, survivorship bias, and transaction costs.
```

**Relevance to Your Project**: Your Phase 7.9 and Phase 9.x validation pipelines would benefit from the rigorous bias-prevention techniques this skill teaches.

---

### 2. quant-analyst ⭐ 168
**Why**: Builds financial models, backtest trading strategies, and analyze market data with proper risk metrics and portfolio optimization.

```
Repository: rmyndharis/antigravity-skills
Description: Build financial models, backtest trading strategies, and analyze market data. Implements risk metrics, portfolio optimization, and quantitative analysis workflows.
```

**Relevance to Your Project**: Complements your existing ML meta-labeling (Phase 8.0) and Kelly sizer components.

---

### 3. tradeblocks-wfa (Walk-Forward Analysis) ⭐ 30
**Why**: Critical for validating that optimized parameters hold on out-of-sample data - exactly what your Phase 9.4 walk-forward validation does.

```
Repository: davidromeo/tradeblocks
Description: Walk-forward analysis for trading strategies. Tests whether optimized parameters hold up on out-of-sample data.
```

**Relevance to Your Project**: Direct alignment with your `scripts/walk_forward_validation.py` - could improve methodology.

---

### 4. tradeblocks-optimize ⭐ 30
**Why**: Parameter exploration for trading backtests. Analyzes patterns across parameters like time of day, DTE, delta.

```
Repository: davidromeo/tradeblocks
Description: Parameter exploration for trading backtests. Analyzes trade data to find patterns across parameters like time of day, DTE, delta.
```

**Relevance to Your Project**: Would enhance your Phase 7.7 extended optimization (56-hour Bayesian optimization).

---

### 5. tradeblocks-health-check ⭐ 30
**Why**: Strategy health diagnostics with stress tests and risk indicators.

```
Repository: davidromeo/tradeblocks
Description: Strategy health check for trading backtests. Analyzes performance metrics, runs stress tests, and surfaces risk indicators.
```

**Relevance to Your Project**: Complements your Phase 9.5 stress test (`phase95_stress_test.py`).

---

## Category 2: Testing & Code Quality

### 6. python-testing-patterns ⭐ 27.1k
**Why**: Comprehensive testing strategies with pytest, fixtures, mocking, and TDD - essential for a production trading system.

```
Repository: wshobson/agents
Description: Implement comprehensive testing strategies with pytest, fixtures, mocking, and test-driven development. Use when writing Python tests.
```

**Relevance to Your Project**: Your `tests/` directory has 19 test files - this skill will help maintain quality.

---

### 7. testing-python ⭐ 22.4k
**Why**: Write and evaluate effective Python tests using pytest. Use when writing tests, reviewing test code, debugging test failures.

```
Repository: jlowin/fastmcp
Description: Write and evaluate effective Python tests using pytest. Use when writing tests, reviewing test code, debugging test failures.
```

**Relevance to Your Project**: Directly applicable to your validation test suite.

---

### 8. code-reviewer ⭐ 92.8k
**Why**: The most popular skill on SkillsMP. Reviews both local changes and remote Pull Requests with automated analysis.

```
Repository: google-gemini/gemini-cli
Description: Use this skill to review code. It supports both local changes (staged or working tree) and remote Pull Requests (by ID or URL).
```

**Relevance to Your Project**: Essential for maintaining code quality as your system grows.

---

### 9. code-review-excellence ⭐ 27.1k
**Why**: Master effective code review practices - constructive feedback, catch bugs early, foster knowledge sharing.

```
Repository: wshobson/agents
Description: Master effective code review practices to provide constructive feedback, catch bugs early, and foster knowledge sharing.
```

**Relevance to Your Project**: Helps maintain quality across your 30+ source files.

---

### 10. python-linting ⭐ 466
**Why**: Lint Python code using ruff - fast, modern linter covering style, security, and correctness issues.

```
Repository: OpenHands/software-agent-sdk
Description: This skill helps lint Python code using ruff. Use when the user asks to "lint", "check code quality", or "fix style issues".
```

**Relevance to Your Project**: Ensures consistent code style across your codebase.

---

## Category 3: Data Analysis & ML

### 11. pandas-data-analysis ⭐ 272
**Why**: Master data manipulation, analysis, and visualization with Pandas, NumPy, and Matplotlib.

```
Repository: benchflow-ai/skillsbench
Description: Master data manipulation, analysis, and visualization with Pandas, NumPy, and Matplotlib.
```

**Relevance to Your Project**: Your data pipeline (`src/data_engine.py`) and parquet files would benefit from best practices.

---

### 12. shap (Model Explainability) ⭐ 18.6k
**Why**: Model interpretability using SHAP (SHapley Additive eXplanations) - critical for understanding why your ML models make predictions.

```
Repository: davila7/claude-code-templates
Description: Model interpretability and explainability using SHAP. Use this skill when explaining machine learning model predictions.
```

**Relevance to Your Project**: Your XGBoost predictor (`src/ml/predictor.py`) needs explainability - SHAP would reveal why ML failed in Phase 8.0 (AUC 0.53).

---

### 13. data-validation-reporter ⭐ 1
**Why**: Generate interactive validation reports with quality scoring, missing data analysis, and type checking.

```
Repository: vamseeachanta/workspace-hub
Description: Generate interactive validation reports with quality scoring, missing data analysis, and type checking. Combines Pandas validation.
```

**Relevance to Your Project**: Complements your VRD 2.0 validation dossiers and HTML reports.

---

## Category 4: Documentation & Git Workflow

### 14. api-documentation-generator ⭐ 18.6k
**Why**: Generate comprehensive, developer-friendly API documentation from code.

```
Repository: davila7/claude-code-templates
Description: Generate comprehensive, developer-friendly API documentation from code, including endpoints, parameters, examples, and best practices.
```

**Relevance to Your Project**: Your 1,000+ line CLAUDE.md could be complemented with auto-generated API docs.

---

### 15. changelog-generator ⭐ 1.1k
**Why**: Generate changelog from git commits - auto-activating skill for Technical Documentation.

```
Repository: jeremylongshore/claude-code-plugins
Description: Generate changelog operations. Auto-activating skill for Technical Documentation. Triggers on: changelog, release notes.
```

**Relevance to Your Project**: Your project has extensive git history through Phases 7.x-9.x - a changelog would help track evolution.

---

### 16. git-pushing ⭐ 18.6k
**Why**: Stage, commit, and push git changes with conventional commit messages. Essential for professional git workflow.

```
Repository: davila7/claude-code-templates
Description: Stage, commit, and push git changes with conventional commit messages. Use when user wants to commit and push changes.
```

**Relevance to Your Project**: Standardizes your commit practices across all development phases.

---

### 17. gh-issue-fix-flow ⭐ 1.5k
**Why**: End-to-end GitHub issue fix workflow - from issue to code changes to PR.

```
Repository: Dimillian/Skills
Description: End-to-end GitHub issue fix workflow using gh, local code changes, builds/tests, and git push.
```

**Relevance to Your Project**: Streamlines bug fixes and feature implementations.

---

## Category 5: Error Handling & Debugging

### 18. error-logger ⭐ 124
**Why**: Structured JSON logging with correlation IDs for multi-service systems.

```
Repository: aiskillstore/marketplace
Description: Structured JSON logging with correlation IDs for multi-service systems. Use when implementing logging, debugging failures.
```

**Relevance to Your Project**: Your trading system needs robust logging - especially for forward testing on Bybit.

---

### 19. api-error-handling ⭐ 54
**Why**: Implement comprehensive error handling with standardized responses, logging, and user-friendly messages.

```
Repository: aj-geddes/useful-ai-prompts
Description: Implement comprehensive error handling with standardized responses, logging, monitoring, and user-friendly messages.
```

**Relevance to Your Project**: Critical for your Bybit API integration (`src/execution/bybit_client.py`).

---

### 20. precommit ⭐ 7
**Why**: Pre-commit hooks framework for multi-language code quality automation.

```
Repository: julianobarbosa/claude-code-skills
Description: Pre-commit hooks framework for multi-language code quality automation. USE WHEN setting up pre-commit OR configuring git hooks.
```

**Relevance to Your Project**: Catch issues before they enter your codebase - prevents bugs in trading logic.

---

## Priority Installation Order

Based on your current project state (ready for Bybit forward testing), I recommend installing in this order:

### Tier 1: Immediate (Before Forward Testing)
1. `error-logger` - Critical for monitoring live trades
2. `api-error-handling` - Bybit API resilience
3. `tradeblocks-health-check` - Validate strategy before going live

### Tier 2: Code Quality (This Week)
4. `python-testing-patterns` - Improve test coverage
5. `code-reviewer` - Review changes before commits
6. `python-linting` - Enforce code standards
7. `precommit` - Automate quality checks

### Tier 3: Enhancement (Next Sprint)
8. `backtesting-frameworks` - Improve validation methodology
9. `quant-analyst` - Enhance risk analysis
10. `shap` - Understand ML model behavior
11. `tradeblocks-wfa` - Better walk-forward analysis
12. `pandas-data-analysis` - Data pipeline best practices

### Tier 4: Documentation & Workflow
13. `git-pushing` - Standardize commits
14. `changelog-generator` - Track releases
15. `api-documentation-generator` - Auto-generate docs
16-20. Remaining skills as needed

---

## Quick Install Script

```bash
#!/bin/bash
# Create skills directory if it doesn't exist
mkdir -p .claude/skills

# Tier 1: Critical for forward testing
git clone https://github.com/aiskillstore/marketplace .claude/skills/error-logger
git clone https://github.com/aj-geddes/useful-ai-prompts .claude/skills/api-error-handling
git clone https://github.com/davidromeo/tradeblocks .claude/skills/tradeblocks

# Tier 2: Code quality
git clone https://github.com/wshobson/agents .claude/skills/wshobson-agents
git clone https://github.com/google-gemini/gemini-cli .claude/skills/code-reviewer
git clone https://github.com/OpenHands/software-agent-sdk .claude/skills/python-linting

echo "Skills installed! Restart Claude Code to activate."
```

---

## Sources

- [SkillsMP - Agent Skills Marketplace](https://skillsmp.com)
- [GitHub - travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills)
- [GitHub - tradermonty/claude-trading-skills](https://github.com/tradermonty/claude-trading-skills)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)

---

*This document was generated by analyzing your QML Trading System codebase and matching it with relevant skills from SkillsMP.com's 107,000+ skill directory.*
