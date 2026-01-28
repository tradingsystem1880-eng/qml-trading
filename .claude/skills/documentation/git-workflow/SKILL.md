# Git Workflow Skill

Conventional commits and Git best practices.

## When to Use
- Writing commit messages
- Creating branches
- Managing Git workflow
- Following project conventions

## Conventional Commits

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

| Type | When to Use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Build process, tooling, dependencies |

### Examples

```bash
# New feature
git commit -m "feat(detection): add hierarchical swing detector

Implements 3-layer swing detection:
- Geometry layer: raw swing identification
- Significance layer: filter noise
- Context layer: trend awareness"

# Bug fix
git commit -m "fix(simulation): correct trailing stop activation level

Trailing stop was activating at entry price instead of 1.5R profit.
This caused premature exits with tiny profits."

# Documentation
git commit -m "docs: update CLAUDE.md with Phase 9.5 results"

# Refactor
git commit -m "refactor(risk): consolidate position sizing into single module"

# Performance
git commit -m "perf(backtest): parallelize symbol processing

Reduced backtest time from 45min to 8min using multiprocessing."

# Chore
git commit -m "chore: update dependencies to latest versions"
```

### Scopes (QML Project)

| Scope | Area |
|-------|------|
| `detection` | Pattern detection |
| `simulation` | Trade simulation |
| `risk` | Risk management |
| `ml` | Machine learning |
| `dashboard` | Streamlit UI |
| `data` | Data pipeline |
| `validation` | Strategy validation |
| `execution` | Trade execution |

## Branch Naming

### Format

```
<type>/<short-description>
```

### Examples

```bash
# Feature branches
git checkout -b feat/funding-rate-filter
git checkout -b feat/bybit-paper-trading

# Bug fix branches
git checkout -b fix/trailing-stop-bug
git checkout -b fix/regime-filter-timing

# Documentation
git checkout -b docs/phase97-summary

# Experimental
git checkout -b experiment/ml-meta-labeling
```

## Common Workflows

### Feature Development

```bash
# 1. Start from main
git checkout main
git pull

# 2. Create feature branch
git checkout -b feat/new-feature

# 3. Make changes with atomic commits
git add src/detection/new_detector.py
git commit -m "feat(detection): add new detector base class"

git add src/detection/swing_detector.py
git commit -m "feat(detection): implement swing detection logic"

git add tests/test_swing_detector.py
git commit -m "test(detection): add swing detector tests"

# 4. Push and create PR
git push -u origin feat/new-feature
gh pr create --title "feat(detection): add swing detector" --body "..."
```

### Quick Fix

```bash
# 1. Create fix branch from main
git checkout main
git pull
git checkout -b fix/bug-description

# 2. Fix and commit
git add .
git commit -m "fix(module): description of fix"

# 3. Push and merge
git push -u origin fix/bug-description
gh pr create --title "fix: description" --body "..."
```

### Stashing Work

```bash
# Save current work
git stash save "WIP: description"

# List stashes
git stash list

# Apply most recent
git stash pop

# Apply specific stash
git stash apply stash@{1}

# Drop stash
git stash drop stash@{0}
```

## Useful Commands

### Status and History

```bash
# Status
git status
git status -s  # Short format

# History
git log --oneline -10
git log --oneline --graph --all  # Visual branch graph
git log --since="2 weeks ago"

# Show specific commit
git show abc1234
```

### Undoing Changes

```bash
# Unstage file
git reset HEAD <file>

# Discard changes in working directory
git checkout -- <file>

# Amend last commit (careful!)
git commit --amend -m "new message"

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes - careful!)
git reset --hard HEAD~1
```

### Comparing

```bash
# Working directory vs staged
git diff

# Staged vs last commit
git diff --staged

# Between branches
git diff main..feature-branch

# Specific file between branches
git diff main..feature-branch -- path/to/file.py
```

### Remote Operations

```bash
# Fetch without merge
git fetch origin

# Pull with rebase (cleaner history)
git pull --rebase origin main

# Push new branch
git push -u origin branch-name

# Delete remote branch
git push origin --delete branch-name
```

## Git Aliases

```bash
# Add to ~/.gitconfig
[alias]
    co = checkout
    br = branch
    ci = commit
    st = status -s
    lg = log --oneline --graph --all -15
    last = log -1 HEAD
    unstage = reset HEAD --
    amend = commit --amend --no-edit
```

## Pre-Commit Message Template

```bash
# Set up commit template
git config commit.template ~/.gitmessage

# ~/.gitmessage
# <type>(<scope>): <subject>
#
# [body]
#
# [footer]
#
# Types: feat, fix, docs, style, refactor, perf, test, chore
# Scopes: detection, simulation, risk, ml, dashboard, data, validation, execution
```

## QML-Specific Guidelines

1. **Always reference phase numbers** in feature commits:
   ```
   feat(validation): add Phase 9.5 validation suite
   ```

2. **Include key metrics** for trading changes:
   ```
   fix(simulation): correct trailing stop logic

   Before: WR 79%, PF 7.16, many dust wins
   After: WR 85.7%, PF 8.53, no dust wins
   ```

3. **Mark experiments clearly**:
   ```
   experiment(ml): test meta-labeling approach

   RESULT: AUC 0.53 - no predictive power
   VERDICT: FAIL - do not integrate
   ```

4. **Document breaking changes**:
   ```
   refactor(config)!: change config structure

   BREAKING CHANGE: Config now uses nested dict for risk params.
   Update config files accordingly.
   ```
