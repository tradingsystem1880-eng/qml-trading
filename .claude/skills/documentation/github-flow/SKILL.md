# GitHub Flow Skill

Issue to PR workflow for GitHub-based development.

## When to Use
- Creating GitHub issues
- Managing pull requests
- Code review workflow
- Project management

## GitHub Flow Overview

```
1. Create Issue → 2. Create Branch → 3. Make Changes → 4. Open PR → 5. Review → 6. Merge
```

## Issue Management

### Creating Issues

```bash
# Create issue via CLI
gh issue create --title "feat: Add funding rate filter" \
  --body "## Description
Add a filter to skip trades during extreme funding rate periods.

## Motivation
Extreme funding (±0.01%) correlates with increased volatility.

## Acceptance Criteria
- [ ] Fetch historical funding rates
- [ ] Filter trades when |funding| > 0.01%
- [ ] Validate with permutation test

## Related
- Phase 9.7 planning doc"

# Create with labels
gh issue create --title "bug: Trailing stop activates too early" \
  --label "bug" --label "priority:high"

# Create from template
gh issue create --template bug_report.md
```

### Issue Templates

```markdown
<!-- .github/ISSUE_TEMPLATE/feature_request.md -->
---
name: Feature Request
about: Suggest a new feature
labels: enhancement
---

## Summary
<!-- One-line description -->

## Motivation
<!-- Why is this needed? -->

## Proposed Solution
<!-- How should it work? -->

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Alternatives Considered
<!-- Other approaches you've thought about -->

## Additional Context
<!-- Screenshots, links, related issues -->
```

```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: Bug Report
about: Report a bug
labels: bug
---

## Description
<!-- What happened? -->

## Expected Behavior
<!-- What should have happened? -->

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Environment
- Python version:
- OS:
- Relevant config:

## Logs/Screenshots
<!-- Paste relevant logs or screenshots -->
```

### Working with Issues

```bash
# List issues
gh issue list
gh issue list --label "bug"
gh issue list --assignee "@me"

# View issue
gh issue view 123

# Close issue
gh issue close 123 --comment "Fixed in #125"

# Reopen
gh issue reopen 123

# Add labels
gh issue edit 123 --add-label "priority:high"

# Assign
gh issue edit 123 --add-assignee "username"
```

## Pull Request Workflow

### Creating PRs

```bash
# Create branch from issue
gh issue develop 123 --checkout

# Or manually
git checkout -b feat/123-funding-filter

# Make changes, commit...

# Push and create PR
git push -u origin feat/123-funding-filter

gh pr create \
  --title "feat: Add funding rate filter" \
  --body "## Summary
Implements funding rate filter for Phase 9.7.

## Changes
- Added `src/data/funding_rates.py`
- Added filter logic in pattern scorer
- Added validation script

## Test Plan
- [x] Unit tests pass
- [x] Permutation test (p < 0.05)
- [x] Walk-forward validation (4/5 folds)

Closes #123" \
  --assignee "@me" \
  --reviewer "reviewer-username"
```

### PR Template

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->
## Summary
<!-- What does this PR do? -->

## Changes
<!-- List of changes -->
-
-

## Test Plan
<!-- How was this tested? -->
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Screenshots
<!-- If applicable -->

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
<!-- Link related issues -->
Closes #
```

### Managing PRs

```bash
# List PRs
gh pr list
gh pr list --state open --author "@me"

# View PR
gh pr view 125
gh pr view 125 --web  # Open in browser

# Check out PR locally
gh pr checkout 125

# Review PR
gh pr review 125 --approve
gh pr review 125 --request-changes --body "Please fix X"
gh pr review 125 --comment --body "Looks good, minor suggestion..."

# Merge PR
gh pr merge 125 --squash --delete-branch

# Close without merging
gh pr close 125
```

### Code Review

```bash
# View PR diff
gh pr diff 125

# View files changed
gh pr view 125 --json files --jq '.files[].path'

# Add comment on specific line
gh api repos/{owner}/{repo}/pulls/125/comments \
  -f body="Consider using..." \
  -f commit_id="abc123" \
  -f path="src/file.py" \
  -F line=42

# View review comments
gh pr view 125 --comments
```

## Branch Protection

```yaml
# Settings for main branch (set via GitHub UI or API)
# Settings > Branches > Add rule for "main"

# Required:
- Require pull request reviews before merging
  - Required approving reviews: 1
  - Dismiss stale pull request approvals when new commits are pushed
- Require status checks to pass before merging
  - Required checks: lint, test
- Require conversation resolution before merging
- Do not allow bypassing the above settings
```

## Workflow Automation

### Auto-assign Reviewers

```yaml
# .github/workflows/auto-assign.yml
name: Auto Assign

on:
  pull_request:
    types: [opened]

jobs:
  assign:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v1
        with:
          configuration-path: '.github/auto-assign.yml'
```

```yaml
# .github/auto-assign.yml
addReviewers: true
addAssignees: true
numberOfReviewers: 1
reviewers:
  - reviewer-username
```

### Auto-label PRs

```yaml
# .github/workflows/labeler.yml
name: Labeler

on:
  pull_request:
    types: [opened]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v4
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

```yaml
# .github/labeler.yml
detection:
  - src/detection/**/*

simulation:
  - src/optimization/**/*

documentation:
  - docs/**/*
  - '**/*.md'
```

## Complete Workflow Example

```bash
# 1. Find or create issue
gh issue list --label "priority:high"
gh issue create --title "feat: Add new filter"

# 2. Create branch from issue
gh issue develop 123 --checkout
# Creates: feat/123-add-new-filter

# 3. Make changes
# ... code ...
git add .
git commit -m "feat(filter): implement new filter logic"

# 4. Push and create PR
git push -u origin feat/123-add-new-filter
gh pr create --fill  # Uses commit message and branch name

# 5. Request review
gh pr edit 125 --add-reviewer teammate

# 6. Address feedback
git commit -m "fix: address review feedback"
git push

# 7. Merge after approval
gh pr merge 125 --squash --delete-branch

# 8. Verify issue closed
gh issue view 123
```

## Quick Reference

| Task | Command |
|------|---------|
| Create issue | `gh issue create` |
| List issues | `gh issue list` |
| Create PR | `gh pr create` |
| List PRs | `gh pr list` |
| View PR | `gh pr view 123` |
| Checkout PR | `gh pr checkout 123` |
| Review PR | `gh pr review 123 --approve` |
| Merge PR | `gh pr merge 123 --squash` |
| Close issue | `gh issue close 123` |
