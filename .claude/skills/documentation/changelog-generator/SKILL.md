# Changelog Generator Skill

Auto-generate release notes from Git commits.

## When to Use
- Creating release notes
- Summarizing changes between versions
- Generating CHANGELOG.md updates
- Documenting phase completions

## Changelog Format (Keep a Changelog)

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Vulnerability fixes

## [1.0.0] - 2026-01-28

### Added
- Initial release
```

## Generate from Commits

### Python Script

```python
#!/usr/bin/env python
"""Generate changelog from conventional commits."""

import subprocess
import re
from collections import defaultdict
from datetime import datetime

def get_commits_since_tag(tag: str = None) -> list:
    """Get commits since last tag or all commits."""
    if tag:
        cmd = f"git log {tag}..HEAD --pretty=format:'%H|%s|%b---'"
    else:
        cmd = "git log --pretty=format:'%H|%s|%b---'"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    commits = result.stdout.split('---')

    parsed = []
    for commit in commits:
        if '|' not in commit:
            continue
        parts = commit.strip().split('|', 2)
        if len(parts) >= 2:
            parsed.append({
                'hash': parts[0][:7],
                'subject': parts[1],
                'body': parts[2] if len(parts) > 2 else ''
            })
    return parsed

def parse_conventional_commit(subject: str) -> tuple:
    """Parse conventional commit format."""
    # Pattern: type(scope): subject
    pattern = r'^(\w+)(?:\(([^)]+)\))?!?:\s*(.+)$'
    match = re.match(pattern, subject)

    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, subject

def categorize_commits(commits: list) -> dict:
    """Group commits by type."""
    categories = defaultdict(list)

    type_to_category = {
        'feat': 'Added',
        'fix': 'Fixed',
        'docs': 'Documentation',
        'style': 'Changed',
        'refactor': 'Changed',
        'perf': 'Performance',
        'test': 'Testing',
        'chore': 'Maintenance',
    }

    for commit in commits:
        commit_type, scope, description = parse_conventional_commit(commit['subject'])
        category = type_to_category.get(commit_type, 'Other')

        entry = description
        if scope:
            entry = f"**{scope}**: {description}"

        categories[category].append({
            'text': entry,
            'hash': commit['hash']
        })

    return categories

def generate_changelog(
    version: str = "Unreleased",
    since_tag: str = None,
    include_hash: bool = True
) -> str:
    """Generate changelog markdown."""
    commits = get_commits_since_tag(since_tag)
    categories = categorize_commits(commits)

    # Order of sections
    section_order = ['Added', 'Changed', 'Fixed', 'Performance', 'Documentation', 'Testing', 'Maintenance', 'Other']

    lines = []
    date = datetime.now().strftime('%Y-%m-%d')
    lines.append(f"## [{version}] - {date}\n")

    for section in section_order:
        if section in categories:
            lines.append(f"### {section}\n")
            for item in categories[section]:
                if include_hash:
                    lines.append(f"- {item['text']} ({item['hash']})")
                else:
                    lines.append(f"- {item['text']}")
            lines.append("")

    return '\n'.join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='Unreleased')
    parser.add_argument('--since', help='Tag to start from')
    parser.add_argument('--no-hash', action='store_true')
    args = parser.parse_args()

    changelog = generate_changelog(
        version=args.version,
        since_tag=args.since,
        include_hash=not args.no_hash
    )
    print(changelog)
```

### Usage

```bash
# Generate changelog for unreleased changes
python scripts/generate_changelog.py

# Generate for specific version since tag
python scripts/generate_changelog.py --version 1.0.0 --since v0.9.0

# Without commit hashes
python scripts/generate_changelog.py --no-hash

# Append to CHANGELOG.md
python scripts/generate_changelog.py >> CHANGELOG.md
```

## QML Phase Changelog

### Template for Phase Completion

```markdown
## Phase 9.X: [Phase Name] - YYYY-MM-DD

### Summary
[One paragraph describing what this phase accomplished]

### Results
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Win Rate | X% | Y% | +Z% |
| Profit Factor | X | Y | +Z |
| Expectancy | XR | YR | +ZR |

### Added
- [New feature 1]
- [New feature 2]

### Changed
- [Change 1]
- [Change 2]

### Key Files
- `src/module/file.py` - Description
- `scripts/script.py` - Description

### Verdict
**[PASS/FAIL]**: [Explanation]

### Next Steps
- [Next step 1]
- [Next step 2]
```

### Example: Phase 9.5

```markdown
## Phase 9.5: Final Validation Suite + Bybit Integration - 2026-01-27

### Summary
Complete validation suite (6 statistical tests) and Bybit testnet paper trading integration for forward testing.

### Results
| Test | Result | Status |
|------|--------|--------|
| Permutation | p < 0.05 | ✅ PASS |
| Monte Carlo | 95% DD < 20% | ✅ PASS |
| OOS Holdout | PF > 2.0 | ✅ PASS |
| Parameter Sensitivity | PF range < 1.5 | ✅ PASS |
| Stress Test | Avg PF > 1.0 | ✅ PASS |
| Trade Correlation | \|r\| < 0.1 | ✅ PASS |

### Added
- 6 validation test scripts in `scripts/phase95_*.py`
- `src/execution/` module for Bybit trading
- Forward test dashboard page
- Phase-based risk scaling

### Changed
- Paper trader now uses HierarchicalSwingDetector
- State persistence added for restarts

### Key Files
- `src/execution/bybit_client.py` - CCXT wrapper
- `src/execution/paper_trader_bybit.py` - Paper trading logic
- `scripts/run_bybit_paper_trader.py` - CLI interface

### Verdict
**PASS**: All validation tests passed. Ready for forward testing.

### Next Steps
1. Configure Bybit testnet API
2. Start Phase 1 paper trading (50 trades @ 0.5% risk)
3. Monitor with dashboard
```

## Git Commands for Changelog

```bash
# List commits since tag
git log v1.0.0..HEAD --oneline

# List commits by type
git log --oneline | grep "^[a-f0-9]* feat"

# Count commits by type
git log --oneline | cut -d' ' -f2 | cut -d'(' -f1 | cut -d':' -f1 | sort | uniq -c

# Show files changed in recent commits
git log --oneline --name-only -10

# Compare tags
git log v1.0.0..v1.1.0 --oneline
```

## Automated Changelog Tools

### git-cliff

```bash
# Install
cargo install git-cliff

# Generate
git cliff -o CHANGELOG.md

# Since specific commit
git cliff --since v1.0.0
```

### conventional-changelog

```bash
# Install
npm install -g conventional-changelog-cli

# Generate
conventional-changelog -p angular -i CHANGELOG.md -s
```

## Integration with CI

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  changelog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        run: |
          python scripts/generate_changelog.py \
            --version ${GITHUB_REF#refs/tags/} \
            --since $(git describe --tags --abbrev=0 HEAD^) \
            > RELEASE_NOTES.md

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body_path: RELEASE_NOTES.md
```
