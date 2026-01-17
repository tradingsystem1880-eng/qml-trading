# Repository Setup Guide

## ✅ Current Status

Your QML_SYSTEM workspace is now a fully initialized Git repository with:
- ✓ Comprehensive `.gitignore` for Python, data files, models, and results
- ✓ Clean commit history with meaningful messages
- ✓ All code organized and archived legacy files moved to `.archive/`
- ✓ Ready to push to a remote repository (GitHub, GitLab, etc.)

## 📊 Repository Statistics

```bash
# View commit history
git log --oneline --graph

# Check repository size
du -sh .git/

# View file statistics
git ls-files | wc -l
```

## 🚀 Next Steps: Push to Remote

### Option 1: GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name: `qml-trading-system` (or your preferred name)
   - Description: "Quantitative Machine Learning Trading System - QML Pattern Detection & Validation"
   - Make it **Private** (recommended for trading systems)
   - Don't initialize with README, .gitignore, or license (we already have them)

2. **Connect your local repo to GitHub:**
   ```bash
   cd /Users/hunternovotny/Desktop/QML_SYSTEM
   git remote add origin https://github.com/YOUR_USERNAME/qml-trading-system.git
   git branch -M main
   git push -u origin main
   ```

### Option 2: GitLab

1. **Create a new project on GitLab:**
   - Go to https://gitlab.com/projects/new
   - Name: `qml-trading-system`
   - Visibility: Private

2. **Connect your local repo:**
   ```bash
   cd /Users/hunternovotny/Desktop/QML_SYSTEM
   git remote add origin https://gitlab.com/YOUR_USERNAME/qml-trading-system.git
   git branch -M main
   git push -u origin main
   ```

### Option 3: Bitbucket

1. **Create a new repository on Bitbucket:**
   - Go to https://bitbucket.org/repo/create
   - Name: `qml-trading-system`
   - Access: Private

2. **Connect your local repo:**
   ```bash
   cd /Users/hunternovotny/Desktop/QML_SYSTEM
   git remote add origin https://bitbucket.org/YOUR_USERNAME/qml-trading-system.git
   git branch -M main
   git push -u origin main
   ```

## 🔐 Security Best Practices

### Sensitive Files Already Excluded

Your `.gitignore` already excludes:
- `.env` files (API keys, secrets)
- `logs/` directory
- Large data files (`.csv`, `.parquet`)
- Model files (`.pkl`, `.h5`, `.pth`)
- Database files (`.db`, `.sqlite`)

### Additional Security Steps

1. **Never commit API keys or credentials:**
   ```bash
   # Create a .env file for secrets (already in .gitignore)
   echo "BINANCE_API_KEY=your_key" >> .env
   echo "BINANCE_SECRET=your_secret" >> .env
   ```

2. **Use environment variables in code:**
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   api_key = os.getenv('BINANCE_API_KEY')
   ```

3. **Scan for accidentally committed secrets:**
   ```bash
   # Install git-secrets
   brew install git-secrets
   
   # Setup
   git secrets --install
   git secrets --register-aws
   ```

## 📝 Git Workflow Tips

### Daily Workflow

```bash
# Check status
git status

# Stage specific files
git add src/detection/new_detector.py

# Commit with descriptive message
git commit -m "feat: Add RSI divergence detector for QML patterns"

# Push to remote
git push origin main
```

### Branching Strategy

```bash
# Create feature branch
git checkout -b feature/ml-predictor

# Work on your feature...
git add .
git commit -m "feat: Implement XGBoost pattern predictor"

# Push branch
git push -u origin feature/ml-predictor

# Merge back to main (after review)
git checkout main
git merge feature/ml-predictor
git push origin main
```

### Common Commands

```bash
# View changes before committing
git diff

# Undo unstaged changes
git restore filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View commit history
git log --oneline --graph --all

# Check which files are tracked
git ls-files
```

## 📦 Repository Structure

```
QML_SYSTEM/
├── .git/                  # Git repository data
├── .gitignore            # Files to exclude from Git
├── README.md             # Project overview
├── REPOSITORY_SETUP.md   # This file
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
│
├── cli/                  # Command-line interfaces
├── src/                  # Source code
├── qml/                  # QML module
├── config/               # Configuration files
├── docs/                 # Documentation
├── tests/                # Unit tests
│
├── data/                 # Data files (mostly ignored)
├── results/              # Results (mostly ignored)
├── models/               # ML models (ignored)
├── logs/                 # Log files (ignored)
│
└── .archive/             # Legacy code (tracked but inactive)
```

## 🏷️ Recommended Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(detection): Add MACD divergence detector"
git commit -m "fix(backtest): Correct position sizing calculation"
git commit -m "docs: Update QUICKSTART.md with new examples"
git commit -m "refactor(validation): Simplify bootstrap validation logic"
```

## 🔄 Keeping Repository Clean

### Remove Large Files If Needed

```bash
# Find large files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sort -k3 -n -r | \
  head -20

# Remove file from history (use carefully!)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file' \
  --prune-empty --tag-name-filter cat -- --all
```

### Cleanup

```bash
# Remove untracked files (dry run)
git clean -n

# Actually remove (be careful!)
git clean -f

# Also remove untracked directories
git clean -fd
```

## 📊 Repository Metrics

```bash
# Count commits
git rev-list --count main

# Contributors
git shortlog -sn

# Code statistics
git ls-files | xargs wc -l

# Repository size
du -sh .git/
```

## 🆘 Troubleshooting

### Problem: Accidentally committed large files

```bash
# Remove from staging
git rm --cached large_file.csv

# Update .gitignore
echo "large_file.csv" >> .gitignore

# Commit
git commit -m "Remove large file from tracking"
```

### Problem: Wrong commit message

```bash
# Change last commit message
git commit --amend -m "New message"

# If already pushed (use carefully!)
git push --force origin main
```

### Problem: Need to undo changes

```bash
# Discard all local changes
git reset --hard HEAD

# Go back to specific commit
git reset --hard <commit-hash>
```

## 🎯 Recommended Repository Settings

If using GitHub:
1. **Enable branch protection** for `main`
2. **Require pull request reviews** before merging
3. **Enable automated tests** with GitHub Actions
4. **Add repository topics**: `algorithmic-trading`, `python`, `quantitative-finance`, `machine-learning`
5. **Add a LICENSE** file (if open-sourcing)

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Git Best Practices](https://sethrobertson.github.io/GitBestPractices/)

---

**Repository initialized on:** January 17, 2026
**Last updated:** January 17, 2026
