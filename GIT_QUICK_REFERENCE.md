# Git Quick Reference for QML_SYSTEM

## 📋 Current Repository Status

✅ **Repository initialized**: `/Users/hunternovotny/Desktop/QML_SYSTEM`  
✅ **Branch**: `main`  
✅ **Total commits**: 6  
✅ **Working tree**: Clean  
✅ **Remote**: Not configured (ready to add)  

## 🚀 Quick Commands

### Check Status
```bash
git status          # See what's changed
git log --oneline   # View commit history
git diff            # See unstaged changes
```

### Make Changes
```bash
git add .                              # Stage all changes
git add src/detection/new_file.py      # Stage specific file
git commit -m "feat: Add new detector" # Commit with message
```

### Push to Remote (after setup)
```bash
# First time setup (replace with your repo URL)
git remote add origin https://github.com/username/qml-system.git
git push -u origin main

# After that, just:
git push
```

### Branching
```bash
git branch feature/new-feature    # Create branch
git checkout feature/new-feature  # Switch to branch
git checkout -b feature/xyz       # Create and switch
git merge feature/xyz             # Merge branch into current
```

### Undo Changes
```bash
git restore file.py           # Discard changes in file
git restore --staged file.py  # Unstage file
git reset --soft HEAD~1       # Undo last commit (keep changes)
git reset --hard HEAD~1       # Undo last commit (discard changes)
```

## 📦 What's Tracked vs Ignored

### ✅ Tracked (in Git)
- All Python source code (`src/`, `cli/`, `qml/`)
- Configuration files (`config/`, `pyproject.toml`)
- Documentation (`docs/`, `README.md`)
- Tests (`tests/`)
- Requirements (`requirements.txt`)
- GitHub workflows (`.github/`)
- Sample data structures

### 🚫 Ignored (not in Git)
- Large data files (`data/**/*.csv`, `*.parquet`)
- Model files (`models/*.pkl`, `*.h5`)
- Results (`results/backtests/*.csv`)
- Logs (`logs/`, `*.log`)
- Database files (`*.db`, `*.sqlite`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)
- Virtual environments (`venv/`)

## 🎯 Common Workflows

### Daily Work Cycle
```bash
# 1. Start work
git status  # See what's changed

# 2. Make changes to code
# ... edit files ...

# 3. Check what changed
git diff

# 4. Stage and commit
git add src/detection/new_detector.py
git commit -m "feat(detection): Add momentum divergence detector"

# 5. Push to remote
git push
```

### Before Starting New Feature
```bash
git checkout main                  # Switch to main branch
git pull                          # Get latest changes
git checkout -b feature/my-feature # Create feature branch
# ... work on feature ...
git add .
git commit -m "feat: Implement my feature"
git push -u origin feature/my-feature
```

### Update from Remote
```bash
git pull origin main  # Get latest from remote
git fetch            # Download remote changes (don't merge)
```

## 📊 Repository Info

```bash
# Count files tracked
git ls-files | wc -l

# Repository size
du -sh .git/

# File statistics
git ls-files | xargs wc -l | sort -rn | head -20

# Largest files
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# Contributors
git shortlog -sn

# Recent activity
git log --since="1 week ago" --oneline
```

## 🔗 Remote Repository Setup

When you're ready to push to GitHub/GitLab:

```bash
# GitHub
git remote add origin https://github.com/YOUR_USERNAME/qml-system.git
git push -u origin main

# Check remote
git remote -v

# Change remote URL if needed
git remote set-url origin https://github.com/NEW_USERNAME/new-repo.git
```

## 💡 Tips

1. **Commit often** - Small, focused commits are better
2. **Write clear messages** - Future you will thank you
3. **Check before pushing** - Review `git log` and `git diff`
4. **Use branches** - Keep `main` stable, experiment in branches
5. **Pull before push** - Avoid merge conflicts

## 🆘 Emergency Commands

```bash
# Accidentally added wrong files
git reset HEAD file.py
git restore --staged file.py

# Commit had typo in message
git commit --amend -m "Corrected message"

# Need to start over from last commit
git reset --hard HEAD

# See what would be deleted
git clean -n
# Actually delete untracked files
git clean -fd
```

## 📚 Next Steps

1. ✅ Repository is initialized
2. ⬜ Create remote repository (GitHub/GitLab)
3. ⬜ Push to remote
4. ⬜ Set up branch protection
5. ⬜ Configure CI/CD (tests run automatically)
6. ⬜ Add collaborators (if needed)

---

**Quick Help**: See `REPOSITORY_SETUP.md` for detailed instructions
