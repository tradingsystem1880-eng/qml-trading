# Commit

Create a git commit with proper message format.

## Usage
```
/commit [message]
```

## Examples
- `/commit` - Auto-generate commit message from changes
- `/commit "fix: resolve import error in pattern_scorer"` - Use provided message

## Instructions

When the user invokes this skill:

1. Run `git status` and `git diff` to understand changes

2. If no message provided, analyze changes and generate one:
   - Use conventional commit format: `type: description`
   - Types: feat, fix, refactor, docs, test, chore
   - Keep description concise (50 chars or less)

3. Stage appropriate files:
   - Stage modified files related to the change
   - DO NOT stage: .env, credentials, large data files, __pycache__

4. Create commit:
   ```bash
   git commit -m "$(cat <<'EOF'
   {message}

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   EOF
   )"
   ```

5. Show commit hash and summary

6. Ask if user wants to push (don't push automatically)
