# Review

Review code for issues, improvements, and best practices.

## Usage
```
/review [file_or_directory]
```

## Examples
- `/review src/detection/pattern_scorer.py` - Review specific file
- `/review src/ml/` - Review entire ML module
- `/review` - Review recent changes (git diff)

## Instructions

When the user invokes this skill:

1. If no path provided, review `git diff HEAD` (recent changes)

2. Read the code and analyze for:

   **Bugs & Logic Errors:**
   - Off-by-one errors
   - Incorrect conditionals
   - Missing null/None checks
   - Data type mismatches

   **Security Issues:**
   - Hardcoded credentials
   - SQL injection (if applicable)
   - Unsafe data handling

   **Trading-Specific Issues:**
   - Lookahead bias (using future data)
   - Train/test leakage
   - Survivorship bias
   - Incorrect P&L calculations

   **Code Quality:**
   - Unclear variable names
   - Missing error handling
   - Overly complex logic
   - Duplicate code

3. Report findings with:
   - Severity: HIGH / MEDIUM / LOW
   - Line number and code snippet
   - Explanation of the issue
   - Suggested fix

4. Offer to implement fixes if user approves
