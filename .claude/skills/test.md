# Test

Run tests for the QML trading system.

## Usage
```
/test [path]
```

## Examples
- `/test` - Run all tests
- `/test tests/test_detection.py` - Run specific test file
- `/test tests/test_validation.py::test_monte_carlo` - Run specific test

## Instructions

When the user invokes this skill:

1. Activate virtual environment if exists

2. Run pytest:
   ```bash
   python -m pytest {path or "tests/"} -v --tb=short
   ```

3. Report:
   - Tests passed/failed/skipped
   - Any failures with brief explanation
   - Coverage if available

4. If tests fail, offer to investigate the failure
