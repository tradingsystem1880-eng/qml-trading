# Debug

Debug an issue in the QML trading system.

## Usage
```
/debug [description]
```

## Examples
- `/debug backtest returns no trades`
- `/debug dashboard not loading`
- `/debug pattern detection missing obvious patterns`
- `/debug import error in src/detection`

## Instructions

When the user invokes this skill:

1. Understand the issue from the description

2. Systematic debugging approach:

   **For import/module errors:**
   - Check PYTHONPATH
   - Verify virtual environment
   - Check for circular imports

   **For no trades/patterns:**
   - Verify data exists and has correct format
   - Check detection thresholds in config/default.yaml
   - Run with verbose logging

   **For dashboard issues:**
   - Check if port is in use
   - Look for Streamlit errors
   - Verify all dependencies installed

   **For unexpected results:**
   - Check for train/test leakage (Phase 8.0 lesson)
   - Verify data timestamps
   - Look for lookahead bias

3. Run diagnostic commands as needed

4. Explain findings in beginner-friendly terms

5. Propose fix and ask before implementing
