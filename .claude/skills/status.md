# Status

Show current system status and recent results.

## Usage
```
/status
```

## Instructions

When the user invokes this skill:

1. Check git status for uncommitted changes

2. Query recent experiments from `results/experiments.db`:
   ```bash
   sqlite3 results/experiments.db "SELECT * FROM experiments ORDER BY created_at DESC LIMIT 5"
   ```

3. Check for running processes:
   - Dashboard (port 8501)
   - Any background optimizations

4. Report:
   - **Git Status**: Clean or pending changes
   - **Recent Experiments**: Last 5 with key metrics
   - **Active Processes**: What's running
   - **Current Phase**: Reference CLAUDE.md for project phase
   - **Recommendations**: What to do next based on project state

5. Reference the current priority from CLAUDE.md
