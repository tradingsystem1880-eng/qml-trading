# Dashboard

Launch the QML Trading Dashboard.

## Usage
```
/dashboard
```

## Instructions

When the user invokes this skill:

1. Check if a dashboard is already running on port 8501:
   ```bash
   lsof -ti:8501
   ```

2. If running, ask user if they want to restart it

3. Launch the JARVIS-style dashboard (app_v2.py is the recommended version):
   ```bash
   streamlit run qml/dashboard/app_v2.py --server.port 8501
   ```

4. Open in browser:
   ```bash
   open http://localhost:8501
   ```

5. Inform user the dashboard is running and how to stop it (Ctrl+C or `/stop-dashboard`)
