# Explain

Explain a concept, file, or component of the QML trading system.

## Usage
```
/explain [topic]
```

## Examples
- `/explain QML pattern` - What is a QML pattern?
- `/explain detection logic` - How does pattern detection work?
- `/explain src/detection/pattern_scorer.py` - Explain this file
- `/explain validation pipeline` - How does validation work?
- `/explain Kelly criterion` - Explain position sizing math

## Instructions

When the user invokes this skill:

1. Identify the topic type:
   - **Concept**: QML patterns, backtesting, validation, etc.
   - **File**: Read and explain the code
   - **Component**: Detection, scoring, optimization, etc.

2. For concepts:
   - Explain in beginner-friendly terms (user is learning Python)
   - Use analogies where helpful
   - Reference relevant files in the codebase

3. For files:
   - Read the file
   - Explain purpose, key functions, and how it fits in the system
   - Highlight important logic

4. For components:
   - Show the data flow
   - List key files involved
   - Explain configuration options

5. Always offer to dive deeper into specific parts
