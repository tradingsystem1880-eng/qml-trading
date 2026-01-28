# API Docs Skill

Auto-generate documentation from Python docstrings.

## When to Use
- Documenting module APIs
- Generating reference documentation
- Creating function/class documentation
- Building project documentation

## Docstring Standards

### Google Style (Recommended)

```python
def detect_pattern(
    df: pd.DataFrame,
    config: DetectionConfig,
    min_quality: float = 0.5
) -> list[Signal]:
    """
    Detect QML patterns in OHLCV data.

    Scans the provided price data for Quasimodo (QML) reversal patterns
    using hierarchical swing detection and quality scoring.

    Args:
        df: OHLCV DataFrame with datetime index. Must contain columns:
            open, high, low, close, volume.
        config: Detection configuration object containing parameters
            for swing detection and pattern validation.
        min_quality: Minimum quality score (0-1) for pattern to be
            included in results. Defaults to 0.5.

    Returns:
        List of Signal objects representing detected patterns. Each signal
        contains entry/SL/TP levels and quality metadata.

    Raises:
        ValueError: If df is empty or missing required columns.
        DetectionError: If pattern detection fails.

    Example:
        >>> df = pd.read_parquet("data/BTCUSDT_4h.parquet")
        >>> config = DetectionConfig(swing_lookback=5)
        >>> signals = detect_pattern(df, config)
        >>> print(f"Found {len(signals)} patterns")
        Found 12 patterns

    Note:
        This function is computationally intensive. For multi-symbol
        detection, use parallel_detect() instead.

    See Also:
        - parallel_detect: Parallel detection across symbols
        - Signal: Signal dataclass definition
        - DetectionConfig: Configuration options
    """
    pass
```

### Class Documentation

```python
class HierarchicalSwingDetector:
    """
    Three-layer hierarchical swing point detector.

    Implements a sophisticated swing detection algorithm that operates
    across three layers:

    1. **Geometry Layer**: Identifies raw swing points based on
       local high/low comparisons.
    2. **Significance Layer**: Filters noise by requiring minimum
       price movement between swings.
    3. **Context Layer**: Validates swings against trend context
       to ensure meaningful reversals.

    Attributes:
        config (SwingConfig): Configuration parameters.
        geometry_detector (GeometryDetector): First layer detector.
        significance_filter (SignificanceFilter): Second layer filter.
        context_validator (ContextValidator): Third layer validator.

    Example:
        >>> detector = HierarchicalSwingDetector(SwingConfig(lookback=5))
        >>> swings = detector.detect(df)
        >>> for swing in swings:
        ...     print(f"{swing.type} at {swing.time}: {swing.price}")
        HIGH at 2024-01-15 08:00:00: 42500.0
        LOW at 2024-01-15 16:00:00: 41200.0

    Note:
        The detector requires at least 100 bars of data for reliable
        results due to the lookback requirements of all three layers.
    """

    def __init__(self, config: SwingConfig):
        """
        Initialize the hierarchical detector.

        Args:
            config: Swing detection configuration. See SwingConfig
                for available parameters.
        """
        self.config = config
```

## Generating Documentation

### Sphinx Setup

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Initialize in docs/
cd docs
sphinx-quickstart

# Generate API docs
sphinx-apidoc -o source/ ../src/
```

### conf.py Configuration

```python
# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'QML Trading System'
author = 'Hunter Novotny'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

html_theme = 'sphinx_rtd_theme'
```

### Build Documentation

```bash
# Build HTML
cd docs
make html

# View locally
open _build/html/index.html
```

## pdoc (Lightweight Alternative)

```bash
# Install
pip install pdoc

# Generate HTML docs
pdoc --html --output-dir docs/api src/

# Serve locally with auto-reload
pdoc --http : src/

# Generate single module
pdoc src/detection/pattern_scorer.py
```

## mkdocs with mkdocstrings

```yaml
# mkdocs.yml
site_name: QML Trading System
theme:
  name: material

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true

nav:
  - Home: index.md
  - API Reference:
    - Detection: api/detection.md
    - Simulation: api/simulation.md
    - Risk: api/risk.md
```

```markdown
<!-- docs/api/detection.md -->
# Detection Module

::: src.detection.pattern_scorer
    options:
      show_source: true
      members:
        - PatternScorer
        - score
```

```bash
# Build
mkdocs build

# Serve with live reload
mkdocs serve
```

## Quick API Reference Generator

```python
#!/usr/bin/env python
"""Generate quick API reference markdown."""

import ast
import inspect
from pathlib import Path

def extract_docstrings(filepath: Path) -> dict:
    """Extract docstrings from Python file."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    docs = {'module': ast.get_docstring(tree), 'classes': {}, 'functions': {}}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            docs['classes'][node.name] = {
                'docstring': ast.get_docstring(node),
                'methods': {}
            }
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    docs['classes'][node.name]['methods'][item.name] = ast.get_docstring(item)

        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            docs['functions'][node.name] = ast.get_docstring(node)

    return docs

def generate_markdown(filepath: Path) -> str:
    """Generate markdown documentation."""
    docs = extract_docstrings(filepath)
    lines = [f"# {filepath.stem}\n"]

    if docs['module']:
        lines.append(docs['module'])
        lines.append("")

    for class_name, class_info in docs['classes'].items():
        lines.append(f"## class `{class_name}`\n")
        if class_info['docstring']:
            lines.append(class_info['docstring'])
            lines.append("")

        for method_name, method_doc in class_info['methods'].items():
            if not method_name.startswith('_') or method_name == '__init__':
                lines.append(f"### `{method_name}()`\n")
                if method_doc:
                    lines.append(method_doc)
                lines.append("")

    for func_name, func_doc in docs['functions'].items():
        if not func_name.startswith('_'):
            lines.append(f"## `{func_name}()`\n")
            if func_doc:
                lines.append(func_doc)
            lines.append("")

    return '\n'.join(lines)

if __name__ == "__main__":
    import sys
    filepath = Path(sys.argv[1])
    print(generate_markdown(filepath))
```

### Usage

```bash
# Generate docs for single file
python scripts/generate_api_docs.py src/detection/pattern_scorer.py > docs/api/pattern_scorer.md

# Generate for all modules
for f in src/**/*.py; do
    python scripts/generate_api_docs.py "$f" > "docs/api/$(basename ${f%.py}).md"
done
```

## Type Hints for Better Docs

```python
from typing import Optional, Union, Literal
from dataclasses import dataclass

@dataclass
class Signal:
    """
    Trading signal with entry/exit levels.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        direction: Trade direction, either "LONG" or "SHORT"
        entry_price: Suggested entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        validity_score: Pattern quality score (0.0 to 1.0)
        pattern_type: Type of pattern detected
    """
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    stop_loss: float
    take_profit: float
    validity_score: float
    pattern_type: str
    metadata: Optional[dict] = None
```

## Documentation Best Practices

1. **Document public APIs**: All public functions/classes need docstrings
2. **Include examples**: Show typical usage in docstring
3. **Type hints + docstrings**: Use both for best IDE support
4. **Keep updated**: Treat docs as code - update with changes
5. **Link related items**: Use "See Also" to connect related functions
