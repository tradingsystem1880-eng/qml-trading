#!/usr/bin/env python3
"""
Forensic Audit Script - State of the Union Report
Lead Systems Architect Analysis Tool
"""

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
ROOT_DIR = Path("/Users/hunternovotny/Desktop/QML_SYSTEM")
IGNORE_DIRS = {'.venv', 'venv', 'node_modules', '.git', '.pytest_cache', 
               '__pycache__', '.ipynb_checkpoints', '.obsidian', '.cursor', '.jupyter'}

# File categories
CATEGORIES = {
    'Python Scripts': ['.py'],
    'Jupyter Notebooks': ['.ipynb'],
    'Data Files': ['.csv', '.parquet', '.pkl', '.pickle', '.h5', '.hdf5'],
    'Config Files': ['.json', '.yaml', '.yml', '.toml', '.env', '.ini'],
    'Documentation': ['.md', '.txt', '.rst'],
    'Other/Junk': ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.html']
}

def categorize_file(ext):
    """Categorize file by extension."""
    for category, extensions in CATEGORIES.items():
        if ext.lower() in extensions:
            return category
    return 'Uncategorized'

def build_tree(root_path, prefix="", ignore_dirs=IGNORE_DIRS):
    """Build a visual tree structure."""
    lines = []
    path = Path(root_path)
    
    # Get all items, filter ignored
    items = sorted([p for p in path.iterdir() 
                   if p.name not in ignore_dirs and not p.name.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        
        if item.is_dir():
            lines.append(f"{prefix}{connector}📁 {item.name}/")
            extension = "    " if is_last else "│   "
            lines.extend(build_tree(item, prefix + extension, ignore_dirs))
        else:
            # Add icon based on type
            ext = item.suffix.lower()
            if ext == '.py':
                icon = "🐍"
            elif ext == '.ipynb':
                icon = "📓"
            elif ext in ['.csv', '.parquet', '.pkl']:
                icon = "📊"
            elif ext in ['.md', '.txt']:
                icon = "📝"
            elif ext in ['.json', '.yaml', '.yml', '.toml']:
                icon = "⚙️"
            elif ext in ['.png', '.jpg', '.pdf']:
                icon = "🖼️"
            else:
                icon = "📄"
            lines.append(f"{prefix}{connector}{icon} {item.name}")
    
    return lines

def scan_files(root_path):
    """Scan all files and categorize them."""
    inventory = defaultdict(list)
    all_files = []
    
    for root, dirs, files in os.walk(root_path):
        # Filter ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in files:
            if file.startswith('.'):
                continue
            filepath = Path(root) / file
            ext = filepath.suffix.lower()
            category = categorize_file(ext)
            rel_path = filepath.relative_to(root_path)
            
            file_info = {
                'path': str(rel_path),
                'abs_path': str(filepath),
                'name': file,
                'ext': ext,
                'size': filepath.stat().st_size if filepath.exists() else 0
            }
            inventory[category].append(file_info)
            all_files.append(file_info)
    
    return inventory, all_files

def extract_imports(content):
    """Extract import statements from Python code."""
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except:
        # Fallback regex
        import_pattern = r'^(?:from\s+(\S+)|import\s+(\S+))'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imports.append(match.group(1) or match.group(2))
    return imports

def find_parameter_names(content):
    """Find variable names that look like parameters."""
    param_patterns = [
        r'(stop_loss|take_profit|window|threshold|period|lookback|atr|rsi|ema|sma|ma_period)',
        r'([A-Z_]+_(?:WINDOW|PERIOD|THRESHOLD|RATIO|SIZE|LIMIT))',
        r'(\w+_(?:size|window|period|threshold|limit|ratio))\s*=',
    ]
    params = set()
    for pattern in param_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            params.add(match.group(1))
    return list(params)

def find_main_indicators(content):
    """Find if this looks like a main entry point."""
    main_indicators = {
        'if __name__': 'Has __main__ block',
        'def main(': 'Has main() function',
        'backtest': 'Contains backtest logic',
        'run_backtest': 'Contains run_backtest function',
        'execute_trades': 'Contains trade execution',
        'argparse': 'Uses argparse (CLI entry)',
    }
    found = []
    for pattern, desc in main_indicators.items():
        if pattern.lower() in content.lower():
            found.append(desc)
    return found

def find_file_operations(content):
    """Find file save/load operations."""
    operations = []
    patterns = {
        r'\.to_csv\(': 'CSV Export',
        r'\.to_parquet\(': 'Parquet Export',
        r'pickle\.dump': 'Pickle Save',
        r'json\.dump': 'JSON Save',
        r'\.read_csv\(': 'CSV Import',
        r'\.read_parquet\(': 'Parquet Import',
        r'pickle\.load': 'Pickle Load',
        r'json\.load': 'JSON Load',
        r'open\([^)]+["\']w["\']': 'File Write',
        r'savefig\(': 'Figure Save',
    }
    for pattern, desc in patterns.items():
        if re.search(pattern, content):
            operations.append(desc)
    return operations

def analyze_python_file(filepath):
    """Analyze a Python file for structure and purpose."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return None
    
    analysis = {
        'imports': extract_imports(content),
        'parameters': find_parameter_names(content),
        'main_indicators': find_main_indicators(content),
        'file_operations': find_file_operations(content),
        'line_count': len(content.split('\n')),
        'has_classes': bool(re.search(r'^class\s+\w+', content, re.MULTILINE)),
        'function_count': len(re.findall(r'^def\s+\w+', content, re.MULTILINE)),
    }
    
    # Try to extract function names
    try:
        tree = ast.parse(content)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        analysis['functions'] = functions[:20]  # Limit
        analysis['classes'] = classes
    except:
        analysis['functions'] = []
        analysis['classes'] = []
    
    return analysis

def analyze_notebook(filepath):
    """Analyze a Jupyter notebook."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except:
        return None
    
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    
    # Combine all code
    all_code = '\n'.join([''.join(c.get('source', [])) for c in code_cells])
    
    return {
        'total_cells': len(cells),
        'code_cells': len(code_cells),
        'markdown_cells': len(markdown_cells),
        'imports': extract_imports(all_code),
        'parameters': find_parameter_names(all_code),
        'file_operations': find_file_operations(all_code),
    }

def find_data_files(inventory):
    """Identify data file locations and types."""
    data_info = []
    for file_info in inventory.get('Data Files', []):
        path = file_info['path']
        size_mb = file_info['size'] / (1024 * 1024)
        data_info.append({
            'path': path,
            'size_mb': round(size_mb, 2),
            'type': file_info['ext']
        })
    return data_info

def identify_architecture(py_analyses):
    """Identify code architecture patterns."""
    patterns = {
        'detection': [],
        'execution': [],
        'backtest': [],
        'utility': [],
        'config': [],
        'visualization': [],
        'validation': [],
    }
    
    for filepath, analysis in py_analyses.items():
        if not analysis:
            continue
        
        name_lower = Path(filepath).name.lower()
        functions_str = ' '.join(analysis.get('functions', []))
        
        if any(x in name_lower for x in ['detect', 'pattern', 'signal']):
            patterns['detection'].append(filepath)
        elif any(x in name_lower for x in ['execute', 'trade', 'order']):
            patterns['execution'].append(filepath)
        elif any(x in name_lower for x in ['backtest', 'run_', 'engine']):
            patterns['backtest'].append(filepath)
        elif any(x in name_lower for x in ['util', 'helper', 'common']):
            patterns['utility'].append(filepath)
        elif any(x in name_lower for x in ['config', 'setup', 'settings']):
            patterns['config'].append(filepath)
        elif any(x in name_lower for x in ['visual', 'plot', 'chart', 'report']):
            patterns['visualization'].append(filepath)
        elif any(x in name_lower for x in ['valid', 'test', 'verify']):
            patterns['validation'].append(filepath)
    
    return patterns

def generate_report():
    """Generate the complete forensic audit report."""
    print("=" * 80)
    print("🔬 FORENSIC AUDIT: STATE OF THE UNION REPORT")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Root: {ROOT_DIR}")
    print("=" * 80)
    
    # 1. Tree View
    print("\n" + "═" * 80)
    print("📁 DIRECTORY TREE STRUCTURE")
    print("═" * 80)
    tree_lines = build_tree(ROOT_DIR)
    for line in tree_lines[:100]:  # Limit output
        print(line)
    if len(tree_lines) > 100:
        print(f"... and {len(tree_lines) - 100} more items")
    
    # 2. File Inventory
    print("\n" + "═" * 80)
    print("📊 FILE INVENTORY BY CATEGORY")
    print("═" * 80)
    inventory, all_files = scan_files(ROOT_DIR)
    
    total_files = 0
    for category in sorted(inventory.keys()):
        files = inventory[category]
        count = len(files)
        total_files += count
        total_size = sum(f['size'] for f in files) / 1024  # KB
        print(f"\n🏷️  {category}: {count} files ({total_size:.1f} KB total)")
        
        # Show top files by size
        for f in sorted(files, key=lambda x: -x['size'])[:5]:
            size_str = f"{f['size']/1024:.1f}KB" if f['size'] > 1024 else f"{f['size']}B"
            print(f"    → {f['path']} ({size_str})")
    
    print(f"\n📈 TOTAL: {total_files} files scanned")
    
    # 3. Python Analysis
    print("\n" + "═" * 80)
    print("🐍 PYTHON CODE ANALYSIS")
    print("═" * 80)
    
    py_analyses = {}
    for f in inventory.get('Python Scripts', []):
        analysis = analyze_python_file(f['abs_path'])
        if analysis:
            py_analyses[f['path']] = analysis
    
    # Find main entry points
    print("\n🚀 POTENTIAL MAIN ENTRY POINTS:")
    main_files = []
    for path, analysis in py_analyses.items():
        if analysis['main_indicators']:
            main_files.append((path, analysis['main_indicators'], analysis['line_count']))
    
    for path, indicators, lines in sorted(main_files, key=lambda x: -x[2]):
        print(f"  📍 {path} ({lines} lines)")
        for ind in indicators:
            print(f"       • {ind}")
    
    # Architecture patterns
    print("\n🏗️  ARCHITECTURE CLASSIFICATION:")
    patterns = identify_architecture(py_analyses)
    for pattern_type, files in patterns.items():
        if files:
            print(f"\n  [{pattern_type.upper()}]")
            for f in files[:5]:
                print(f"    → {f}")
    
    # Large files (potential spaghetti)
    print("\n⚠️  LARGEST FILES (Potential Spaghetti):")
    large_files = [(p, a['line_count'], a['function_count']) 
                   for p, a in py_analyses.items() if a['line_count'] > 200]
    for path, lines, funcs in sorted(large_files, key=lambda x: -x[1])[:10]:
        status = "🍝 SPAGHETTI RISK" if lines > 500 and funcs > 15 else ""
        print(f"  {path}: {lines} lines, {funcs} functions {status}")
    
    # 4. Jupyter Notebook Analysis
    print("\n" + "═" * 80)
    print("📓 JUPYTER NOTEBOOK ANALYSIS")
    print("═" * 80)
    
    for f in inventory.get('Jupyter Notebooks', []):
        nb_analysis = analyze_notebook(f['abs_path'])
        if nb_analysis:
            print(f"\n📓 {f['path']}")
            print(f"   Cells: {nb_analysis['total_cells']} total ({nb_analysis['code_cells']} code, {nb_analysis['markdown_cells']} markdown)")
            if nb_analysis['file_operations']:
                print(f"   File Ops: {', '.join(nb_analysis['file_operations'])}")
    
    # 5. Data State
    print("\n" + "═" * 80)
    print("📊 DATA & ARTIFACT LOCATIONS")
    print("═" * 80)
    
    data_files = find_data_files(inventory)
    if data_files:
        print("\n📂 DATA FILES FOUND:")
        for d in sorted(data_files, key=lambda x: -x['size_mb']):
            print(f"  📊 {d['path']} ({d['size_mb']} MB) [{d['type']}]")
        
        # Identify BTC data
        btc_files = [d for d in data_files if 'btc' in d['path'].lower() or 'bitcoin' in d['path'].lower()]
        if btc_files:
            print("\n🪙 BTC DATA IDENTIFIED:")
            for btc in btc_files:
                print(f"  ✅ {btc['path']}")
        else:
            print("\n⚠️  NO EXPLICIT BTC DATA FILE FOUND - may be named differently")
    else:
        print("❌ NO DATA FILES FOUND IN REPOSITORY")
    
    # Results/output directories
    print("\n📁 OUTPUT/RESULTS DIRECTORIES:")
    for path, _, files in os.walk(ROOT_DIR):
        rel = Path(path).relative_to(ROOT_DIR)
        if any(x in str(rel).lower() for x in ['result', 'output', 'artifact', 'log']):
            file_count = len([f for f in files if not f.startswith('.')])
            if file_count > 0:
                print(f"  📂 {rel}/ ({file_count} files)")
    
    # 6. Logic Map / Dependencies
    print("\n" + "═" * 80)
    print("🔗 LOGIC MAP & DEPENDENCIES")
    print("═" * 80)
    
    # Find internal imports
    internal_modules = set()
    for f in py_analyses:
        internal_modules.add(Path(f).stem)
    
    print("\n📦 INTERNAL MODULE DEPENDENCIES:")
    for path, analysis in list(py_analyses.items())[:20]:
        internal_imports = [i for i in analysis['imports'] 
                          if any(i.startswith(m) for m in ['src', 'qml', 'scripts', 'experiments'])]
        if internal_imports:
            print(f"\n  {path}:")
            for imp in internal_imports[:10]:
                print(f"    → imports {imp}")
    
    # 7. Parameter Discovery
    print("\n" + "═" * 80)
    print("🎛️  DISCOVERED PARAMETERS")
    print("═" * 80)
    
    all_params = set()
    for analysis in py_analyses.values():
        all_params.update(analysis.get('parameters', []))
    
    if all_params:
        print("\nPotential tunable parameters found:")
        for param in sorted(all_params):
            print(f"  • {param}")
    
    # 8. Summary Assessment
    print("\n" + "═" * 80)
    print("📋 EXECUTIVE SUMMARY")
    print("═" * 80)
    
    print(f"""
    📁 Structure:     {len([d for d in ROOT_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])} top-level directories
    🐍 Python Files:  {len(inventory.get('Python Scripts', []))} scripts
    📓 Notebooks:     {len(inventory.get('Jupyter Notebooks', []))} notebooks
    📊 Data Files:    {len(inventory.get('Data Files', []))} data files
    📝 Documentation: {len(inventory.get('Documentation', []))} docs
    
    🏗️  Architecture Assessment:
    """)
    
    # Determine if modular or spaghetti
    if len(patterns['detection']) > 0 and len(patterns['execution']) > 0:
        print("    ✅ SEPARATION EXISTS: Detection and Execution logic are in different files")
    else:
        print("    ⚠️  MONOLITHIC RISK: Detection and Execution may be intermingled")
    
    if any(a['line_count'] > 800 for a in py_analyses.values()):
        print("    🍝 SPAGHETTI DETECTED: Files with 800+ lines exist")
    
    src_files = len([f for f in py_analyses if f.startswith('src/')])
    root_files = len([f for f in py_analyses if '/' not in f])
    if root_files > 5:
        print(f"    ⚠️  ROOT CLUTTER: {root_files} Python files in root directory")
    
    print("\n" + "=" * 80)
    print("🔬 FORENSIC AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    generate_report()
