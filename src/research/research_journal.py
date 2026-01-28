"""
Research Journal for QML Trading System
========================================
Tracks experiments and prevents re-testing failed ideas.

Based on DeepSeek analysis: With PF 4.49, adding features is more likely to hurt than help.
This journal enforces disciplined experimentation.

Usage:
    from src.research.research_journal import ResearchJournal

    journal = ResearchJournal()

    # Check if feature was already tested
    if journal.check_if_tested('funding_rate_filter'):
        print("Already tested!")

    # Log experiment results
    journal.log_experiment({
        'hypothesis': 'Funding rate filter will improve PF by filtering against-crowd trades',
        'feature_name': 'funding_rate_filter',
        'methodology': 'Walk-forward + permutation test',
        'results': {'pf_change': -0.15, 'wr_change': -0.02},
        'conclusion': 'FAIL - PF degradation',
        'notes': 'Filter rejected 30% of trades',
    })
"""

import json
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ExperimentResult:
    """Single experiment record."""
    id: str
    timestamp: str
    hypothesis: str
    feature_name: str
    methodology: str
    results: Dict[str, Any]
    conclusion: str  # 'PASS', 'FAIL', 'INCONCLUSIVE'
    notes: str
    verdict: str = ""  # 'DEPLOY', 'REJECT', 'NEEDS_MORE_DATA'
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentResult':
        """Create from dictionary."""
        return cls(
            id=d.get('id', ''),
            timestamp=d.get('timestamp', ''),
            hypothesis=d.get('hypothesis', ''),
            feature_name=d.get('feature_name', ''),
            methodology=d.get('methodology', ''),
            results=d.get('results', {}),
            conclusion=d.get('conclusion', ''),
            notes=d.get('notes', ''),
            verdict=d.get('verdict', ''),
            tags=d.get('tags', []),
        )


class ResearchJournal:
    """
    Tracks experiments and prevents re-testing failed ideas.

    Key principles:
    1. Every experiment must have a clear hypothesis
    2. Failed experiments are documented to prevent retry
    3. All experiments follow same validation methodology
    4. One feature at a time (prevent confounding)
    """

    DEFAULT_JOURNAL_PATH = PROJECT_ROOT / "research" / "journal.json"

    def __init__(self, journal_path: Optional[Path] = None):
        """
        Initialize research journal.

        Args:
            journal_path: Path to journal JSON file
        """
        self.journal_path = journal_path or self.DEFAULT_JOURNAL_PATH
        self.entries: List[ExperimentResult] = []
        self._load()

    def _load(self):
        """Load journal from disk."""
        if self.journal_path.exists():
            with open(self.journal_path, 'r') as f:
                data = json.load(f)
                self.entries = [ExperimentResult.from_dict(e) for e in data.get('experiments', [])]
        else:
            self.entries = []

    def _save(self):
        """Save journal to disk."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_experiments': len(self.entries),
            'experiments': [asdict(e) for e in self.entries],
        }
        with open(self.journal_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, feature_name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{feature_name}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{feature_name}_{timestamp}_{short_hash}"

    def get_git_commit_hash(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            result = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                cwd=PROJECT_ROOT
            )
            return result.decode().strip()[:8]
        except Exception:
            return 'unknown'

    def get_file_hash(self, filepath: str) -> str:
        """Get MD5 hash of data file for reproducibility."""
        try:
            path = Path(filepath)
            if path.is_dir():
                # Hash directory by hashing all file names and sizes
                content = ""
                for f in sorted(path.glob("**/*")):
                    if f.is_file():
                        content += f"{f.name}:{f.stat().st_size};"
                return hashlib.md5(content.encode()).hexdigest()[:8]
            else:
                with open(filepath, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()[:8]
        except Exception:
            return 'unknown'

    def log_experiment(self, experiment: Dict, data_files: List[str] = None) -> Dict:
        """
        Log a new experiment to the journal with automatic metadata.

        Required fields:
            - hypothesis: Clear statement of what you're testing
            - feature_name: Unique identifier for the feature
            - methodology: How the test was conducted
            - results: Dict of numeric results
            - conclusion: 'PASS', 'FAIL', or 'INCONCLUSIVE'
            - notes: Additional observations

        Args:
            experiment: Dict with experiment details
            data_files: Optional list of file paths to hash for reproducibility

        Returns:
            Dict with experiment record including generated ID, git commit, and data hashes
        """
        required = ['hypothesis', 'feature_name', 'methodology', 'results', 'conclusion', 'notes']
        missing = [f for f in required if f not in experiment]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Validate conclusion
        valid_conclusions = ['PASS', 'FAIL', 'INCONCLUSIVE']
        if experiment['conclusion'] not in valid_conclusions:
            raise ValueError(f"conclusion must be one of: {valid_conclusions}")

        # Determine verdict based on conclusion
        conclusion = experiment['conclusion']
        if conclusion == 'PASS':
            verdict = 'DEPLOY'
        elif conclusion == 'FAIL':
            verdict = 'REJECT'
        else:
            verdict = 'NEEDS_MORE_DATA'

        # Create record
        record = ExperimentResult(
            id=self._generate_id(experiment['feature_name']),
            timestamp=datetime.now().isoformat(),
            hypothesis=experiment['hypothesis'],
            feature_name=experiment['feature_name'],
            methodology=experiment['methodology'],
            results=experiment['results'],
            conclusion=conclusion,
            notes=experiment['notes'],
            verdict=verdict,
            tags=experiment.get('tags', []),
        )

        # Convert to dict and add metadata
        record_dict = asdict(record)
        record_dict['git_commit'] = self.get_git_commit_hash()

        # Hash data files if provided
        record_dict['data_hashes'] = {}
        if data_files:
            for fp in data_files:
                record_dict['data_hashes'][Path(fp).name] = self.get_file_hash(fp)

        self.entries.append(record)
        self._save()

        return record_dict

    def check_if_tested(self, feature_name: str) -> bool:
        """
        Check if a feature has already been tested.

        Args:
            feature_name: Feature identifier

        Returns:
            True if feature was tested (PASS or FAIL), False otherwise
        """
        for entry in self.entries:
            if entry.feature_name == feature_name and entry.conclusion in ['PASS', 'FAIL']:
                return True
        return False

    def get_experiment_history(self, feature_name: str) -> List[ExperimentResult]:
        """Get all experiments for a feature."""
        return [e for e in self.entries if e.feature_name == feature_name]

    def get_failed_experiments(self) -> List[ExperimentResult]:
        """Get all failed experiments."""
        return [e for e in self.entries if e.conclusion == 'FAIL']

    def get_passed_experiments(self) -> List[ExperimentResult]:
        """Get all passed experiments."""
        return [e for e in self.entries if e.conclusion == 'PASS']

    def get_pending_experiments(self) -> List[ExperimentResult]:
        """Get experiments marked as INCONCLUSIVE."""
        return [e for e in self.entries if e.conclusion == 'INCONCLUSIVE']

    def get_experiment_by_id(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get specific experiment by ID."""
        for entry in self.entries:
            if entry.id == experiment_id:
                return entry
        return None

    def generate_summary(self) -> str:
        """Generate human-readable summary of journal."""
        lines = []
        lines.append("=" * 70)
        lines.append("RESEARCH JOURNAL SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Total Experiments: {len(self.entries)}")

        passed = len(self.get_passed_experiments())
        failed = len(self.get_failed_experiments())
        pending = len(self.get_pending_experiments())

        lines.append(f"  PASS: {passed}")
        lines.append(f"  FAIL: {failed}")
        lines.append(f"  INCONCLUSIVE: {pending}")
        lines.append("")

        # Recent experiments
        if self.entries:
            lines.append("-" * 70)
            lines.append("RECENT EXPERIMENTS (last 5)")
            lines.append("-" * 70)
            for entry in self.entries[-5:]:
                status_emoji = "✅" if entry.conclusion == 'PASS' else "❌" if entry.conclusion == 'FAIL' else "⏳"
                lines.append(f"{status_emoji} [{entry.timestamp[:10]}] {entry.feature_name}: {entry.conclusion}")
                lines.append(f"   Hypothesis: {entry.hypothesis[:60]}...")

        # Failed features (DO NOT RETRY)
        failed_features = list(set(e.feature_name for e in self.get_failed_experiments()))
        if failed_features:
            lines.append("")
            lines.append("-" * 70)
            lines.append("DO NOT RETRY (Failed Features)")
            lines.append("-" * 70)
            for feat in failed_features:
                lines.append(f"  - {feat}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def should_proceed(self, feature_name: str) -> tuple[bool, str]:
        """
        Check if we should proceed with testing a feature.

        Returns:
            Tuple of (should_proceed, reason)
        """
        history = self.get_experiment_history(feature_name)

        if not history:
            return True, "Feature not tested before - proceed"

        # Check if already passed
        passed = [e for e in history if e.conclusion == 'PASS']
        if passed:
            return False, f"Feature already PASSED on {passed[-1].timestamp[:10]} - already deployed"

        # Check if already failed
        failed = [e for e in history if e.conclusion == 'FAIL']
        if failed:
            last_fail = failed[-1]
            return False, f"Feature already FAILED on {last_fail.timestamp[:10]}: {last_fail.notes[:100]}"

        # Inconclusive - can retry
        return True, "Previous tests inconclusive - retry with more data"
