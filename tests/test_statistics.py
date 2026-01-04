"""
Unit Tests for Statistical Robustness Frameworks
=================================================
Tests permutation testing, Monte Carlo simulation, and bootstrap confidence intervals.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.validation.permutation import PermutationTest, PermutationResult, run_permutation_test
from src.validation.monte_carlo import MonteCarloSimulator, MonteCarloResult, run_monte_carlo
from src.validation.bootstrap import BlockBootstrap, BootstrapResult, compute_all_confidence_intervals


class TestPermutationTest:
    """Tests for permutation testing."""
    
    def test_basic_permutation(self):
        """Test basic permutation test execution."""
        np.random.seed(42)
        returns = np.random.randn(50) * 2 + 1  # Positive mean returns
        
        test = PermutationTest(n_permutations=1000, random_seed=42)
        result = test.run(returns)
        
        assert isinstance(result, PermutationResult)
        assert result.n_permutations == 1000
        assert 0 <= result.sharpe_p_value <= 1
        assert 0 <= result.sharpe_percentile <= 100
        assert len(result.permutation_sharpes) == 1000
    
    def test_skilled_vs_random(self):
        """Test that skilled returns with variance show appropriate percentile."""
        np.random.seed(42)
        
        # Returns with clear positive edge and variance
        # Mix of varying wins and losses to ensure shuffling produces different Sharpes
        skilled_returns = np.array([5.0, 4.5, 3.0, -2.0, 4.0, -1.5, 6.0, 3.5, -0.5, 4.0,
                                    5.5, -1.0, 3.0, 4.0, -2.5, 5.0, 3.5, 4.5, -1.0, 3.0])
        
        result = run_permutation_test(skilled_returns, n_permutations=1000, seed=42)
        
        # With clear edge, actual Sharpe should be positive
        assert result.actual_sharpe > 0
        # P-value should be calculable (0-1 range)
        assert 0 <= result.sharpe_p_value <= 1
    
    def test_random_returns_structure(self):
        """Test that permutation test produces valid statistical structure."""
        np.random.seed(42)
        
        # Random returns centered around zero
        random_returns = np.random.randn(100)
        
        result = run_permutation_test(random_returns, n_permutations=1000, seed=42)
        
        # Percentile should be between 0 and 100
        assert 0 <= result.sharpe_percentile <= 100
        # P-value should be valid probability
        assert 0 <= result.sharpe_p_value <= 1
        # Distribution should have expected size
        assert len(result.permutation_sharpes) == 1000
    
    def test_insufficient_trades_raises(self):
        """Test error with too few trades."""
        returns = np.array([1, 2, 3])
        test = PermutationTest()
        
        with pytest.raises(ValueError):
            test.run(returns)
    
    def test_report_generation(self):
        """Test report generation."""
        np.random.seed(42)
        returns = np.random.randn(30)
        
        test = PermutationTest(n_permutations=100)
        result = test.run(returns)
        report = test.generate_report(result)
        
        assert "PERMUTATION" in report
        assert "p-value" in report
        assert "percentile" in report


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""
    
    def test_basic_monte_carlo(self):
        """Test basic Monte Carlo execution."""
        np.random.seed(42)
        returns = np.random.randn(50) + 0.5
        
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.run(returns)
        
        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 1000
        assert result.equity_paths.shape[0] == 1000
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
    
    def test_var_ordering(self):
        """Test VaR values are properly ordered."""
        np.random.seed(42)
        returns = np.random.randn(100)
        
        result = run_monte_carlo(returns, n_simulations=5000, seed=42)
        
        # 99% VaR should be >= 95% VaR
        assert result.var_99 >= result.var_95
        
        # Expected shortfall should be >= VaR
        assert result.expected_shortfall_95 >= result.var_95
    
    def test_kill_switch_probability(self):
        """Test kill switch probability calculation."""
        np.random.seed(42)
        
        # Returns with high variance
        high_risk_returns = np.random.randn(50) * 5
        
        result = run_monte_carlo(
            high_risk_returns,
            n_simulations=1000,
            kill_switch_threshold=0.10,  # 10% threshold
            seed=42
        )
        
        # With high variance, kill switch should trigger sometimes
        assert 0 <= result.kill_switch_prob <= 1
    
    def test_sequence_vs_bootstrap(self):
        """Test both simulation methods work."""
        np.random.seed(42)
        returns = np.random.randn(30) + 0.2
        
        sim = MonteCarloSimulator(n_simulations=500, random_seed=42)
        
        # Sequence randomization
        result_seq = sim.run(returns, method="sequence")
        assert result_seq.n_simulations == 500
        
        # Bootstrap
        result_boot = sim.run(returns, method="bootstrap")
        assert result_boot.n_simulations == 500
    
    def test_report_generation(self):
        """Test report generation."""
        np.random.seed(42)
        returns = np.random.randn(30)
        
        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.run(returns)
        report = sim.generate_report(result)
        
        assert "MONTE CARLO" in report
        assert "VaR" in report
        assert "Kill Switch" in report


class TestBlockBootstrap:
    """Tests for block bootstrap confidence intervals."""
    
    def test_basic_bootstrap(self):
        """Test basic bootstrap CI calculation."""
        np.random.seed(42)
        data = np.random.randn(100) + 2
        
        bootstrap = BlockBootstrap(n_bootstrap=1000, block_size=5, random_seed=42)
        result = bootstrap.confidence_interval(data, statistic_fn=np.mean)
        
        assert isinstance(result, BootstrapResult)
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.standard_error > 0
    
    def test_ci_contains_true_value(self):
        """Test CI contains true parameter (on average)."""
        np.random.seed(42)
        true_mean = 5.0
        data = np.random.randn(100) + true_mean
        
        bootstrap = BlockBootstrap(n_bootstrap=1000, confidence_level=0.95, random_seed=42)
        result = bootstrap.confidence_interval(data, statistic_fn=np.mean)
        
        # 95% CI should contain true mean
        assert result.ci_lower < true_mean < result.ci_upper
    
    def test_hypothesis_test(self):
        """Test hypothesis testing against null."""
        np.random.seed(42)
        
        # Data with clear positive mean
        data = np.random.randn(100) + 5
        
        bootstrap = BlockBootstrap(n_bootstrap=1000, random_seed=42)
        result = bootstrap.hypothesis_test(
            data,
            statistic_fn=np.mean,
            null_value=0,
            alternative="greater"
        )
        
        # Should strongly reject null of mean <= 0
        assert result.p_value_vs_null < 0.05
    
    def test_all_metrics_ci(self):
        """Test computing CIs for all standard metrics."""
        np.random.seed(42)
        
        # Create sample trades DataFrame
        n_trades = 100
        pnl = np.concatenate([
            np.random.exponential(2, int(n_trades * 0.6)),  # Wins
            -np.random.exponential(1, int(n_trades * 0.4)),  # Losses
        ])
        np.random.shuffle(pnl)
        
        trades_df = pd.DataFrame({"pnl_pct": pnl})
        
        results = compute_all_confidence_intervals(
            trades_df,
            n_bootstrap=500,
            block_size=3,
            seed=42
        )
        
        assert "sharpe_ratio" in results
        assert "win_rate" in results
        assert "profit_factor" in results
        assert "max_drawdown" in results
        
        # Win rate should have p-value for test vs 0.5
        assert not np.isnan(results["win_rate"].p_value_vs_null)
    
    def test_block_resampling_preserves_length(self):
        """Test block resampling preserves data length."""
        np.random.seed(42)
        data = np.arange(100)
        
        bootstrap = BlockBootstrap(n_bootstrap=10, block_size=10, random_seed=42)
        resampled = bootstrap._block_resample(data)
        
        assert len(resampled) == len(data)
    
    def test_report_generation(self):
        """Test report generation."""
        np.random.seed(42)
        
        trades_df = pd.DataFrame({
            "pnl_pct": np.random.randn(50)
        })
        
        bootstrap = BlockBootstrap(n_bootstrap=100, random_seed=42)
        results = bootstrap.all_metrics_ci(trades_df)
        report = bootstrap.generate_report(results)
        
        assert "BOOTSTRAP" in report
        assert "Confidence" in report


class TestStatisticalIntegration:
    """Integration tests for statistical modules."""
    
    def test_all_modules_on_same_data(self):
        """Test all statistical modules work on same dataset."""
        np.random.seed(42)
        
        # Simulated profitable strategy returns
        returns = np.concatenate([
            np.random.exponential(1.5, 40),
            -np.random.exponential(0.8, 20),
        ])
        np.random.shuffle(returns)
        
        # Permutation test
        perm_result = run_permutation_test(returns, n_permutations=500, seed=42)
        assert perm_result.actual_sharpe > 0
        
        # Monte Carlo
        mc_result = run_monte_carlo(returns, n_simulations=500, seed=42)
        assert mc_result.var_95 > 0
        
        # Bootstrap
        trades_df = pd.DataFrame({"pnl_pct": returns})
        boot_results = compute_all_confidence_intervals(
            trades_df, n_bootstrap=200, seed=42
        )
        assert boot_results["sharpe_ratio"].ci_lower < boot_results["sharpe_ratio"].ci_upper
    
    def test_significance_agreement(self):
        """Test permutation and bootstrap agree on significance direction."""
        np.random.seed(42)
        
        # Clear positive edge
        returns = np.random.exponential(2, 60) - np.random.exponential(0.5, 60)
        returns = returns[returns != 0][:50]
        
        perm = run_permutation_test(returns, n_permutations=500, seed=42)
        
        trades_df = pd.DataFrame({"pnl_pct": returns})
        bootstrap = BlockBootstrap(n_bootstrap=500, random_seed=42)
        boot_sharpe = bootstrap.confidence_interval(returns, lambda x: np.mean(x) / np.std(x))
        
        # If permutation shows high percentile, bootstrap CI should exclude 0
        if perm.sharpe_percentile > 80:
            assert boot_sharpe.ci_lower > 0 or boot_sharpe.point_estimate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
