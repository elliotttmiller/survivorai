"""
Monte Carlo simulation for NFL Survivor Pool variance analysis.

Simulates thousands of possible season outcomes to:
- Estimate variance in win-out probabilities
- Calculate confidence intervals
- Identify risk factors
- Compare strategy robustness
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    mean_win_out_prob: float
    std_win_out_prob: float
    confidence_interval_95: Tuple[float, float]
    survival_rate_by_week: List[float]
    win_out_count: int
    total_simulations: int
    percentiles: Dict[int, float]


class MonteCarloSimulator:
    """Monte Carlo simulator for survivor pool analysis."""
    
    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_path(
        self, 
        path: List[Dict],
        n_simulations: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate outcomes for a given path.
        
        Args:
            path: List of picks with win probabilities
            n_simulations: Number of simulations (overrides default)
        
        Returns:
            SimulationResult with statistics
        """
        n_sims = n_simulations or self.n_simulations
        n_weeks = len(path)
        
        # Extract win probabilities
        win_probs = np.array([pick['win_probability'] for pick in path])
        
        # Run simulations
        # Each row is a simulation, each column is a week
        # Generate random values and compare to win probabilities
        random_vals = np.random.random((n_sims, n_weeks))
        outcomes = random_vals < win_probs  # True if team wins
        
        # Calculate survival per week
        # Cumulative product to see if survived all weeks up to that point
        survival_by_week = np.cumprod(outcomes, axis=1)
        
        # Win out = survived all weeks
        win_out = survival_by_week[:, -1]
        win_out_count = np.sum(win_out)
        win_out_prob = win_out_count / n_sims
        
        # Calculate statistics
        survival_rate_by_week = np.mean(survival_by_week, axis=0).tolist()
        
        # For confidence intervals, we need to model variance
        # Using binomial distribution assumption
        std_dev = np.sqrt(win_out_prob * (1 - win_out_prob) / n_sims)
        
        # 95% confidence interval
        z_score = 1.96  # For 95% confidence
        ci_lower = max(0, win_out_prob - z_score * std_dev)
        ci_upper = min(1, win_out_prob + z_score * std_dev)
        
        # Calculate percentiles
        # For each week, what's the probability of surviving to that week
        percentiles = {
            10: np.percentile(np.sum(survival_by_week, axis=1), 10),
            25: np.percentile(np.sum(survival_by_week, axis=1), 25),
            50: np.percentile(np.sum(survival_by_week, axis=1), 50),
            75: np.percentile(np.sum(survival_by_week, axis=1), 75),
            90: np.percentile(np.sum(survival_by_week, axis=1), 90),
        }
        
        return SimulationResult(
            mean_win_out_prob=win_out_prob,
            std_win_out_prob=std_dev,
            confidence_interval_95=(ci_lower, ci_upper),
            survival_rate_by_week=survival_rate_by_week,
            win_out_count=win_out_count,
            total_simulations=n_sims,
            percentiles=percentiles
        )
    
    def compare_paths(
        self,
        paths: List[List[Dict]],
        path_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple paths using Monte Carlo simulation.
        
        Args:
            paths: List of paths to compare
            path_names: Names for each path
        
        Returns:
            DataFrame comparing paths
        """
        if path_names is None:
            path_names = [f"Path {i+1}" for i in range(len(paths))]
        
        results = []
        
        for i, path in enumerate(paths):
            sim_result = self.simulate_path(path)
            
            results.append({
                'path_name': path_names[i],
                'expected_win_out': sim_result.mean_win_out_prob,
                'std_dev': sim_result.std_win_out_prob,
                'ci_lower_95': sim_result.confidence_interval_95[0],
                'ci_upper_95': sim_result.confidence_interval_95[1],
                'weeks_covered': len(path),
                'avg_win_prob': np.mean([p['win_probability'] for p in path]),
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('expected_win_out', ascending=False)
        
        return df
    
    def sensitivity_analysis(
        self,
        path: List[Dict],
        variance_range: float = 0.1
    ) -> Dict:
        """
        Perform sensitivity analysis by varying win probabilities.
        
        Tests how robust the path is to changes in win probability estimates.
        
        Args:
            path: Path to analyze
            variance_range: How much to vary probabilities (+/- this amount)
        
        Returns:
            Dictionary with sensitivity metrics
        """
        base_result = self.simulate_path(path)
        
        # Test pessimistic scenario (lower probabilities)
        pessimistic_path = [
            {**p, 'win_probability': max(0.01, p['win_probability'] - variance_range)}
            for p in path
        ]
        pessimistic_result = self.simulate_path(pessimistic_path)
        
        # Test optimistic scenario (higher probabilities)
        optimistic_path = [
            {**p, 'win_probability': min(0.99, p['win_probability'] + variance_range)}
            for p in path
        ]
        optimistic_result = self.simulate_path(optimistic_path)
        
        return {
            'base_win_out': base_result.mean_win_out_prob,
            'pessimistic_win_out': pessimistic_result.mean_win_out_prob,
            'optimistic_win_out': optimistic_result.mean_win_out_prob,
            'downside_risk': base_result.mean_win_out_prob - pessimistic_result.mean_win_out_prob,
            'upside_potential': optimistic_result.mean_win_out_prob - base_result.mean_win_out_prob,
            'risk_ratio': (base_result.mean_win_out_prob - pessimistic_result.mean_win_out_prob) / 
                         (optimistic_result.mean_win_out_prob - base_result.mean_win_out_prob)
                         if optimistic_result.mean_win_out_prob > base_result.mean_win_out_prob else float('inf')
        }
    
    def calculate_sharpe_ratio(
        self,
        path: List[Dict],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe-like ratio for the path.
        
        Measures risk-adjusted returns.
        
        Args:
            path: Path to analyze
            risk_free_rate: Baseline probability (usually 0)
        
        Returns:
            Risk-adjusted score
        """
        result = self.simulate_path(path)
        
        expected_return = result.mean_win_out_prob - risk_free_rate
        std_dev = result.std_win_out_prob
        
        if std_dev == 0:
            return float('inf') if expected_return > 0 else 0
        
        sharpe = expected_return / std_dev
        return sharpe
    
    def simulate_pool_dynamics(
        self,
        path: List[Dict],
        initial_pool_size: int,
        pick_distributions: Optional[List[Dict[str, float]]] = None
    ) -> Dict:
        """
        Simulate pool dynamics week by week.
        
        Estimates how many people survive each week based on
        pick distributions and game outcomes.
        
        Args:
            path: Your path
            initial_pool_size: Starting pool size
            pick_distributions: Distribution of picks for each week
        
        Returns:
            Dictionary with pool dynamics
        """
        # Simplified version - assumes uniform pick distribution if not provided
        n_sims = self.n_simulations
        n_weeks = len(path)
        
        pool_sizes = np.zeros((n_sims, n_weeks + 1))
        pool_sizes[:, 0] = initial_pool_size
        
        for week_idx in range(n_weeks):
            pick = path[week_idx]
            win_prob = pick['win_probability']
            
            # Simulate outcome
            outcomes = np.random.random(n_sims) < win_prob
            
            # Estimate elimination rate (simplified)
            # Assume average survival rate of 0.6 for the pool
            avg_survival = 0.6
            pool_sizes[:, week_idx + 1] = pool_sizes[:, week_idx] * avg_survival
            
            # Your survival
            pool_sizes[~outcomes, week_idx + 1] = 0  # You're out if you lose
        
        final_pool_sizes = pool_sizes[:, -1]
        
        return {
            'avg_final_pool_size': np.mean(final_pool_sizes[final_pool_sizes > 0]),
            'median_final_pool_size': np.median(final_pool_sizes[final_pool_sizes > 0]),
            'win_probability': np.mean(final_pool_sizes > 0),
            'expected_payout_rank': np.mean(1 / final_pool_sizes[final_pool_sizes > 0])
        }


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    print("Testing Monte Carlo Simulation")
    print("=" * 60)
    
    simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)
    
    # Test path
    test_path = [
        {'week': 7, 'team': 'Team A', 'win_probability': 0.75},
        {'week': 8, 'team': 'Team B', 'win_probability': 0.70},
        {'week': 9, 'team': 'Team C', 'win_probability': 0.65},
        {'week': 10, 'team': 'Team D', 'win_probability': 0.60},
        {'week': 11, 'team': 'Team E', 'win_probability': 0.70},
    ]
    
    print("\n1. Simulating path...")
    result = simulator.simulate_path(test_path)
    print(f"   Win-out probability: {result.mean_win_out_prob:.4f}")
    print(f"   Standard deviation: {result.std_win_out_prob:.4f}")
    print(f"   95% CI: [{result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}]")
    print(f"   Survival rates by week: {[f'{s:.3f}' for s in result.survival_rate_by_week]}")
    
    print("\n2. Sensitivity analysis...")
    sensitivity = simulator.sensitivity_analysis(test_path, variance_range=0.1)
    print(f"   Base case: {sensitivity['base_win_out']:.4f}")
    print(f"   Pessimistic: {sensitivity['pessimistic_win_out']:.4f}")
    print(f"   Optimistic: {sensitivity['optimistic_win_out']:.4f}")
    print(f"   Downside risk: {sensitivity['downside_risk']:.4f}")
    print(f"   Risk ratio: {sensitivity['risk_ratio']:.2f}")
    
    print("\n3. Sharpe ratio...")
    sharpe = simulator.calculate_sharpe_ratio(test_path)
    print(f"   Risk-adjusted score: {sharpe:.2f}")
    
    print("\n✓ Monte Carlo simulation test complete")


if __name__ == "__main__":
    test_monte_carlo()
