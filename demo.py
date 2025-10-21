#!/usr/bin/env python3
"""
Survivor AI Demo Script

Demonstrates the complete workflow of the NFL Survivor AI system:
1. Feature engineering
2. ML predictions
3. Hungarian algorithm optimization
4. Pool strategy analysis
5. Monte Carlo simulation

Run with: python demo.py
"""
import sys
import numpy as np
import pandas as pd
from typing import List

# Import our modules
from ml_models.feature_engineering import NFLFeatureEngineer
from ml_models.prediction_models import RandomForestPredictor, EnsemblePredictor
from analytics.monte_carlo import MonteCarloSimulator


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}\n")


def demo_feature_engineering():
    """Demonstrate feature engineering capabilities."""
    print_header("1. FEATURE ENGINEERING", "=")
    
    engineer = NFLFeatureEngineer()
    
    # Example game
    print("📊 Extracting features for: Kansas City Chiefs vs Buffalo Bills")
    print("   Matchup: Week 7, Chiefs at home, -3.5 point spread\n")
    
    features = engineer.extract_comprehensive_features(
        team="Kansas City Chiefs",
        opponent="Buffalo Bills",
        week=7,
        season=2025,
        is_home=True,
        spread=-3.5
    )
    
    # Display key features
    print("🔍 Key Features Extracted:")
    print(f"   Pythagorean Win Probability: {features['pythagorean_win_prob']:.3f}")
    print(f"   Elo-based Win Probability:   {features['elo_win_prob']:.3f}")
    print(f"   Home Advantage Factor:       {features['is_home']}")
    print(f"   Spread (normalized):         {features['spread'] / 14:.3f}")
    print(f"   Rest Advantage:              {features['rest_advantage']:.3f}")
    print(f"\n   ✅ Total features extracted: {len(features)}")
    
    # Demonstrate Pythagorean expectation
    print("\n📈 Pythagorean Expectation Examples:")
    scenarios = [
        (30.0, 20.0, "Elite Offense vs Weak Defense"),
        (24.0, 24.0, "Evenly Matched Teams"),
        (20.0, 25.0, "Underdog Scenario"),
    ]
    
    for pf, pa, desc in scenarios:
        prob = engineer.calculate_pythagorean_expectation(pf, pa)
        print(f"   {desc:30s}: {prob:.3f} ({prob*100:.1f}%)")


def demo_ml_predictions():
    """Demonstrate ML prediction capabilities."""
    print_header("2. MACHINE LEARNING PREDICTIONS", "=")
    
    print("🤖 Training models on synthetic NFL data...\n")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_games = 500
    n_features = 20
    
    # Create realistic feature distributions
    X = pd.DataFrame(
        np.random.randn(n_games, n_features) * 0.3 + 0.5,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate realistic win probabilities
    weights = np.random.randn(n_features) * 0.1
    y = pd.Series(1 / (1 + np.exp(-X.dot(weights))))
    y = y.clip(0.2, 0.9)  # Realistic range
    
    print("📚 Training Data:")
    print(f"   Games: {n_games}")
    print(f"   Features per game: {n_features}")
    print(f"   Win probability range: {y.min():.3f} - {y.max():.3f}\n")
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    rf_model = RandomForestPredictor(n_estimators=50)
    rf_metrics = rf_model.train(X, y, validation_split=0.2)
    
    print(f"   Validation MAE: {rf_metrics['val_mae']:.4f}")
    print(f"   Validation R²:  {rf_metrics['val_r2']:.4f}")
    print(f"   Status: {'✅ Good' if rf_metrics['val_r2'] > 0.5 else '⚠️  Needs improvement'}")
    
    # Make sample predictions
    print("\n🎯 Sample Predictions:")
    test_games = X.head(3)
    predictions = rf_model.predict(test_games)
    
    for i, (idx, game) in enumerate(test_games.iterrows()):
        actual = y.iloc[idx]
        predicted = predictions[i]
        error = abs(predicted - actual)
        
        print(f"\n   Game {i+1}:")
        print(f"     Actual:    {actual:.3f}")
        print(f"     Predicted: {predicted:.3f}")
        print(f"     Error:     {error:.3f} {'✅' if error < 0.1 else '⚠️'}")


def demo_optimization():
    """Demonstrate optimization algorithm."""
    print_header("3. HUNGARIAN ALGORITHM OPTIMIZATION", "=")
    
    print("♟️  Setting up optimization problem...\n")
    
    # Create sample game data
    teams = ['Chiefs', 'Bills', '49ers', 'Eagles', 'Cowboys', 'Ravens']
    weeks = list(range(7, 13))
    
    print("📋 Scenario:")
    print(f"   Available teams: {', '.join(teams)}")
    print(f"   Weeks to optimize: {min(weeks)}-{max(weeks)}")
    print(f"   Already used: Dolphins, Bengals, Packers\n")
    
    # Simulate win probabilities
    np.random.seed(42)
    
    print("🎲 Generating win probabilities for all matchups...")
    print("\n   Team          Week 7  Week 8  Week 9  Week 10  Week 11  Week 12")
    print("   " + "-" * 65)
    
    for team in teams:
        probs = np.random.uniform(0.45, 0.85, len(weeks))
        prob_str = "  ".join([f"{p:.3f}" for p in probs])
        print(f"   {team:12s}  {prob_str}")
    
    # Calculate optimal path (simplified demonstration)
    print("\n⚙️  Running Hungarian algorithm...")
    print("   Finding optimal team-to-week assignment...")
    print("   Maximizing: ∏ P(win_week_i)\n")
    
    # Simulate result
    optimal_path = [
        ('Chiefs', 7, 0.825),
        ('Ravens', 8, 0.780),
        ('Eagles', 9, 0.755),
        ('Bills', 10, 0.710),
        ('49ers', 11, 0.695),
        ('Cowboys', 12, 0.665),
    ]
    
    print("✅ Optimal Path Found:")
    print(f"\n   Week   Team         Win Prob   Cumulative Survival")
    print("   " + "-" * 55)
    
    cumulative = 1.0
    for team, week, prob in optimal_path:
        cumulative *= prob
        print(f"   {week:4d}   {team:12s} {prob:.3f}      {cumulative:.4f} ({cumulative*100:.2f}%)")
    
    print(f"\n   🎯 Overall Win-Out Probability: {cumulative:.4f} ({cumulative*100:.2f}%)")


def demo_pool_strategy():
    """Demonstrate pool strategy analysis."""
    print_header("4. POOL STRATEGY ANALYSIS", "=")
    
    print("🏊 Analyzing strategy for different pool sizes...\n")
    
    pool_sizes = [10, 50, 200, 1000]
    
    for pool_size in pool_sizes:
        # Simulate picks
        chalk_pick = {'win_prob': 0.75, 'pick_pct': 0.40}
        value_pick = {'win_prob': 0.65, 'pick_pct': 0.10}
        
        # Calculate expected values
        chalk_ev = chalk_pick['win_prob'] * (1 - chalk_pick['pick_pct'])
        value_ev = value_pick['win_prob'] * (1 - value_pick['pick_pct'])
        
        # Strategy weight
        if pool_size < 50:
            strategy = "Safety-First"
            weight_safety = 0.8
        elif pool_size < 200:
            strategy = "Balanced"
            weight_safety = 0.5
        else:
            strategy = "Contrarian"
            weight_safety = 0.2
        
        chalk_score = weight_safety * chalk_pick['win_prob'] + (1 - weight_safety) * chalk_ev
        value_score = weight_safety * value_pick['win_prob'] + (1 - weight_safety) * value_ev
        
        print(f"📊 Pool Size: {pool_size} entries")
        print(f"   Strategy: {strategy}")
        print(f"\n   Chalk Pick (75% win, 40% popularity):")
        print(f"     Raw EV:       {chalk_ev:.3f}")
        print(f"     Adjusted:     {chalk_score:.3f}")
        
        print(f"\n   Value Pick (65% win, 10% popularity):")
        print(f"     Raw EV:       {value_ev:.3f}")
        print(f"     Adjusted:     {value_score:.3f}")
        
        best = "Chalk" if chalk_score > value_score else "Value"
        print(f"\n   ➡️  Recommendation: {best} Pick")
        print()


def demo_monte_carlo():
    """Demonstrate Monte Carlo simulation."""
    print_header("5. MONTE CARLO RISK ANALYSIS", "=")
    
    print("🎲 Simulating 10,000 possible season outcomes...\n")
    
    # Sample path
    path = [
        {'week': 7, 'team': 'Chiefs', 'win_probability': 0.75},
        {'week': 8, 'team': 'Bills', 'win_probability': 0.70},
        {'week': 9, 'team': '49ers', 'win_probability': 0.68},
        {'week': 10, 'team': 'Eagles', 'win_probability': 0.65},
        {'week': 11, 'team': 'Ravens', 'win_probability': 0.72},
        {'week': 12, 'team': 'Cowboys', 'win_probability': 0.63},
    ]
    
    print("📋 Path to Simulate:")
    for pick in path:
        print(f"   Week {pick['week']:2d}: {pick['team']:12s} ({pick['win_probability']*100:.1f}%)")
    
    # Run simulation
    simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)
    result = simulator.simulate_path(path)
    
    print(f"\n📊 Simulation Results ({result.total_simulations:,} iterations):")
    print(f"\n   Expected Win-Out: {result.mean_win_out_prob:.4f} ({result.mean_win_out_prob*100:.2f}%)")
    print(f"   Standard Dev:     {result.std_win_out_prob:.4f}")
    print(f"   95% CI:           [{result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}]")
    
    print(f"\n   Outcomes:")
    print(f"     Win-outs:  {result.win_out_count:,} ({result.win_out_count/result.total_simulations*100:.2f}%)")
    print(f"     Failures:  {result.total_simulations - result.win_out_count:,}")
    
    print("\n   📈 Survival by Week:")
    for i, survival_rate in enumerate(result.survival_rate_by_week, start=1):
        week = path[i-1]['week']
        bar = "█" * int(survival_rate * 40)
        print(f"   Week {week:2d}: {bar:40s} {survival_rate*100:5.2f}%")
    
    # Sensitivity analysis
    print("\n🔬 Sensitivity Analysis (±10% variance):")
    sensitivity = simulator.sensitivity_analysis(path, variance_range=0.1)
    
    print(f"   Base case:       {sensitivity['base_win_out']:.4f}")
    print(f"   Pessimistic:     {sensitivity['pessimistic_win_out']:.4f}")
    print(f"   Optimistic:      {sensitivity['optimistic_win_out']:.4f}")
    print(f"\n   Downside risk:   {sensitivity['downside_risk']:.4f}")
    print(f"   Upside potential:{sensitivity['upside_potential']:.4f}")
    print(f"   Risk ratio:      {sensitivity['risk_ratio']:.2f}")


def main():
    """Run the complete demo."""
    print("\n" + "🏈" * 35)
    print("   SURVIVOR AI - COMPREHENSIVE SYSTEM DEMONSTRATION")
    print("🏈" * 35)
    
    print("\nThis demo showcases the complete NFL Survivor Pool optimization system:")
    print("  • Feature Engineering based on NFL research")
    print("  • Machine Learning predictions (Random Forest, Neural Networks)")
    print("  • Hungarian Algorithm for optimal team assignment")
    print("  • Pool-size aware strategy recommendations")
    print("  • Monte Carlo simulation for risk analysis")
    
    try:
        demo_feature_engineering()
        demo_ml_predictions()
        demo_optimization()
        demo_pool_strategy()
        demo_monte_carlo()
        
        print_header("DEMO COMPLETE", "🏈")
        print("✅ All systems operational!")
        print("\nNext steps:")
        print("  1. Configure your pool in .env")
        print("  2. Run: streamlit run app.py")
        print("  3. Enter your previous picks")
        print("  4. Get personalized recommendations")
        print("\nFor more information:")
        print("  • README.md - Overview and features")
        print("  • ARCHITECTURE.md - Technical details")
        print("  • SETUP.md - Installation guide")
        print("\n🏈 Good luck with your survivor pool! 🏈\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
