"""
Example: Using Injury Analysis in Survivor AI

This example demonstrates how to use the injury analysis feature
to enhance prediction accuracy by factoring in key player injuries.
"""
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.injury_reports import (
    InjuryReportCollector,
    InjuryImpactAnalyzer,
    calculate_injury_adjusted_win_probability,
    enrich_game_data_with_injuries
)
from ml_models.feature_engineering import NFLFeatureEngineer


def example_1_basic_injury_analysis():
    """Example 1: Basic injury impact analysis."""
    print("="*70)
    print("Example 1: Basic Injury Impact Analysis")
    print("="*70)
    
    # Create analyzer
    analyzer = InjuryImpactAnalyzer()
    
    # Simulate some injuries
    injuries = [
        {'player_name': 'Patrick Mahomes', 'position': 'QB', 'status': 'OUT'},
        {'player_name': 'Travis Kelce', 'position': 'TE', 'status': 'QUESTIONABLE'},
    ]
    
    # Calculate impact
    impact = analyzer.calculate_team_injury_impact(injuries)
    critical = analyzer.get_critical_injuries(injuries)
    
    print(f"\nTeam Injuries:")
    for inj in injuries:
        print(f"  - {inj['player_name']} ({inj['position']}): {inj['status']}")
    
    print(f"\nOverall Team Impact Score: {impact:.2%}")
    print(f"Critical Injuries: {len(critical)}")
    
    for inj in critical:
        print(f"  - {inj['player_name']}: Impact = {inj['impact_score']:.3f}")
    
    print("\nInterpretation:")
    if impact > 0.5:
        print("  ⚠️  SEVERE IMPACT - Team significantly weakened")
    elif impact > 0.3:
        print("  ⚡ MODERATE IMPACT - Notable effect on performance")
    else:
        print("  ℹ️  MINOR IMPACT - Limited effect")


def example_2_win_probability_adjustment():
    """Example 2: Adjusting win probability based on injuries."""
    print("\n" + "="*70)
    print("Example 2: Win Probability Adjustment")
    print("="*70)
    
    analyzer = InjuryImpactAnalyzer()
    
    # Scenario: Our QB is out, opponent is healthy
    our_injuries = [
        {'player_name': 'QB1', 'position': 'QB', 'status': 'OUT'}
    ]
    opponent_injuries = []
    
    our_impact = analyzer.calculate_team_injury_impact(our_injuries)
    opponent_impact = analyzer.calculate_team_injury_impact(opponent_injuries)
    
    # Calculate adjusted win probability
    base_win_prob = 0.65
    adjusted_win_prob = calculate_injury_adjusted_win_probability(
        base_win_prob, our_impact, opponent_impact
    )
    
    change = adjusted_win_prob - base_win_prob
    
    print(f"\nScenario: Our starting QB is OUT")
    print(f"Base Win Probability: {base_win_prob:.1%}")
    print(f"Our Injury Impact: {our_impact:.2%}")
    print(f"Opponent Injury Impact: {opponent_impact:.2%}")
    print(f"\nAdjusted Win Probability: {adjusted_win_prob:.1%}")
    print(f"Change: {change:+.1%}")
    
    if change < 0:
        print(f"\n⚠️  Our injuries reduce our win probability by {abs(change):.1%}")


def example_3_game_data_enrichment():
    """Example 3: Enriching game data with injury analysis."""
    print("\n" + "="*70)
    print("Example 3: Game Data Enrichment")
    print("="*70)
    
    # Initialize components
    collector = InjuryReportCollector()
    analyzer = InjuryImpactAnalyzer()
    
    # Create sample game data
    games_df = pd.DataFrame({
        'team': ['Kansas City Chiefs', 'Buffalo Bills', 'San Francisco 49ers'],
        'opponent': ['Buffalo Bills', 'Miami Dolphins', 'Dallas Cowboys'],
        'week': [7, 7, 7],
        'win_probability': [0.65, 0.72, 0.58]
    })
    
    print("\nOriginal Game Data:")
    print(games_df[['team', 'opponent', 'win_probability']].to_string(index=False))
    
    # Enrich with injury data
    enriched_df = enrich_game_data_with_injuries(games_df, collector, analyzer)
    
    print("\n\nEnriched with Injury Analysis:")
    cols_to_show = [
        'team', 'opponent', 'win_probability', 
        'team_injury_impact', 'opponent_injury_impact',
        'injury_adjusted_win_probability'
    ]
    print(enriched_df[cols_to_show].to_string(index=False))
    
    # Show changes
    print("\n\nInjury Impact Summary:")
    for idx, row in enriched_df.iterrows():
        team = row['team']
        base = row['win_probability']
        adjusted = row['injury_adjusted_win_probability']
        change = adjusted - base
        
        print(f"\n{team}:")
        print(f"  Base: {base:.1%} → Adjusted: {adjusted:.1%} ({change:+.1%})")


def example_4_ml_integration():
    """Example 4: ML feature engineering with injury analysis."""
    print("\n" + "="*70)
    print("Example 4: ML Feature Engineering Integration")
    print("="*70)
    
    # Create feature engineer with injury analysis enabled
    engineer = NFLFeatureEngineer(use_injury_data=True)
    
    # Extract features for a matchup
    features = engineer.extract_comprehensive_features(
        team='Kansas City Chiefs',
        opponent='Buffalo Bills',
        week=7,
        season=2025,
        is_home=True,
        spread=-3.5
    )
    
    print(f"\nTotal Features Extracted: {len(features)}")
    
    # Show injury-specific features
    injury_features = {k: v for k, v in features.items() if 'injury' in k.lower()}
    
    print("\nInjury-Related Features:")
    for feat_name, feat_value in injury_features.items():
        print(f"  {feat_name}: {feat_value}")
    
    print("\nThese features are automatically included in ML model predictions!")


def example_5_position_weights():
    """Example 5: Understanding position impact weights."""
    print("\n" + "="*70)
    print("Example 5: Position Impact Weights")
    print("="*70)
    
    from data_collection.injury_reports import POSITION_IMPACT_WEIGHTS
    
    print("\nPosition Impact Weights (Higher = More Important):")
    print("\n  Position | Weight | Description")
    print("  " + "-"*60)
    
    # Sort by weight (descending)
    sorted_positions = sorted(
        POSITION_IMPACT_WEIGHTS.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    descriptions = {
        'QB': 'Quarterback - touches ball every play',
        'OL': 'Offensive Line - protects QB, enables run game',
        'RB': 'Running Back - key offensive weapon',
        'DL': 'Defensive Line - pass rush and run defense',
        'WR': 'Wide Receiver - important but replaceable',
        'LB': 'Linebacker - defensive versatility',
        'DB': 'Defensive Back - coverage responsibilities',
        'TE': 'Tight End - dual threat but limited snaps',
        'K': 'Kicker - minimal overall impact',
        'P': 'Punter - very limited impact',
    }
    
    for pos, weight in sorted_positions:
        desc = descriptions.get(pos, 'NFL player')
        print(f"  {pos:8s} | {weight:.2f}  | {desc}")
    
    print("\nExample Impact Calculations:")
    
    scenarios = [
        ('QB OUT', [{'position': 'QB', 'status': 'OUT'}]),
        ('WR OUT + RB QUESTIONABLE', [
            {'position': 'WR', 'status': 'OUT'},
            {'position': 'RB', 'status': 'QUESTIONABLE'}
        ]),
        ('K OUT', [{'position': 'K', 'status': 'OUT'}])
    ]
    
    analyzer = InjuryImpactAnalyzer()
    
    for scenario_name, injuries in scenarios:
        impact = analyzer.calculate_team_injury_impact(injuries)
        print(f"\n  {scenario_name}: {impact:.1%} team impact")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("INJURY ANALYSIS EXAMPLES - Survivor AI v3.0")
    print("="*70)
    
    # Run all examples
    example_1_basic_injury_analysis()
    example_2_win_probability_adjustment()
    example_3_game_data_enrichment()
    example_4_ml_integration()
    example_5_position_weights()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
    print("\nFor more details, see: INJURY_ANALYSIS.md")
    print("For API reference, see: data_collection/injury_reports.py")
    print("\n")


if __name__ == "__main__":
    main()
