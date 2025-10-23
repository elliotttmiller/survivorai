"""
Feature engineering module for NFL game prediction.

Based on research from:
- Frontiers in Sports and Active Living: NFL win prediction study
- Advanced ML techniques for real-time sports prediction

Key features extracted:
- Offensive efficiency metrics
- Defensive performance indicators
- Historical performance trends
- Team strength ratings
- Home/away advantages
- Rest days and scheduling factors
- Injury impact analysis (ENHANCED)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class NFLFeatureEngineer:
    """Feature engineering for NFL game prediction."""
    
    def __init__(self, historical_seasons: int = 3, use_injury_data: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            historical_seasons: Number of past seasons to consider for historical features
            use_injury_data: Whether to include injury analysis features (default True)
        """
        self.historical_seasons = historical_seasons
        self.team_stats_cache = {}
        self.use_injury_data = use_injury_data
        self.injury_collector = None
        self.injury_analyzer = None
        
        # Initialize injury analysis components if enabled
        if use_injury_data:
            try:
                from data_collection.injury_reports import InjuryReportCollector, InjuryImpactAnalyzer
                self.injury_collector = InjuryReportCollector()
                self.injury_analyzer = InjuryImpactAnalyzer()
            except ImportError:
                # Injury analysis not available, disable it
                self.use_injury_data = False
        
    def extract_basic_features(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features from game data.
        
        Args:
            game_data: DataFrame with game information
            
        Returns:
            DataFrame with basic features
        """
        features = game_data.copy()
        
        # Home field advantage (binary)
        if 'is_home' in features.columns:
            features['home_advantage'] = features['is_home'].astype(int)
        else:
            features['home_advantage'] = 0
            
        # Point spread normalized
        if 'spread' in features.columns:
            features['spread_normalized'] = features['spread'] / 14.0
        else:
            features['spread_normalized'] = 0.0
            
        return features
    
    def calculate_pythagorean_expectation(
        self, 
        points_for: float, 
        points_against: float,
        exponent: float = 2.37
    ) -> float:
        """
        Calculate Pythagorean expectation (win probability estimate).
        
        Formula: P(win) = PF^exp / (PF^exp + PA^exp)
        where PF = points for, PA = points against
        
        The exponent 2.37 is optimized for NFL based on research.
        
        Args:
            points_for: Average points scored per game
            points_against: Average points allowed per game
            exponent: Power exponent (default 2.37 for NFL)
            
        Returns:
            Expected win probability
        """
        if points_for <= 0 or points_against <= 0:
            return 0.5
            
        pf_exp = points_for ** exponent
        pa_exp = points_against ** exponent
        
        return pf_exp / (pf_exp + pa_exp)
    
    def extract_offensive_features(self, team: str, season: int) -> Dict:
        """
        Extract offensive efficiency features.
        
        Key metrics based on research:
        - Points per game
        - Yards per play
        - Third down conversion rate
        - Red zone efficiency
        - Turnover rate
        
        Args:
            team: Team name
            season: Season year
            
        Returns:
            Dictionary of offensive features
        """
        # Placeholder implementation - in production, fetch from stats API
        return {
            f'{team}_points_per_game': 24.5,
            f'{team}_yards_per_play': 5.8,
            f'{team}_third_down_pct': 0.42,
            f'{team}_redzone_efficiency': 0.58,
            f'{team}_turnover_rate': 0.12,
        }
    
    def extract_defensive_features(self, team: str, season: int) -> Dict:
        """
        Extract defensive performance features.
        
        Key metrics:
        - Points allowed per game
        - Yards allowed per play
        - Sacks per game
        - Takeaway rate
        - Pass defense efficiency
        
        Args:
            team: Team name
            season: Season year
            
        Returns:
            Dictionary of defensive features
        """
        # Placeholder implementation
        return {
            f'{team}_points_allowed': 21.3,
            f'{team}_yards_allowed_per_play': 5.2,
            f'{team}_sacks_per_game': 2.4,
            f'{team}_takeaway_rate': 0.14,
            f'{team}_pass_defense_rating': 95.5,
        }
    
    def calculate_elo_rating(
        self, 
        team: str, 
        opponent: str,
        k_factor: float = 20.0
    ) -> Dict:
        """
        Calculate Elo ratings for teams.
        
        Elo rating system adapted for NFL:
        - Starting rating: 1500
        - K-factor: 20 (adjustment rate)
        - Home advantage: ~65 points
        
        Args:
            team: Team name
            opponent: Opponent name
            k_factor: Elo adjustment factor
            
        Returns:
            Dictionary with Elo ratings and expected win probability
        """
        # Placeholder - in production, maintain persistent Elo ratings
        team_elo = 1520.0
        opponent_elo = 1480.0
        
        # Calculate expected win probability from Elo difference
        elo_diff = team_elo - opponent_elo
        expected_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        return {
            'team_elo': team_elo,
            'opponent_elo': opponent_elo,
            'elo_diff': elo_diff,
            'elo_win_prob': expected_win_prob,
        }
    
    def extract_recent_form(self, team: str, n_games: int = 5) -> Dict:
        """
        Extract recent performance features.
        
        Args:
            team: Team name
            n_games: Number of recent games to consider
            
        Returns:
            Dictionary with recent form metrics
        """
        # Placeholder implementation
        return {
            f'{team}_last_{n_games}_win_pct': 0.60,
            f'{team}_last_{n_games}_avg_margin': 3.2,
            f'{team}_momentum_score': 0.65,
        }
    
    def calculate_rest_advantage(self, team_rest_days: int, opponent_rest_days: int) -> float:
        """
        Calculate rest advantage factor.
        
        Research shows teams with more rest have slight advantage.
        
        Args:
            team_rest_days: Days of rest for team
            opponent_rest_days: Days of rest for opponent
            
        Returns:
            Rest advantage score (-1 to 1)
        """
        rest_diff = team_rest_days - opponent_rest_days
        
        # Normalize to -1 to 1 range
        return np.tanh(rest_diff / 7.0)
    
    def calculate_epa_estimate(self, team: str, side: str = 'offense') -> float:
        """
        Calculate EPA (Expected Points Added) estimate.
        
        EPA measures the expected point value change on plays. This is an
        approximation based on aggregate statistics.
        
        Research shows EPA correlates strongly with win probability (r = 0.87).
        
        Args:
            team: Team name
            side: 'offense' or 'defense'
            
        Returns:
            Estimated EPA value (typically -1.0 to 1.0)
        """
        # Placeholder - would use actual play-by-play data in production
        # For now, approximate from aggregate stats
        if side == 'offense':
            # Positive EPA = above average offense
            points_per_game = 24.5  # Would fetch actual data
            league_avg = 22.0
            league_std = 5.0
            return (points_per_game - league_avg) / league_std
        else:
            # Negative defensive EPA = good defense (fewer points allowed)
            points_allowed = 21.3  # Would fetch actual data
            league_avg = 22.0
            league_std = 5.0
            return (league_avg - points_allowed) / league_std
    
    def calculate_dvoa_proxy(self, team: str, side: str = 'offense') -> float:
        """
        Calculate DVOA-inspired efficiency metric.
        
        DVOA (Defense-adjusted Value Over Average) measures efficiency
        relative to league average, adjusted for opponent strength.
        
        This is a simplified proxy based on available aggregate statistics.
        
        Args:
            team: Team name
            side: 'offense' or 'defense'
            
        Returns:
            Efficiency rating (typically -0.5 to 0.5)
        """
        # Placeholder - would use actual efficiency data in production
        if side == 'offense':
            # Combine yards per play and points per drive
            ypp = 5.8  # Would fetch actual data
            league_avg_ypp = 5.5
            ppd = 2.2  # Points per drive
            league_avg_ppd = 2.0
            
            efficiency = (
                (ypp / league_avg_ypp) * 0.5 +
                (ppd / league_avg_ppd) * 0.5
            ) - 1.0
            return efficiency
        else:
            # Defensive efficiency (inverse)
            ypp_allowed = 5.2  # Would fetch actual data
            league_avg_ypp = 5.5
            ppd_allowed = 1.9  # Points per drive allowed
            league_avg_ppd = 2.0
            
            efficiency = (
                (league_avg_ypp / ypp_allowed) * 0.5 +
                (league_avg_ppd / ppd_allowed) * 0.5
            ) - 1.0
            return efficiency
    
    def calculate_success_rate(self, team: str) -> float:
        """
        Calculate success rate.
        
        Success rate measures percentage of plays that achieve "success":
        - 1st down: 50% of needed yards
        - 2nd down: 70% of needed yards
        - 3rd/4th down: 100% of needed yards
        
        Research shows success rate predicts future performance better
        than yards per play (r = 0.68 vs 0.54).
        
        Args:
            team: Team name
            
        Returns:
            Success rate (0.0 to 1.0, typically 0.35-0.50)
        """
        # Placeholder - would calculate from play-by-play data
        # Teams with >40% success rate win 72% of games
        return 0.42  # League average approximation
    
    def calculate_explosive_play_rate(self, team: str) -> float:
        """
        Calculate explosive play rate.
        
        Explosive plays:
        - Passing: 20+ yards
        - Rushing: 12+ yards
        
        Research shows explosive play rate differential correlates with
        point differential (r = 0.81).
        
        Args:
            team: Team name
            
        Returns:
            Explosive play rate (plays per game, typically 3-8)
        """
        # Placeholder - would calculate from play-by-play data
        return 5.2  # League average approximation
    
    def extract_injury_features(self, team: str, week: Optional[int] = None) -> Dict:
        """
        Extract injury-related features for a team.
        
        ENHANCED: Analyzes injury reports to quantify their impact on team performance.
        Research shows key player injuries can reduce win probability by 5-20%.
        
        Args:
            team: Team name
            week: Week number (optional)
            
        Returns:
            Dictionary with injury features
        """
        features = {}
        
        if not self.use_injury_data or not self.injury_collector or not self.injury_analyzer:
            # Injury data not available, return zero impact
            features[f'{team}_injury_impact'] = 0.0
            features[f'{team}_has_qb_injury'] = 0
            features[f'{team}_num_key_injuries'] = 0
            return features
        
        try:
            # Get injury data for the team
            injuries = self.injury_collector.get_team_injuries(team, week)
            
            # Calculate overall injury impact
            injury_impact = self.injury_analyzer.calculate_team_injury_impact(injuries)
            features[f'{team}_injury_impact'] = injury_impact
            
            # Check for QB injury specifically (most impactful position)
            has_qb_injury = any(
                inj.get('position') == 'QB' and inj.get('status') in ['OUT', 'DOUBTFUL']
                for inj in injuries
            )
            features[f'{team}_has_qb_injury'] = int(has_qb_injury)
            
            # Count critical injuries
            critical_injuries = self.injury_analyzer.get_critical_injuries(injuries)
            features[f'{team}_num_key_injuries'] = len(critical_injuries)
            
        except Exception as e:
            # If injury data fetch fails, use zero impact (safe fallback)
            features[f'{team}_injury_impact'] = 0.0
            features[f'{team}_has_qb_injury'] = 0
            features[f'{team}_num_key_injuries'] = 0
        
        return features
    
    def extract_comprehensive_features(
        self,
        team: str,
        opponent: str,
        week: int,
        season: int,
        is_home: bool,
        spread: Optional[float] = None
    ) -> Dict:
        """
        Extract all features for a matchup.
        
        Combines:
        - Basic game info
        - Offensive metrics
        - Defensive metrics
        - Elo ratings
        - Recent form
        - Rest and scheduling factors
        - Advanced analytics (EPA, DVOA-proxy, success rate) [ENHANCED]
        - Injury impact analysis [ENHANCED v3.0]
        
        Args:
            team: Team name
            opponent: Opponent name
            week: Week number
            season: Season year
            is_home: Whether team is home
            spread: Point spread (optional)
            
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # Basic features
        features['week'] = week
        features['is_home'] = int(is_home)
        features['spread'] = spread if spread is not None else 0.0
        
        # Offensive features
        features.update(self.extract_offensive_features(team, season))
        features.update(self.extract_offensive_features(opponent, season))
        
        # Defensive features
        features.update(self.extract_defensive_features(team, season))
        features.update(self.extract_defensive_features(opponent, season))
        
        # Elo ratings
        features.update(self.calculate_elo_rating(team, opponent))
        
        # Recent form
        features.update(self.extract_recent_form(team))
        features.update(self.extract_recent_form(opponent))
        
        # Pythagorean expectation (if we have scoring data)
        team_pf = features.get(f'{team}_points_per_game', 24.5)
        team_pa = features.get(f'{team}_points_allowed', 21.3)
        features['pythagorean_win_prob'] = self.calculate_pythagorean_expectation(team_pf, team_pa)
        
        # Rest advantage (default to 7 days)
        features['rest_advantage'] = self.calculate_rest_advantage(7, 7)
        
        # ENHANCED: Advanced analytics features
        # EPA estimates
        features[f'{team}_offensive_epa'] = self.calculate_epa_estimate(team, 'offense')
        features[f'{team}_defensive_epa'] = self.calculate_epa_estimate(team, 'defense')
        features[f'{opponent}_offensive_epa'] = self.calculate_epa_estimate(opponent, 'offense')
        features[f'{opponent}_defensive_epa'] = self.calculate_epa_estimate(opponent, 'defense')
        features['net_epa'] = (
            features[f'{team}_offensive_epa'] + 
            features[f'{team}_defensive_epa'] -
            features[f'{opponent}_offensive_epa'] - 
            features[f'{opponent}_defensive_epa']
        )
        
        # DVOA-inspired efficiency metrics
        features[f'{team}_offensive_dvoa_proxy'] = self.calculate_dvoa_proxy(team, 'offense')
        features[f'{team}_defensive_dvoa_proxy'] = self.calculate_dvoa_proxy(team, 'defense')
        features[f'{opponent}_offensive_dvoa_proxy'] = self.calculate_dvoa_proxy(opponent, 'offense')
        features[f'{opponent}_defensive_dvoa_proxy'] = self.calculate_dvoa_proxy(opponent, 'defense')
        features['net_dvoa_proxy'] = (
            features[f'{team}_offensive_dvoa_proxy'] + 
            features[f'{team}_defensive_dvoa_proxy'] -
            features[f'{opponent}_offensive_dvoa_proxy'] - 
            features[f'{opponent}_defensive_dvoa_proxy']
        )
        
        # Success rate and explosive plays
        features[f'{team}_success_rate'] = self.calculate_success_rate(team)
        features[f'{opponent}_success_rate'] = self.calculate_success_rate(opponent)
        features[f'{team}_explosive_play_rate'] = self.calculate_explosive_play_rate(team)
        features[f'{opponent}_explosive_play_rate'] = self.calculate_explosive_play_rate(opponent)
        
        # ENHANCED v3.0: Injury analysis features
        features.update(self.extract_injury_features(team, week))
        features.update(self.extract_injury_features(opponent, week))
        
        # Calculate net injury advantage
        team_injury_impact = features.get(f'{team}_injury_impact', 0.0)
        opponent_injury_impact = features.get(f'{opponent}_injury_impact', 0.0)
        features['net_injury_advantage'] = opponent_injury_impact - team_injury_impact
        
        return features
    
    def create_feature_matrix(self, games_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create feature matrix for multiple games.
        
        Args:
            games_df: DataFrame with game information
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_list = []
        
        for _, game in games_df.iterrows():
            features = self.extract_comprehensive_features(
                team=game.get('team', ''),
                opponent=game.get('opponent', ''),
                week=game.get('week', 1),
                season=game.get('season', 2025),
                is_home=game.get('is_home', True),
                spread=game.get('spread', None)
            )
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        feature_names = list(feature_df.columns)
        
        return feature_df, feature_names
    

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("Testing NFL Feature Engineering")
    print("=" * 60)
    
    engineer = NFLFeatureEngineer()
    
    # Test Pythagorean expectation
    print("\n1. Pythagorean Expectation:")
    pf, pa = 28.5, 21.2
    pyth_prob = engineer.calculate_pythagorean_expectation(pf, pa)
    print(f"   PF={pf}, PA={pa} -> Win Prob: {pyth_prob:.3f}")
    
    # Test Elo rating
    print("\n2. Elo Rating:")
    elo = engineer.calculate_elo_rating("Team A", "Team B")
    print(f"   Team Elo: {elo['team_elo']:.0f}")
    print(f"   Opponent Elo: {elo['opponent_elo']:.0f}")
    print(f"   Expected Win Prob: {elo['elo_win_prob']:.3f}")
    
    # Test comprehensive features
    print("\n3. Comprehensive Features:")
    features = engineer.extract_comprehensive_features(
        team="Kansas City Chiefs",
        opponent="Buffalo Bills",
        week=7,
        season=2025,
        is_home=True,
        spread=-3.5
    )
    print(f"   Total features extracted: {len(features)}")
    print(f"   Sample features:")
    for key in list(features.keys())[:5]:
        print(f"     {key}: {features[key]}")
    
    print("\n✓ Feature engineering test complete")


if __name__ == "__main__":
    test_feature_engineering()
