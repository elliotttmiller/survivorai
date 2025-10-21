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
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class NFLFeatureEngineer:
    """Feature engineering for NFL game prediction."""
    
    def __init__(self, historical_seasons: int = 3):
        """
        Initialize feature engineer.
        
        Args:
            historical_seasons: Number of past seasons to consider for historical features
        """
        self.historical_seasons = historical_seasons
        self.team_stats_cache = {}
        
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
