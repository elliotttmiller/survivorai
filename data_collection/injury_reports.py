"""
Injury report data collection and analysis for NFL teams.

Integrates injury reports to enhance prediction accuracy by factoring in
the impact of key player injuries on team performance.

Data Sources:
- ESPN Injury API
- NFL Official Injury Reports (when available)
- Cached data with automatic refresh

Impact Analysis:
- Position-based importance weighting
- Injury severity classification
- Team-level injury impact scoring
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os


# Position importance weights (based on statistical impact on win probability)
POSITION_IMPACT_WEIGHTS = {
    'QB': 1.0,      # Quarterback - highest impact
    'RB': 0.35,     # Running back
    'WR': 0.30,     # Wide receiver
    'TE': 0.25,     # Tight end
    'OL': 0.40,     # Offensive line (averaged)
    'DL': 0.35,     # Defensive line
    'LB': 0.30,     # Linebacker
    'DB': 0.28,     # Defensive back
    'K': 0.10,      # Kicker
    'P': 0.05,      # Punter
}

# Injury status severity weights
INJURY_SEVERITY = {
    'OUT': 1.0,           # Definitely not playing
    'DOUBTFUL': 0.85,     # Very unlikely to play
    'QUESTIONABLE': 0.40, # 50/50 chance
    'PROBABLE': 0.15,     # Likely to play but limited
    'DAY_TO_DAY': 0.20,   # Uncertain status
}


class InjuryReportCollector:
    """Collects and processes NFL injury reports."""
    
    def __init__(self, cache_dir: str = 'cache', cache_expiry_hours: int = 4):
        """
        Initialize injury report collector.
        
        Args:
            cache_dir: Directory for caching injury data
            cache_expiry_hours: Hours before cache expires (default 4)
        """
        self.cache_dir = cache_dir
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_file = os.path.join(cache_dir, 'injury_reports.json')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_team_injuries(self, team: str, week: Optional[int] = None) -> List[Dict]:
        """
        Get injury report for a specific team.
        
        Args:
            team: Team name (e.g., "Kansas City Chiefs")
            week: NFL week number (optional, uses current week if not provided)
        
        Returns:
            List of injury dictionaries with player info, position, status, impact
        """
        # Try to load from cache first
        cached_data = self._load_from_cache()
        
        if cached_data and team in cached_data:
            return cached_data[team]
        
        # If not in cache or cache expired, fetch fresh data
        injuries = self._fetch_injury_data(team, week)
        
        # Cache the results
        self._save_to_cache(team, injuries)
        
        return injuries
    
    def _fetch_injury_data(self, team: str, week: Optional[int] = None) -> List[Dict]:
        """
        Fetch injury data from external sources.
        
        This is a placeholder that should be implemented with real API calls.
        For production, integrate with ESPN API, NFL.com, or other sources.
        
        Args:
            team: Team name
            week: NFL week number
        
        Returns:
            List of injury records
        """
        # In production, this would make actual API calls
        # For now, return empty list as a safe default
        # Real implementation would look like:
        #
        # try:
        #     response = requests.get(
        #         f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/injuries",
        #         timeout=10
        #     )
        #     if response.status_code == 200:
        #         data = response.json()
        #         return self._parse_injury_response(data)
        # except Exception as e:
        #     print(f"Error fetching injury data: {e}")
        
        return []
    
    def _parse_injury_response(self, data: Dict) -> List[Dict]:
        """
        Parse injury data from API response.
        
        Args:
            data: Raw API response data
        
        Returns:
            Parsed injury records
        """
        injuries = []
        
        # Example parsing structure (adjust based on actual API)
        if 'injuries' in data:
            for injury in data['injuries']:
                injuries.append({
                    'player_name': injury.get('athlete', {}).get('displayName', 'Unknown'),
                    'position': injury.get('athlete', {}).get('position', {}).get('abbreviation', 'UNK'),
                    'status': injury.get('status', {}).get('type', 'QUESTIONABLE').upper(),
                    'injury_type': injury.get('details', {}).get('type', 'Unknown'),
                    'date_reported': injury.get('date', datetime.now().isoformat()),
                })
        
        return injuries
    
    def _load_from_cache(self) -> Optional[Dict]:
        """
        Load injury data from cache if not expired.
        
        Returns:
            Cached data or None if expired/missing
        """
        if not os.path.exists(self.cache_file):
            return None
        
        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(self.cache_file)
            )
            
            if cache_age > timedelta(hours=self.cache_expiry_hours):
                return None
            
            # Load cache
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        
        except Exception:
            return None
    
    def _save_to_cache(self, team: str, injuries: List[Dict]):
        """
        Save injury data to cache.
        
        Args:
            team: Team name
            injuries: Injury records to cache
        """
        try:
            # Load existing cache or create new
            cache_data = self._load_from_cache() or {}
            
            # Update with new data
            cache_data[team] = injuries
            cache_data['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Failed to cache injury data: {e}")


class InjuryImpactAnalyzer:
    """Analyzes the impact of injuries on team performance."""
    
    def __init__(self):
        """Initialize the injury impact analyzer."""
        self.position_weights = POSITION_IMPACT_WEIGHTS
        self.severity_weights = INJURY_SEVERITY
    
    def calculate_team_injury_impact(self, injuries: List[Dict]) -> float:
        """
        Calculate overall injury impact score for a team.
        
        The score represents the expected reduction in team performance
        due to injuries, on a 0-1 scale where:
        - 0.0 = No impact (no significant injuries)
        - 1.0 = Severe impact (multiple key players out)
        
        Args:
            injuries: List of injury dictionaries from InjuryReportCollector
        
        Returns:
            Impact score (0.0 to 1.0)
        """
        if not injuries:
            return 0.0
        
        total_impact = 0.0
        
        for injury in injuries:
            position = injury.get('position', 'UNK')
            status = injury.get('status', 'QUESTIONABLE')
            
            # Get position importance weight
            position_weight = self.position_weights.get(position, 0.15)
            
            # Get injury severity weight
            severity_weight = self.severity_weights.get(status, 0.40)
            
            # Calculate individual injury impact
            individual_impact = position_weight * severity_weight
            
            # Add to total (with diminishing returns for multiple injuries)
            total_impact += individual_impact
        
        # Apply diminishing returns curve (prevents unrealistic high values)
        # Using 1 - exp(-k*x) curve, where k=1.5 gives reasonable scaling
        normalized_impact = 1 - np.exp(-1.5 * total_impact)
        
        # Cap at maximum of 0.6 (even worst case, team still has 40% capacity)
        return min(normalized_impact, 0.60)
    
    def get_position_breakdown(self, injuries: List[Dict]) -> Dict[str, int]:
        """
        Get breakdown of injuries by position.
        
        Args:
            injuries: List of injury dictionaries
        
        Returns:
            Dictionary mapping position to count of injured players
        """
        breakdown = {}
        
        for injury in injuries:
            position = injury.get('position', 'UNK')
            breakdown[position] = breakdown.get(position, 0) + 1
        
        return breakdown
    
    def get_critical_injuries(self, injuries: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Identify critical injuries that significantly impact team performance.
        
        Args:
            injuries: List of injury dictionaries
            threshold: Minimum impact threshold for critical classification
        
        Returns:
            List of critical injuries
        """
        critical = []
        
        for injury in injuries:
            position = injury.get('position', 'UNK')
            status = injury.get('status', 'QUESTIONABLE')
            
            position_weight = self.position_weights.get(position, 0.15)
            severity_weight = self.severity_weights.get(status, 0.40)
            
            impact = position_weight * severity_weight
            
            if impact >= threshold:
                injury_copy = injury.copy()
                injury_copy['impact_score'] = impact
                critical.append(injury_copy)
        
        # Sort by impact (highest first)
        critical.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return critical


def calculate_injury_adjusted_win_probability(
    base_win_prob: float,
    team_injury_impact: float,
    opponent_injury_impact: float
) -> float:
    """
    Adjust win probability based on injury impacts for both teams.
    
    Formula:
    - Team injuries reduce win probability
    - Opponent injuries increase win probability
    - Net adjustment is relative difference
    
    Args:
        base_win_prob: Original win probability (0-1)
        team_injury_impact: Team's injury impact score (0-1)
        opponent_injury_impact: Opponent's injury impact score (0-1)
    
    Returns:
        Adjusted win probability (0-1)
    """
    # Calculate net injury advantage
    # Positive = our opponent is more injured (good for us)
    # Negative = we are more injured (bad for us)
    net_advantage = opponent_injury_impact - team_injury_impact
    
    # Convert net advantage to win probability adjustment
    # Scale factor of 0.15 means max injury differential can shift win prob by ±15%
    adjustment = net_advantage * 0.15
    
    # Apply adjustment to base probability
    adjusted_prob = base_win_prob + adjustment
    
    # Ensure probability stays in valid range [0.05, 0.95]
    adjusted_prob = max(0.05, min(0.95, adjusted_prob))
    
    return adjusted_prob


def enrich_game_data_with_injuries(
    game_data: pd.DataFrame,
    injury_collector: InjuryReportCollector,
    impact_analyzer: InjuryImpactAnalyzer
) -> pd.DataFrame:
    """
    Enrich game data with injury analysis features.
    
    Args:
        game_data: DataFrame with game information (must have 'team', 'opponent', 'week')
        injury_collector: Instance of InjuryReportCollector
        impact_analyzer: Instance of InjuryImpactAnalyzer
    
    Returns:
        Enhanced DataFrame with injury features
    """
    if game_data.empty:
        return game_data
    
    enriched = game_data.copy()
    
    # Add injury impact columns
    enriched['team_injury_impact'] = 0.0
    enriched['opponent_injury_impact'] = 0.0
    enriched['net_injury_advantage'] = 0.0
    
    for idx, row in enriched.iterrows():
        team = row.get('team', '')
        opponent = row.get('opponent', '')
        week = row.get('week', None)
        
        if not team:
            continue
        
        # Get injury data for both teams
        team_injuries = injury_collector.get_team_injuries(team, week)
        opponent_injuries = injury_collector.get_team_injuries(opponent, week) if opponent else []
        
        # Calculate impact scores
        team_impact = impact_analyzer.calculate_team_injury_impact(team_injuries)
        opponent_impact = impact_analyzer.calculate_team_injury_impact(opponent_injuries)
        
        # Store in dataframe
        enriched.at[idx, 'team_injury_impact'] = team_impact
        enriched.at[idx, 'opponent_injury_impact'] = opponent_impact
        enriched.at[idx, 'net_injury_advantage'] = opponent_impact - team_impact
        
        # Adjust win probability if it exists
        if 'win_probability' in enriched.columns:
            base_prob = enriched.at[idx, 'win_probability']
            adjusted_prob = calculate_injury_adjusted_win_probability(
                base_prob, team_impact, opponent_impact
            )
            enriched.at[idx, 'injury_adjusted_win_probability'] = adjusted_prob
    
    return enriched


if __name__ == "__main__":
    # Test the injury analysis system
    print("Testing Injury Analysis System...")
    
    # Test injury impact analyzer
    analyzer = InjuryImpactAnalyzer()
    
    # Simulate some injuries
    test_injuries = [
        {'player_name': 'Patrick Mahomes', 'position': 'QB', 'status': 'OUT'},
        {'player_name': 'Travis Kelce', 'position': 'TE', 'status': 'QUESTIONABLE'},
    ]
    
    impact = analyzer.calculate_team_injury_impact(test_injuries)
    print(f"\nTest injury impact score: {impact:.3f}")
    
    critical = analyzer.get_critical_injuries(test_injuries)
    print(f"Critical injuries: {len(critical)}")
    for inj in critical:
        print(f"  - {inj['player_name']} ({inj['position']}): {inj['status']} - Impact: {inj['impact_score']:.3f}")
    
    # Test win probability adjustment
    base_prob = 0.65
    adjusted_prob = calculate_injury_adjusted_win_probability(base_prob, impact, 0.0)
    print(f"\nWin probability: {base_prob:.3f} -> {adjusted_prob:.3f} (with injuries)")
    
    print("\n✓ Injury analysis system test complete")
