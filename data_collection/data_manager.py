"""Data manager to combine and integrate data from multiple sources."""
import pandas as pd
from typing import List, Optional
import config
from data_collection.odds_api import OddsAPIClient
from data_collection.survivorgrid_scraper import SurvivorGridScraper
from data_collection.schedule_data import generate_schedule_based_data
from data_collection.advanced_data_sources import integrate_all_data_sources

# Import injury analysis components
try:
    from data_collection.injury_reports import (
        InjuryReportCollector,
        InjuryImpactAnalyzer,
        enrich_game_data_with_injuries
    )
    INJURY_ANALYSIS_AVAILABLE = True
except ImportError:
    INJURY_ANALYSIS_AVAILABLE = False
    print("Injury analysis not available. Install required dependencies.")

# Try to import ML predictor
try:
    from ml_models.ml_predictor import MLNFLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML models not available. Install ML dependencies or disable ML predictions.")


class DataManager:
    """Manages data collection and integration from multiple sources."""

    def __init__(
        self, 
        use_odds_api: bool = True, 
        use_ml_predictions: bool = False,
        use_advanced_metrics: bool = True,
        use_historical_data: bool = True,
        use_injury_analysis: bool = True
    ):
        """
        Initialize data manager with comprehensive data sources.

        Args:
            use_odds_api: Whether to use The Odds API (requires API key)
            use_ml_predictions: Whether to enhance predictions with ML models
            use_advanced_metrics: Whether to calculate advanced metrics (Elo, etc.)
            use_historical_data: Whether to include historical statistics
            use_injury_analysis: Whether to integrate injury impact analysis (default True)
        """
        self.use_odds_api = use_odds_api
        self.use_ml_predictions = use_ml_predictions and ML_AVAILABLE
        self.use_advanced_metrics = use_advanced_metrics
        self.use_historical_data = use_historical_data
        self.use_injury_analysis = use_injury_analysis and INJURY_ANALYSIS_AVAILABLE
        
        # Initialize data source clients
        self.odds_client = None
        if use_odds_api and config.ODDS_API_KEY:
            try:
                self.odds_client = OddsAPIClient()
                print(f"✓ The Odds API connected")
            except Exception as e:
                print(f"Warning: Could not initialize Odds API: {e}")
        
        self.sg_scraper = SurvivorGridScraper()
        
        # Initialize injury analysis components if enabled
        if self.use_injury_analysis:
            try:
                self.injury_collector = InjuryReportCollector(cache_expiry_hours=4, use_fallback=True)
                self.injury_analyzer = InjuryImpactAnalyzer()
                print(f"✓ Injury analysis initialized")
            except Exception as e:
                print(f"Warning: Could not initialize injury analysis: {e}")
                self.use_injury_analysis = False
                self.injury_collector = None
                self.injury_analyzer = None
        else:
            self.injury_collector = None
            self.injury_analyzer = None
        
        # Initialize ML predictor if requested and available
        if self.use_ml_predictions:
            try:
                self.ml_predictor = MLNFLPredictor(
                    model_type=config.ML_MODEL_TYPE
                )
                print(f"✓ ML predictor initialized ({config.ML_MODEL_TYPE})")
            except Exception as e:
                print(f"Warning: Could not initialize ML predictor: {e}")
                self.use_ml_predictions = False
                self.ml_predictor = None
        else:
            self.ml_predictor = None
        
        print(f"✓ Data sources initialized:")
        print(f"   • SurvivorGrid: Enabled")
        print(f"   • The Odds API: {'Enabled' if self.odds_client else 'Disabled (no API key)'}")
        print(f"   • ML Predictions: {'Enabled' if self.use_ml_predictions else 'Disabled'}")
        print(f"   • Advanced Metrics: {'Enabled' if self.use_advanced_metrics else 'Disabled'}")
        print(f"   • Historical Data: {'Enabled' if self.use_historical_data else 'Disabled'}")
        print(f"   • Injury Analysis: {'Enabled' if self.use_injury_analysis else 'Disabled'}")

    def get_comprehensive_data(self, current_week: Optional[int] = None) -> pd.DataFrame:
        """
        Get comprehensive data combining all sources.

        Strategy:
        - Use The Odds API for current week and next 1-2 weeks (more accurate)
        - Use SurvivorGrid for all weeks including future weeks
        - Auto-detect current_week from SurvivorGrid's earliest available week

        Args:
            current_week: Current NFL week (defaults to auto-detect from SurvivorGrid)

        Returns:
            DataFrame with columns: week, team, win_probability, pick_pct, ev, opponent
        """
        # Get SurvivorGrid data for all weeks (primary source)
        print("Fetching data from SurvivorGrid...")
        # Pass None to get all available weeks without filtering
        sg_data = self.sg_scraper.get_all_weeks_data(current_week=None)
        
        # Auto-detect current week from SurvivorGrid's earliest available week
        if not sg_data.empty:
            available_weeks = sorted(sg_data['week'].unique())
            detected_week = min(available_weeks)
            
            if current_week is None:
                current_week = detected_week
                print(f"✓ Auto-detected current week: {current_week} (from SurvivorGrid)")
            elif current_week != detected_week:
                print(f"⚠️  Note: Requested week {current_week}, but SurvivorGrid shows week {detected_week}")
                print(f"   Using SurvivorGrid's earliest available week: {detected_week}")
                current_week = detected_week
            
            # Update config for the session
            config.CURRENT_WEEK = current_week
        else:
            # Fallback to provided week or config
            current_week = current_week or config.CURRENT_WEEK

        # Get The Odds API data for current week (more accurate for near-term)
        odds_data = pd.DataFrame()
        if self.use_odds_api and self.odds_client:
            try:
                print("Fetching current week odds from The Odds API...")
                odds_data = self.odds_client.get_win_probabilities()
            except Exception as e:
                print(f"Warning: Could not fetch odds API data: {e}")

        # Merge the data sources
        combined_data = self._merge_data_sources(sg_data, odds_data, current_week)
        
        # If we have very limited data, supplement with schedule-based projections
        if combined_data.empty or len(combined_data) < 32:
            print("⚠️  Limited data from primary sources, adding schedule-based projections...")
            schedule_data = generate_schedule_based_data(current_week, 18)
            if combined_data.empty:
                combined_data = schedule_data
            else:
                # Fill in missing weeks
                existing_weeks = set(combined_data['week'].unique())
                schedule_weeks = set(schedule_data['week'].unique())
                missing_weeks = schedule_weeks - existing_weeks
                if missing_weeks:
                    missing_data = schedule_data[schedule_data['week'].isin(missing_weeks)]
                    combined_data = pd.concat([combined_data, missing_data], ignore_index=True)
        
        # Enhance with advanced metrics and historical data
        if (self.use_advanced_metrics or self.use_historical_data) and not combined_data.empty:
            print("Enhancing with advanced metrics and historical data...")
            combined_data = integrate_all_data_sources(
                combined_data,
                use_historical=self.use_historical_data,
                use_advanced_metrics=self.use_advanced_metrics
            )
            
            # Use enhanced probabilities if available
            if 'enhanced_win_probability' in combined_data.columns:
                combined_data['win_probability'] = combined_data['enhanced_win_probability']
            
            # Recalculate EV with enhanced probabilities
            if 'pick_pct' in combined_data.columns:
                combined_data['ev'] = combined_data['win_probability'] * (1 - combined_data['pick_pct'])
            
            print("✓ Advanced metrics applied")
        
        # Enhance with injury analysis if enabled (before ML predictions)
        if self.use_injury_analysis and self.injury_collector and self.injury_analyzer and not combined_data.empty:
            try:
                print("Applying injury impact analysis...")
                combined_data = enrich_game_data_with_injuries(
                    combined_data,
                    self.injury_collector,
                    self.injury_analyzer
                )
                
                # Apply injury adjustments to win probabilities
                if 'injury_adjusted_win_probability' in combined_data.columns:
                    combined_data['win_probability'] = combined_data['injury_adjusted_win_probability']
                    
                    # Recalculate EV with injury-adjusted probabilities
                    if 'pick_pct' in combined_data.columns:
                        combined_data['ev'] = combined_data['win_probability'] * (1 - combined_data['pick_pct'])
                
                print("✓ Injury analysis applied")
            except Exception as e:
                print(f"Warning: Could not apply injury analysis: {e}")
        
        # Enhance with ML predictions if enabled (final layer)
        if self.use_ml_predictions and self.ml_predictor is not None and not combined_data.empty:
            try:
                print("Applying ML predictions (final enhancement)...")
                combined_data = self.ml_predictor.enhance_data_manager_predictions(combined_data)
                print("✓ ML enhancement complete")
            except Exception as e:
                print(f"Warning: Could not apply ML predictions: {e}")

        return combined_data

    def _merge_data_sources(
        self,
        sg_data: pd.DataFrame,
        odds_data: pd.DataFrame,
        current_week: int
    ) -> pd.DataFrame:
        """
        Merge data from SurvivorGrid and Odds API.

        Args:
            sg_data: Data from SurvivorGrid
            odds_data: Data from The Odds API
            current_week: Current week number

        Returns:
            Merged DataFrame
        """
        if sg_data.empty:
            print("Warning: No SurvivorGrid data available")
            return pd.DataFrame()

        # Start with SurvivorGrid data as base
        result = sg_data.copy()
        result.rename(columns={'win_pct': 'win_probability'}, inplace=True)

        # Override with Odds API data if available
        if not odds_data.empty:
            # Get unique weeks from odds data
            odds_weeks = odds_data['week'].unique()

            # Remove those weeks from SurvivorGrid data
            result = result[~result['week'].isin(odds_weeks)]

            # Prepare odds data - DON'T override the week column!
            odds_to_add = odds_data.copy()

            # For each week in odds data, merge pick_pct and spread from SurvivorGrid
            for week in odds_weeks:
                sg_week = sg_data[sg_data['week'] == week][['team', 'pick_pct', 'ev', 'spread']]
                week_mask = odds_to_add['week'] == week

                # Merge pick percentages and spreads for this week
                for idx in odds_to_add[week_mask].index:
                    team = odds_to_add.at[idx, 'team']
                    sg_team_data = sg_week[sg_week['team'] == team]
                    if not sg_team_data.empty:
                        odds_to_add.at[idx, 'pick_pct'] = sg_team_data.iloc[0]['pick_pct']
                        # Also preserve spread from SurvivorGrid since Odds API doesn't provide it
                        if 'spread' in sg_team_data.columns:
                            odds_to_add.at[idx, 'spread'] = sg_team_data.iloc[0]['spread']

            # Fill missing pick_pct with default
            if 'pick_pct' not in odds_to_add.columns:
                odds_to_add['pick_pct'] = 0.05
            else:
                odds_to_add['pick_pct'] = odds_to_add['pick_pct'].fillna(0.05)

            odds_to_add['ev'] = odds_to_add['win_probability'] * (1 - odds_to_add['pick_pct'])

            # Combine
            result = pd.concat([result, odds_to_add], ignore_index=True)

        # Ensure required columns exist
        required_cols = ['week', 'team', 'win_probability', 'pick_pct', 'ev']
        for col in required_cols:
            if col not in result.columns:
                result[col] = 0.0 if col != 'team' else ''

        # Add opponent and moneyline if missing
        if 'opponent' not in result.columns:
            result['opponent'] = ''
        if 'moneyline' not in result.columns:
            result['moneyline'] = None

        # Sort by week and team
        result = result.sort_values(['week', 'team']).reset_index(drop=True)

        return result

    def get_week_data(self, week: int) -> pd.DataFrame:
        """
        Get data for a specific week.

        Args:
            week: Week number

        Returns:
            DataFrame with data for that week only
        """
        all_data = self.get_comprehensive_data()
        return all_data[all_data['week'] == week].copy()

    def get_team_schedule(self, team: str, start_week: Optional[int] = None) -> pd.DataFrame:
        """
        Get schedule/probabilities for a specific team.

        Args:
            team: Team name
            start_week: Starting week (defaults to current week)

        Returns:
            DataFrame with that team's data for all remaining weeks
        """
        start_week = start_week or config.CURRENT_WEEK
        all_data = self.get_comprehensive_data()

        team_data = all_data[
            (all_data['team'] == team) & (all_data['week'] >= start_week)
        ].copy()

        return team_data.sort_values('week')

    def get_available_teams(self, used_teams: List[str]) -> List[str]:
        """
        Get list of teams that haven't been used yet.

        Args:
            used_teams: List of teams already picked

        Returns:
            List of available team names
        """
        all_teams = set(config.NFL_TEAMS)
        used = set(used_teams)
        available = sorted(all_teams - used)
        return available


def test_data_manager():
    """Test the data manager."""
    # Test without Odds API first (doesn't require API key)
    print("Testing with SurvivorGrid only...")
    manager = DataManager(use_odds_api=False)

    data = manager.get_comprehensive_data()
    print(f"\nTotal rows: {len(data)}")
    print(f"Weeks: {sorted(data['week'].unique())}")
    print(f"Teams: {len(data['team'].unique())}")

    if not data.empty:
        print("\nSample data:")
        print(data.head(10))

        print("\nWeek 7 data:")
        week7 = manager.get_week_data(7)
        print(week7.head())

        print("\nKansas City Chiefs schedule:")
        kc_schedule = manager.get_team_schedule('Kansas City Chiefs')
        print(kc_schedule)

    # Test with Odds API if key is available
    if config.ODDS_API_KEY:
        print("\n\nTesting with The Odds API...")
        manager_with_api = DataManager(use_odds_api=True)
        data_with_api = manager_with_api.get_comprehensive_data()
        print(f"Total rows with API: {len(data_with_api)}")


if __name__ == "__main__":
    test_data_manager()
