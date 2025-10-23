"""Tests for injury analysis functionality."""
import unittest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.injury_reports import (
    InjuryReportCollector,
    InjuryImpactAnalyzer,
    calculate_injury_adjusted_win_probability,
    enrich_game_data_with_injuries,
    POSITION_IMPACT_WEIGHTS,
    INJURY_SEVERITY
)


class TestInjuryImpactAnalyzer(unittest.TestCase):
    """Test injury impact analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = InjuryImpactAnalyzer()
    
    def test_no_injuries(self):
        """Test that no injuries results in zero impact."""
        impact = self.analyzer.calculate_team_injury_impact([])
        self.assertEqual(impact, 0.0)
    
    def test_qb_injury_high_impact(self):
        """Test that QB injury has highest impact."""
        qb_injury = [{'player_name': 'Test QB', 'position': 'QB', 'status': 'OUT'}]
        qb_impact = self.analyzer.calculate_team_injury_impact(qb_injury)
        
        # QB out should have significant impact
        self.assertGreater(qb_impact, 0.5)
    
    def test_kicker_injury_low_impact(self):
        """Test that kicker injury has low impact."""
        k_injury = [{'player_name': 'Test K', 'position': 'K', 'status': 'OUT'}]
        k_impact = self.analyzer.calculate_team_injury_impact(k_injury)
        
        # Kicker should have minimal impact
        self.assertLess(k_impact, 0.2)
    
    def test_multiple_injuries(self):
        """Test multiple injuries aggregate properly."""
        injuries = [
            {'player_name': 'Player 1', 'position': 'WR', 'status': 'OUT'},
            {'player_name': 'Player 2', 'position': 'RB', 'status': 'DOUBTFUL'},
            {'player_name': 'Player 3', 'position': 'TE', 'status': 'QUESTIONABLE'},
        ]
        impact = self.analyzer.calculate_team_injury_impact(injuries)
        
        # Multiple injuries should have moderate to high impact
        self.assertGreater(impact, 0.2)
        self.assertLessEqual(impact, 0.6)  # Capped at 0.6
    
    def test_injury_status_severity(self):
        """Test that injury status affects severity correctly."""
        injuries_out = [{'player_name': 'Test', 'position': 'RB', 'status': 'OUT'}]
        injuries_questionable = [{'player_name': 'Test', 'position': 'RB', 'status': 'QUESTIONABLE'}]
        
        impact_out = self.analyzer.calculate_team_injury_impact(injuries_out)
        impact_questionable = self.analyzer.calculate_team_injury_impact(injuries_questionable)
        
        # OUT should have higher impact than QUESTIONABLE
        self.assertGreater(impact_out, impact_questionable)
    
    def test_impact_cap(self):
        """Test that impact is capped at maximum value."""
        # Create many severe injuries
        injuries = [
            {'player_name': f'Player {i}', 'position': 'QB', 'status': 'OUT'}
            for i in range(10)
        ]
        impact = self.analyzer.calculate_team_injury_impact(injuries)
        
        # Should be capped at 0.6
        self.assertLessEqual(impact, 0.60)
    
    def test_position_breakdown(self):
        """Test position breakdown calculation."""
        injuries = [
            {'player_name': 'QB1', 'position': 'QB', 'status': 'OUT'},
            {'player_name': 'WR1', 'position': 'WR', 'status': 'OUT'},
            {'player_name': 'WR2', 'position': 'WR', 'status': 'DOUBTFUL'},
        ]
        breakdown = self.analyzer.get_position_breakdown(injuries)
        
        self.assertEqual(breakdown['QB'], 1)
        self.assertEqual(breakdown['WR'], 2)
    
    def test_critical_injuries_identification(self):
        """Test identification of critical injuries."""
        injuries = [
            {'player_name': 'QB', 'position': 'QB', 'status': 'OUT'},  # Critical
            {'player_name': 'K', 'position': 'K', 'status': 'OUT'},    # Not critical
            {'player_name': 'RB', 'position': 'RB', 'status': 'QUESTIONABLE'},  # Borderline
        ]
        critical = self.analyzer.get_critical_injuries(injuries, threshold=0.3)
        
        # QB injury should be critical
        self.assertGreater(len(critical), 0)
        self.assertEqual(critical[0]['position'], 'QB')
        
        # Should have impact scores
        for inj in critical:
            self.assertIn('impact_score', inj)
            self.assertGreaterEqual(inj['impact_score'], 0.3)


class TestInjuryReportCollector(unittest.TestCase):
    """Test injury report collection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.collector = InjuryReportCollector(cache_dir=self.temp_dir, cache_expiry_hours=1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test collector initialization."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertIsNotNone(self.collector)
    
    def test_get_team_injuries_returns_list(self):
        """Test that get_team_injuries returns a list."""
        injuries = self.collector.get_team_injuries("Kansas City Chiefs")
        self.assertIsInstance(injuries, list)
    
    def test_cache_creation(self):
        """Test that cache file is created."""
        # Get injuries (will create cache)
        self.collector.get_team_injuries("Kansas City Chiefs")
        
        # Save some data to cache
        self.collector._save_to_cache("Kansas City Chiefs", [])
        
        # Check cache file exists
        self.assertTrue(os.path.exists(self.collector.cache_file))
    
    def test_parse_injury_response(self):
        """Test parsing of injury data."""
        sample_data = {
            'injuries': [
                {
                    'athlete': {
                        'displayName': 'Patrick Mahomes',
                        'position': {'abbreviation': 'QB'}
                    },
                    'status': {'type': 'OUT'},
                    'details': {'type': 'Knee'},
                    'date': '2025-10-20'
                }
            ]
        }
        
        parsed = self.collector._parse_injury_response(sample_data)
        
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['player_name'], 'Patrick Mahomes')
        self.assertEqual(parsed[0]['position'], 'QB')
        self.assertEqual(parsed[0]['status'], 'OUT')


class TestInjuryAdjustedWinProbability(unittest.TestCase):
    """Test injury-adjusted win probability calculations."""
    
    def test_no_injuries_no_change(self):
        """Test that no injuries don't change win probability."""
        base_prob = 0.65
        adjusted = calculate_injury_adjusted_win_probability(base_prob, 0.0, 0.0)
        self.assertEqual(adjusted, base_prob)
    
    def test_team_injuries_reduce_probability(self):
        """Test that team injuries reduce win probability."""
        base_prob = 0.65
        adjusted = calculate_injury_adjusted_win_probability(base_prob, 0.4, 0.0)
        
        # Should decrease
        self.assertLess(adjusted, base_prob)
    
    def test_opponent_injuries_increase_probability(self):
        """Test that opponent injuries increase win probability."""
        base_prob = 0.65
        adjusted = calculate_injury_adjusted_win_probability(base_prob, 0.0, 0.4)
        
        # Should increase
        self.assertGreater(adjusted, base_prob)
    
    def test_probability_bounds(self):
        """Test that probability stays within valid bounds."""
        # Extreme case: both teams heavily injured
        adjusted = calculate_injury_adjusted_win_probability(0.95, 0.6, 0.0)
        self.assertLessEqual(adjusted, 0.95)
        self.assertGreaterEqual(adjusted, 0.05)
        
        # Another extreme case
        adjusted = calculate_injury_adjusted_win_probability(0.05, 0.0, 0.6)
        self.assertLessEqual(adjusted, 0.95)
        self.assertGreaterEqual(adjusted, 0.05)
    
    def test_symmetric_injuries_cancel(self):
        """Test that symmetric injuries roughly cancel out."""
        base_prob = 0.65
        adjusted = calculate_injury_adjusted_win_probability(base_prob, 0.3, 0.3)
        
        # Should be close to base (within adjustment tolerance)
        self.assertAlmostEqual(adjusted, base_prob, delta=0.02)


class TestGameDataEnrichment(unittest.TestCase):
    """Test enrichment of game data with injury analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = InjuryReportCollector(cache_dir=self.temp_dir)
        self.analyzer = InjuryImpactAnalyzer()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_empty_dataframe(self):
        """Test enrichment with empty dataframe."""
        empty_df = pd.DataFrame()
        enriched = enrich_game_data_with_injuries(empty_df, self.collector, self.analyzer)
        
        self.assertTrue(enriched.empty)
    
    def test_enrichment_adds_columns(self):
        """Test that enrichment adds injury columns."""
        game_data = pd.DataFrame({
            'team': ['Kansas City Chiefs'],
            'opponent': ['Buffalo Bills'],
            'week': [7],
            'win_probability': [0.65]
        })
        
        enriched = enrich_game_data_with_injuries(game_data, self.collector, self.analyzer)
        
        # Check that new columns are added
        self.assertIn('team_injury_impact', enriched.columns)
        self.assertIn('opponent_injury_impact', enriched.columns)
        self.assertIn('net_injury_advantage', enriched.columns)
        self.assertIn('injury_adjusted_win_probability', enriched.columns)
    
    def test_injury_adjusted_probability_calculated(self):
        """Test that adjusted probability is calculated."""
        game_data = pd.DataFrame({
            'team': ['Kansas City Chiefs'],
            'opponent': ['Buffalo Bills'],
            'week': [7],
            'win_probability': [0.65]
        })
        
        enriched = enrich_game_data_with_injuries(game_data, self.collector, self.analyzer)
        
        # Check that adjusted probability exists and is valid
        adj_prob = enriched.at[0, 'injury_adjusted_win_probability']
        self.assertGreaterEqual(adj_prob, 0.05)
        self.assertLessEqual(adj_prob, 0.95)
    
    def test_multiple_games(self):
        """Test enrichment with multiple games."""
        game_data = pd.DataFrame({
            'team': ['Kansas City Chiefs', 'Buffalo Bills', 'San Francisco 49ers'],
            'opponent': ['Buffalo Bills', 'Miami Dolphins', 'Dallas Cowboys'],
            'week': [7, 7, 7],
            'win_probability': [0.65, 0.72, 0.58]
        })
        
        enriched = enrich_game_data_with_injuries(game_data, self.collector, self.analyzer)
        
        # All games should be enriched
        self.assertEqual(len(enriched), 3)
        self.assertTrue(all(enriched['team_injury_impact'] >= 0))


class TestPositionAndSeverityWeights(unittest.TestCase):
    """Test that position and severity weights are reasonable."""
    
    def test_position_weights_exist(self):
        """Test that all common positions have weights."""
        required_positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        
        for pos in required_positions:
            self.assertIn(pos, POSITION_IMPACT_WEIGHTS)
    
    def test_qb_highest_weight(self):
        """Test that QB has the highest position weight."""
        qb_weight = POSITION_IMPACT_WEIGHTS['QB']
        
        for pos, weight in POSITION_IMPACT_WEIGHTS.items():
            if pos != 'QB':
                self.assertGreaterEqual(qb_weight, weight)
    
    def test_severity_weights_ordered(self):
        """Test that severity weights are properly ordered."""
        severities = ['OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE']
        weights = [INJURY_SEVERITY[s] for s in severities]
        
        # Should be in descending order
        for i in range(len(weights) - 1):
            self.assertGreaterEqual(weights[i], weights[i + 1])
    
    def test_weights_in_valid_range(self):
        """Test that all weights are in valid range [0, 1]."""
        for weight in POSITION_IMPACT_WEIGHTS.values():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)
        
        for weight in INJURY_SEVERITY.values():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)


if __name__ == '__main__':
    unittest.main()
