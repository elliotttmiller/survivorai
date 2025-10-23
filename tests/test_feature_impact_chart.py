"""
Test suite for Feature Impact Analysis chart functionality.

Tests the fix for the Feature Impact Analysis chart not displaying data.
The issue was caused by column name mismatch between data from DataManager
and the feature extraction in app.py.
"""
import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.model_explainer import ModelExplainer
from analytics.visualization import create_feature_contribution_chart
from data_collection.advanced_data_sources import integrate_all_data_sources


class TestFeatureImpactChart(unittest.TestCase):
    """Test cases for Feature Impact Analysis chart."""
    
    def setUp(self):
        """Set up test data."""
        # Create base data simulating DataManager output
        self.base_data = pd.DataFrame({
            'week': [7, 7, 8],
            'team': ['Kansas City Chiefs', 'Buffalo Bills', 'Kansas City Chiefs'],
            'opponent': ['Buffalo Bills', 'Kansas City Chiefs', 'Denver Broncos'],
            'win_probability': [0.72, 0.35, 0.78],
            'pick_pct': [0.35, 0.10, 0.28],
            'spread': [-4.5, 4.5, -7.0],
        })
        
        # Enhance with advanced metrics
        self.enhanced_data = integrate_all_data_sources(
            self.base_data,
            use_historical=True,
            use_advanced_metrics=True
        )
        
        self.explainer = ModelExplainer()
    
    def test_data_has_required_columns(self):
        """Test that enhanced data has the columns we need."""
        required_columns = ['team_elo', 'elo_win_probability', 'recent_win_pct', 'recent_point_diff']
        for col in required_columns:
            self.assertIn(col, self.enhanced_data.columns, 
                         f"Missing required column: {col}")
    
    def test_feature_extraction_with_correct_columns(self):
        """Test that feature extraction works with correct column names."""
        team_data = self.enhanced_data[self.enhanced_data['team'] == 'Kansas City Chiefs']
        self.assertFalse(team_data.empty, "Team data should not be empty")
        
        row = team_data.iloc[0]
        
        # Extract features using the fixed logic
        elo_rating = row.get('team_elo', row.get('elo_rating', 1500))
        pythagorean_win_prob = row.get('elo_win_probability', 
                                       row.get('pythagorean_win_prob', 0.5))
        
        spread_raw = row.get('spread', 0)
        spread_normalized = spread_raw / 14.0 if pd.notna(spread_raw) else 0
        
        recent_win_pct = row.get('recent_win_pct', 0.5)
        recent_point_diff = row.get('recent_point_diff', 0)
        recent_form = (recent_win_pct - 0.5) * 2 + (recent_point_diff / 14.0)
        
        features = {
            'elo_rating': elo_rating,
            'pythagorean_win_prob': pythagorean_win_prob,
            'spread_normalized': spread_normalized,
            'home_advantage': 0,
            'recent_form': recent_form,
            'rest_advantage': 0
        }
        
        # Verify all features are populated with non-default values
        self.assertNotEqual(features['elo_rating'], 1500, 
                           "elo_rating should not be default value")
        self.assertNotEqual(features['pythagorean_win_prob'], 0.5,
                           "pythagorean_win_prob should not be default value")
    
    def test_feature_contributions_are_generated(self):
        """Test that feature contributions are properly generated."""
        team_data = self.enhanced_data[self.enhanced_data['team'] == 'Kansas City Chiefs']
        row = team_data.iloc[0]
        
        # Extract features
        elo_rating = row.get('team_elo', 1500)
        pythagorean_win_prob = row.get('elo_win_probability', 0.5)
        spread_normalized = row.get('spread', 0) / 14.0
        recent_win_pct = row.get('recent_win_pct', 0.5)
        recent_point_diff = row.get('recent_point_diff', 0)
        recent_form = (recent_win_pct - 0.5) * 2 + (recent_point_diff / 14.0)
        
        features = {
            'elo_rating': elo_rating,
            'pythagorean_win_prob': pythagorean_win_prob,
            'spread_normalized': spread_normalized,
            'home_advantage': 0,
            'recent_form': recent_form,
            'rest_advantage': 0
        }
        
        # Generate explanation
        explanation = self.explainer.explain_prediction(
            team='Kansas City Chiefs',
            opponent='Buffalo Bills',
            week=7,
            win_probability=0.72,
            features=features,
            spread=-4.5
        )
        
        # Verify feature contributions exist
        self.assertIn('feature_contributions', explanation)
        self.assertIsInstance(explanation['feature_contributions'], list)
        self.assertGreater(len(explanation['feature_contributions']), 0,
                          "Feature contributions should not be empty")
    
    def test_feature_chart_creation(self):
        """Test that the feature contribution chart can be created with data."""
        team_data = self.enhanced_data[self.enhanced_data['team'] == 'Kansas City Chiefs']
        row = team_data.iloc[0]
        
        features = {
            'elo_rating': row.get('team_elo', 1500),
            'pythagorean_win_prob': row.get('elo_win_probability', 0.5),
            'spread_normalized': row.get('spread', 0) / 14.0,
            'home_advantage': 0,
            'recent_form': (row.get('recent_win_pct', 0.5) - 0.5) * 2,
            'rest_advantage': 0
        }
        
        explanation = self.explainer.explain_prediction(
            team='Kansas City Chiefs',
            opponent='Buffalo Bills',
            week=7,
            win_probability=0.72,
            features=features
        )
        
        # Create chart
        chart = create_feature_contribution_chart(explanation['feature_contributions'])
        
        # Verify chart has data
        self.assertIsNotNone(chart, "Chart should not be None")
        self.assertTrue(len(chart.data) > 0, "Chart should have data traces")
        self.assertTrue(len(chart.data[0].y) > 0, "Chart should have data points")
    
    def test_multiple_teams(self):
        """Test that feature extraction works for multiple teams."""
        for team in self.enhanced_data['team'].unique():
            team_data = self.enhanced_data[self.enhanced_data['team'] == team]
            self.assertFalse(team_data.empty, f"Data for {team} should not be empty")
            
            row = team_data.iloc[0]
            features = {
                'elo_rating': row.get('team_elo', 1500),
                'pythagorean_win_prob': row.get('elo_win_probability', 0.5),
                'spread_normalized': row.get('spread', 0) / 14.0,
                'home_advantage': 0,
                'recent_form': (row.get('recent_win_pct', 0.5) - 0.5) * 2,
                'rest_advantage': 0
            }
            
            explanation = self.explainer.explain_prediction(
                team=team,
                opponent=row.get('opponent', 'Unknown'),
                week=int(row.get('week', 7)),
                win_probability=float(row.get('win_probability', 0.5)),
                features=features
            )
            
            self.assertGreater(len(explanation['feature_contributions']), 0,
                             f"Feature contributions for {team} should not be empty")


if __name__ == '__main__':
    unittest.main()
