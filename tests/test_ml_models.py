"""Tests for ML models."""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.feature_engineering import NFLFeatureEngineer
from ml_models.prediction_models import RandomForestPredictor, NeuralNetworkPredictor, EnsemblePredictor


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = NFLFeatureEngineer()
    
    def test_pythagorean_expectation(self):
        """Test Pythagorean expectation calculation."""
        # Test balanced teams
        prob = self.engineer.calculate_pythagorean_expectation(24.0, 24.0)
        self.assertAlmostEqual(prob, 0.5, places=2)
        
        # Test dominant team
        prob = self.engineer.calculate_pythagorean_expectation(30.0, 20.0)
        self.assertGreater(prob, 0.6)
        
        # Test underdog
        prob = self.engineer.calculate_pythagorean_expectation(20.0, 30.0)
        self.assertLess(prob, 0.4)
    
    def test_elo_rating(self):
        """Test Elo rating calculation."""
        elo = self.engineer.calculate_elo_rating("Team A", "Team B")
        
        self.assertIn('team_elo', elo)
        self.assertIn('opponent_elo', elo)
        self.assertIn('elo_win_prob', elo)
        
        # Win probability should be between 0 and 1
        self.assertGreaterEqual(elo['elo_win_prob'], 0.0)
        self.assertLessEqual(elo['elo_win_prob'], 1.0)
    
    def test_comprehensive_features(self):
        """Test comprehensive feature extraction."""
        features = self.engineer.extract_comprehensive_features(
            team="Kansas City Chiefs",
            opponent="Buffalo Bills",
            week=7,
            season=2025,
            is_home=True,
            spread=-3.5
        )
        
        # Check that required features exist
        self.assertIn('week', features)
        self.assertIn('is_home', features)
        self.assertIn('spread', features)
        self.assertIn('elo_win_prob', features)
        self.assertIn('pythagorean_win_prob', features)
        
        # Check feature count (should have many features)
        self.assertGreater(len(features), 20)


class TestPredictionModels(unittest.TestCase):
    """Test ML prediction models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate target (win probabilities)
        weights = np.random.randn(n_features)
        self.y = pd.Series(1 / (1 + np.exp(-self.X.dot(weights))))
    
    def test_random_forest(self):
        """Test Random Forest model."""
        model = RandomForestPredictor(n_estimators=10)
        
        # Train
        metrics = model.train(self.X, self.y, validation_split=0.2)
        
        # Check metrics exist
        self.assertIn('val_mae', metrics)
        self.assertIn('val_r2', metrics)
        
        # Check model is trained
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X.head())
        self.assertEqual(len(predictions), 5)
        
        # Predictions should be probabilities
        self.assertTrue(all(0 <= p <= 1 for p in predictions))
    
    def test_neural_network(self):
        """Test Neural Network model."""
        model = NeuralNetworkPredictor(
            hidden_layers=(20, 10),
            max_iter=100
        )
        
        # Train
        metrics = model.train(self.X, self.y, validation_split=0.2)
        
        # Check metrics
        self.assertIn('val_mae', metrics)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X.head())
        self.assertEqual(len(predictions), 5)
    
    def test_ensemble(self):
        """Test Ensemble model."""
        ensemble = EnsemblePredictor()
        
        # Train
        metrics = ensemble.train(self.X, self.y, validation_split=0.2)
        
        # Check all models trained
        self.assertIn('random_forest', metrics)
        self.assertIn('neural_network', metrics)
        self.assertTrue(ensemble.is_trained)
        
        # Test prediction
        predictions = ensemble.predict(self.X.head())
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))


if __name__ == '__main__':
    unittest.main()
