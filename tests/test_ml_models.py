"""Tests for ML models."""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.feature_engineering import NFLFeatureEngineer
from ml_models.prediction_models import (
    RandomForestPredictor, 
    NeuralNetworkPredictor, 
    EnsemblePredictor,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE
)

if LIGHTGBM_AVAILABLE:
    from ml_models.prediction_models import LightGBMPredictor

if CATBOOST_AVAILABLE:
    from ml_models.prediction_models import CatBoostPredictor


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
        
        # Check new advanced features
        self.assertIn('net_epa', features)
        self.assertIn('net_dvoa_proxy', features)
        self.assertIn('Kansas City Chiefs_success_rate', features)
        self.assertIn('Kansas City Chiefs_explosive_play_rate', features)
        
        # Check feature count (should be significantly more now)
        self.assertGreater(len(features), 40)
    
    def test_epa_calculation(self):
        """Test EPA calculation."""
        offensive_epa = self.engineer.calculate_epa_estimate("Team A", "offense")
        defensive_epa = self.engineer.calculate_epa_estimate("Team A", "defense")
        
        # EPA should be reasonable values (typically -1 to 1)
        self.assertGreaterEqual(offensive_epa, -2.0)
        self.assertLessEqual(offensive_epa, 2.0)
        self.assertGreaterEqual(defensive_epa, -2.0)
        self.assertLessEqual(defensive_epa, 2.0)
    
    def test_dvoa_proxy_calculation(self):
        """Test DVOA proxy calculation."""
        offensive_dvoa = self.engineer.calculate_dvoa_proxy("Team A", "offense")
        defensive_dvoa = self.engineer.calculate_dvoa_proxy("Team A", "defense")
        
        # DVOA proxy should be reasonable values (typically -0.5 to 0.5)
        self.assertGreaterEqual(offensive_dvoa, -1.0)
        self.assertLessEqual(offensive_dvoa, 1.0)
        self.assertGreaterEqual(defensive_dvoa, -1.0)
        self.assertLessEqual(defensive_dvoa, 1.0)
    
    def test_success_rate(self):
        """Test success rate calculation."""
        success_rate = self.engineer.calculate_success_rate("Team A")
        
        # Success rate should be between 0 and 1
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
    
    def test_explosive_play_rate(self):
        """Test explosive play rate calculation."""
        explosive_rate = self.engineer.calculate_explosive_play_rate("Team A")
        
        # Explosive play rate should be positive
        self.assertGreater(explosive_rate, 0.0)


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
        
        # Check that model names are available
        self.assertIsInstance(ensemble.model_names, list)
        self.assertGreaterEqual(len(ensemble.model_names), 2)
    
    @unittest.skipIf(not LIGHTGBM_AVAILABLE, "LightGBM not available")
    def test_lightgbm(self):
        """Test LightGBM model."""
        model = LightGBMPredictor(n_estimators=50)
        
        # Train
        metrics = model.train(self.X, self.y, validation_split=0.2)
        
        # Check metrics
        self.assertIn('val_mae', metrics)
        self.assertIn('val_r2', metrics)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X.head())
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))
    
    @unittest.skipIf(not CATBOOST_AVAILABLE, "CatBoost not available")
    def test_catboost(self):
        """Test CatBoost model."""
        model = CatBoostPredictor(iterations=50)
        
        # Train
        metrics = model.train(self.X, self.y, validation_split=0.2)
        
        # Check metrics
        self.assertIn('val_mae', metrics)
        self.assertIn('val_r2', metrics)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X.head())
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))
    
    def test_enhanced_ensemble_has_all_models(self):
        """Test that enhanced ensemble includes all available models."""
        ensemble = EnsemblePredictor()
        
        # Should have at least RF and NN
        self.assertIn('RandomForest', ensemble.model_names)
        self.assertIn('NeuralNetwork', ensemble.model_names)
        
        # Check if new models are included when available
        if LIGHTGBM_AVAILABLE:
            self.assertIn('LightGBM', ensemble.model_names)
        
        if CATBOOST_AVAILABLE:
            self.assertIn('CatBoost', ensemble.model_names)


if __name__ == '__main__':
    unittest.main()
