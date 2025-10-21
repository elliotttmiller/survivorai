"""
Integrated ML predictor for NFL Survivor Pool.

Combines feature engineering with machine learning models
to provide enhanced win probability predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import config
from ml_models.feature_engineering import NFLFeatureEngineer
from ml_models.prediction_models import EnsemblePredictor, RandomForestPredictor


class MLNFLPredictor:
    """
    Machine Learning predictor for NFL games.
    
    Integrates:
    - Feature engineering (Elo, Pythagorean, etc.)
    - ML models (Random Forest, Neural Network, XGBoost)
    - Ensemble predictions for robustness
    """
    
    def __init__(
        self, 
        model_type: str = 'ensemble',
        model_dir: Optional[str] = None
    ):
        """
        Initialize ML predictor.
        
        Args:
            model_type: Type of model ('ensemble', 'random_forest', 'neural_network', 'xgboost')
            model_dir: Directory for saving/loading models
        """
        self.model_type = model_type
        self.model_dir = model_dir or getattr(config, 'ML_MODEL_DIR', 'models')
        
        # Initialize feature engineer
        self.feature_engineer = NFLFeatureEngineer()
        
        # Initialize model based on type
        if model_type == 'ensemble':
            self.model = EnsemblePredictor()
        elif model_type == 'random_forest':
            self.model = RandomForestPredictor()
        else:
            # Default to random forest if specified model unavailable
            print(f"Warning: Model type '{model_type}' not fully implemented, using Random Forest")
            self.model = RandomForestPredictor()
        
        self.is_trained = False
        
        # Try to load pre-trained model
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """
        Try to load pre-trained model from disk.
        
        Returns:
            True if model loaded successfully
        """
        if not os.path.exists(self.model_dir):
            return False
        
        model_path = os.path.join(self.model_dir, f'{self.model_type}.pkl')
        
        if os.path.exists(model_path):
            try:
                success = self.model.load(model_path)
                if success:
                    self.is_trained = True
                    print(f"Loaded pre-trained {self.model_type} model")
                    return True
            except Exception as e:
                print(f"Error loading model: {e}")
        
        return False
    
    def prepare_training_data(
        self, 
        historical_games: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical games.
        
        Args:
            historical_games: DataFrame with historical game results
                Required columns: team, opponent, week, season, is_home, 
                                 spread, actual_result (0 or 1 for win/loss)
        
        Returns:
            Tuple of (feature_matrix, target_values)
        """
        # Extract features for all games
        feature_matrix, feature_names = self.feature_engineer.create_feature_matrix(
            historical_games
        )
        
        # Target is the actual result (or win probability if available)
        if 'actual_result' in historical_games.columns:
            target = historical_games['actual_result']
        elif 'win_probability' in historical_games.columns:
            target = historical_games['win_probability']
        else:
            raise ValueError("Historical games must have 'actual_result' or 'win_probability' column")
        
        return feature_matrix, target
    
    def train(
        self, 
        historical_games: pd.DataFrame,
        save_model: bool = True
    ) -> Dict:
        """
        Train the ML model on historical data.
        
        Args:
            historical_games: DataFrame with historical game data
            save_model: Whether to save trained model to disk
        
        Returns:
            Dictionary with training metrics
        """
        print(f"Preparing training data from {len(historical_games)} historical games...")
        X, y = self.prepare_training_data(historical_games)
        
        print(f"Training {self.model_type} model...")
        metrics = self.model.train(X, y)
        
        self.is_trained = True
        
        if save_model:
            self.save_model()
        
        return metrics
    
    def predict_game(
        self,
        team: str,
        opponent: str,
        week: int,
        season: int,
        is_home: bool,
        spread: Optional[float] = None
    ) -> Dict:
        """
        Predict win probability for a single game.
        
        Args:
            team: Team name
            opponent: Opponent name
            week: Week number
            season: Season year
            is_home: Whether team is home
            spread: Point spread (optional)
        
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_engineer.extract_comprehensive_features(
            team=team,
            opponent=opponent,
            week=week,
            season=season,
            is_home=is_home,
            spread=spread
        )
        
        # Convert to DataFrame for prediction
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        if self.is_trained:
            win_prob = self.model.predict(feature_df)[0]
            prediction_source = f'{self.model_type}_ml'
        else:
            # Fallback to Pythagorean expectation if model not trained
            win_prob = features.get('pythagorean_win_prob', 0.5)
            prediction_source = 'pythagorean_fallback'
        
        return {
            'team': team,
            'opponent': opponent,
            'week': week,
            'win_probability': float(win_prob),
            'prediction_source': prediction_source,
            'confidence': self._calculate_confidence(win_prob),
        }
    
    def predict_multiple_games(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict win probabilities for multiple games.
        
        Args:
            games_df: DataFrame with game information
                Required columns: team, opponent, week, season, is_home, spread
        
        Returns:
            DataFrame with predictions added
        """
        # Extract features
        feature_matrix, _ = self.feature_engineer.create_feature_matrix(games_df)
        
        # Make predictions
        if self.is_trained:
            predictions = self.model.predict(feature_matrix)
            prediction_source = f'{self.model_type}_ml'
        else:
            # Fallback to Pythagorean
            predictions = feature_matrix['pythagorean_win_prob'].values
            prediction_source = 'pythagorean_fallback'
        
        # Add to results
        result_df = games_df.copy()
        result_df['ml_win_probability'] = predictions
        result_df['ml_prediction_source'] = prediction_source
        result_df['ml_confidence'] = [
            self._calculate_confidence(p) for p in predictions
        ]
        
        return result_df
    
    def _calculate_confidence(self, win_prob: float) -> str:
        """
        Calculate confidence level for prediction.
        
        Args:
            win_prob: Predicted win probability
        
        Returns:
            Confidence level string
        """
        # Distance from 0.5 indicates confidence
        distance = abs(win_prob - 0.5)
        
        if distance > 0.3:
            return 'high'
        elif distance > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def enhance_data_manager_predictions(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enhance predictions from data manager with ML predictions.
        
        Args:
            data: DataFrame from data manager
        
        Returns:
            Enhanced DataFrame with ML predictions
        """
        if data.empty:
            return data
        
        # Get ML predictions
        enhanced = self.predict_multiple_games(data)
        
        # Blend with existing predictions if available
        if 'win_probability' in enhanced.columns and 'ml_win_probability' in enhanced.columns:
            # Weighted average: 70% ML, 30% original (if ML trained)
            if self.is_trained:
                enhanced['blended_win_probability'] = (
                    0.7 * enhanced['ml_win_probability'] +
                    0.3 * enhanced['win_probability']
                )
                # Use blended as final prediction
                enhanced['win_probability'] = enhanced['blended_win_probability']
            else:
                # If not trained, keep original
                pass
        elif 'ml_win_probability' in enhanced.columns:
            # No original probability, use ML
            enhanced['win_probability'] = enhanced['ml_win_probability']
        
        return enhanced
    
    def save_model(self) -> bool:
        """
        Save model to disk.
        
        Returns:
            True if successful
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.model_type == 'ensemble':
            return self.model.save(self.model_dir)
        else:
            model_path = os.path.join(self.model_dir, f'{self.model_type}.pkl')
            return self.model.save(model_path)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'model_dir': self.model_dir,
        }


def test_ml_predictor():
    """Test ML predictor functionality."""
    print("Testing ML NFL Predictor")
    print("=" * 60)
    
    predictor = MLNFLPredictor(model_type='random_forest')
    
    # Test single game prediction
    print("\n1. Single Game Prediction:")
    result = predictor.predict_game(
        team="Kansas City Chiefs",
        opponent="Buffalo Bills",
        week=7,
        season=2025,
        is_home=True,
        spread=-3.5
    )
    print(f"   Team: {result['team']}")
    print(f"   Win Probability: {result['win_probability']:.3f}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Source: {result['prediction_source']}")
    
    # Test multiple games
    print("\n2. Multiple Games Prediction:")
    games_df = pd.DataFrame([
        {
            'team': 'Dallas Cowboys',
            'opponent': 'Philadelphia Eagles',
            'week': 7,
            'season': 2025,
            'is_home': True,
            'spread': -1.5,
        },
        {
            'team': 'San Francisco 49ers',
            'opponent': 'Seattle Seahawks',
            'week': 7,
            'season': 2025,
            'is_home': False,
            'spread': 3.0,
        }
    ])
    
    predictions = predictor.predict_multiple_games(games_df)
    print(f"   Predicted {len(predictions)} games")
    for _, pred in predictions.iterrows():
        print(f"   {pred['team']}: {pred['ml_win_probability']:.3f}")
    
    print("\n✓ ML predictor test complete")


if __name__ == "__main__":
    test_ml_predictor()
