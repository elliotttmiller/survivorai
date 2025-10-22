"""
Machine Learning prediction models for NFL game outcomes.

Based on research from:
- Frontiers in Sports: Random Forest and Neural Network superiority over traditional models
- Real-time prediction with ensemble methods
- XGBoost for gradient boosting performance

Implements:
1. Random Forest Regression - High accuracy, interpretable
2. Neural Network - Best performance in research
3. XGBoost - Fast, handles non-linear patterns
4. Ensemble Model - Combines all three for robustness
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import joblib
import os

# Scikit-learn models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")


class BasePredictor(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, model_name: str):
        """
        Initialize base predictor.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model."""
        pass
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target values (win probabilities)
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model if not exists
        if self.model is None:
            self.model = self.build_model()
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_r2': r2_score(y_val, y_val_pred),
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted win probabilities
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to [0, 1] for probabilities
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def save(self, filepath: str) -> bool:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if successful
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'model_name': self.model_name,
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class RandomForestPredictor(BasePredictor):
    """
    Random Forest predictor.
    
    Based on research showing Random Forest as highly effective for NFL prediction.
    Advantages:
    - Handles non-linear relationships
    - Robust to outliers
    - Feature importance analysis
    - Good generalization
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        Initialize Random Forest predictor.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None for unlimited)
        """
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def build_model(self) -> RandomForestRegressor:
        """Build Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            min_samples_split=5,
            min_samples_leaf=2,
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        return importance_df.sort_values('importance', ascending=False)


class NeuralNetworkPredictor(BasePredictor):
    """
    Neural Network predictor.
    
    Research shows Neural Networks achieve highest accuracy for NFL prediction.
    Uses Multi-Layer Perceptron with:
    - Hidden layers for pattern learning
    - Regularization to prevent overfitting
    - Adam optimizer for fast convergence
    """
    
    def __init__(
        self, 
        hidden_layers: Tuple[int, ...] = (100, 50, 25),
        learning_rate: float = 0.001,
        max_iter: int = 1000
    ):
        """
        Initialize Neural Network predictor.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Initial learning rate
            max_iter: Maximum iterations
        """
        super().__init__("NeuralNetwork")
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    
    def build_model(self) -> MLPRegressor:
        """Build Neural Network model."""
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.0001,  # L2 regularization
        )


if XGBOOST_AVAILABLE:
    class XGBoostPredictor(BasePredictor):
        """
        XGBoost predictor.
        
        Gradient boosting for:
        - Fast training and prediction
        - Handles missing values
        - Built-in regularization
        - Excellent performance on tabular data
        """
        
        def __init__(
            self, 
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 6
        ):
            """
            Initialize XGBoost predictor.
            
            Args:
                n_estimators: Number of boosting rounds
                learning_rate: Boosting learning rate
                max_depth: Maximum tree depth
            """
            super().__init__("XGBoost")
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
        
        def build_model(self) -> xgb.XGBRegressor:
            """Build XGBoost model."""
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
            )
else:
    # Placeholder when XGBoost not available
    XGBoostPredictor = None


if LIGHTGBM_AVAILABLE:
    class LightGBMPredictor(BasePredictor):
        """
        LightGBM predictor.
        
        State-of-the-art gradient boosting optimized for:
        - Speed: 5-10x faster than XGBoost
        - Memory efficiency: Histogram-based algorithms
        - Accuracy: Leaf-wise tree growth for better performance
        - Large datasets: Handles millions of samples efficiently
        
        Based on research showing LightGBM's superiority for tabular data
        and competitive sports analytics applications.
        """
        
        def __init__(
            self, 
            n_estimators: int = 100,
            learning_rate: float = 0.05,
            num_leaves: int = 31,
            max_depth: int = -1
        ):
            """
            Initialize LightGBM predictor.
            
            Args:
                n_estimators: Number of boosting rounds
                learning_rate: Boosting learning rate
                num_leaves: Maximum number of leaves per tree
                max_depth: Maximum tree depth (-1 for unlimited)
            """
            super().__init__("LightGBM")
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.num_leaves = num_leaves
            self.max_depth = max_depth
        
        def build_model(self) -> lgb.LGBMRegressor:
            """Build LightGBM model."""
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
                objective='regression',
                metric='rmse',
                verbosity=-1,  # Suppress warnings
            )
else:
    # Placeholder when LightGBM not available
    LightGBMPredictor = None


if CATBOOST_AVAILABLE:
    class CatBoostPredictor(BasePredictor):
        """
        CatBoost predictor.
        
        Advanced gradient boosting with:
        - Best categorical feature handling (no preprocessing needed)
        - Ordered boosting to reduce prediction shift
        - Robust to overfitting with built-in regularization
        - Minimal hyperparameter tuning required
        
        Research shows CatBoost excels with mixed feature types and
        provides superior generalization on sports prediction tasks.
        """
        
        def __init__(
            self, 
            iterations: int = 100,
            learning_rate: float = 0.05,
            depth: int = 6
        ):
            """
            Initialize CatBoost predictor.
            
            Args:
                iterations: Number of boosting iterations
                learning_rate: Boosting learning rate
                depth: Tree depth
            """
            super().__init__("CatBoost")
            self.iterations = iterations
            self.learning_rate = learning_rate
            self.depth = depth
        
        def build_model(self) -> cb.CatBoostRegressor:
            """Build CatBoost model."""
            return cb.CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_state=42,
                verbose=False,  # Suppress training output
                loss_function='RMSE',
            )
else:
    # Placeholder when CatBoost not available
    CatBoostPredictor = None


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.
    
    Research shows ensemble methods provide:
    - Better generalization
    - Reduced variance
    - More robust predictions
    
    Enhanced version includes 5 models:
    - Random Forest: Robust, interpretable
    - Neural Network: Complex patterns
    - XGBoost: Fast gradient boosting
    - LightGBM: Efficient, accurate (NEW)
    - CatBoost: Best categorical handling (NEW)
    """
    
    def __init__(self, weights: Optional[Tuple[float, ...]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            weights: Weights for (RF, NN, XGB, LGBM, CatBoost). If None, uses equal weights.
        """
        # Initialize sub-models
        self.rf_model = RandomForestPredictor(n_estimators=100)
        self.nn_model = NeuralNetworkPredictor(hidden_layers=(100, 50, 25))
        
        # Count available advanced models
        available_models = 2  # RF and NN always available
        
        if XGBOOST_AVAILABLE and XGBoostPredictor is not None:
            self.xgb_model = XGBoostPredictor(n_estimators=100)
            available_models += 1
        else:
            self.xgb_model = None
        
        if LIGHTGBM_AVAILABLE and LightGBMPredictor is not None:
            self.lgbm_model = LightGBMPredictor(n_estimators=100)
            available_models += 1
        else:
            self.lgbm_model = None
        
        if CATBOOST_AVAILABLE and CatBoostPredictor is not None:
            self.catboost_model = CatBoostPredictor(iterations=100)
            available_models += 1
        else:
            self.catboost_model = None
        
        # Set weights based on available models
        if weights is not None:
            self.weights = weights
        else:
            # Equal weights for all available models
            equal_weight = 1.0 / available_models
            self.weights = tuple([equal_weight] * available_models)
        
        self.is_trained = False
        self.model_names = self._get_model_names()
    
    def _get_model_names(self) -> List[str]:
        """Get list of available model names."""
        names = ['RandomForest', 'NeuralNetwork']
        if self.xgb_model is not None:
            names.append('XGBoost')
        if self.lgbm_model is not None:
            names.append('LightGBM')
        if self.catboost_model is not None:
            names.append('CatBoost')
        return names
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train all sub-models.
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Validation split fraction
            
        Returns:
            Dictionary with training metrics for all models
        """
        print("Training Random Forest...")
        rf_metrics = self.rf_model.train(X, y, validation_split)
        
        print("Training Neural Network...")
        nn_metrics = self.nn_model.train(X, y, validation_split)
        
        xgb_metrics = {}
        if self.xgb_model is not None:
            print("Training XGBoost...")
            xgb_metrics = self.xgb_model.train(X, y, validation_split)
        
        lgbm_metrics = {}
        if self.lgbm_model is not None:
            print("Training LightGBM...")
            lgbm_metrics = self.lgbm_model.train(X, y, validation_split)
        
        catboost_metrics = {}
        if self.catboost_model is not None:
            print("Training CatBoost...")
            catboost_metrics = self.catboost_model.train(X, y, validation_split)
        
        self.is_trained = True
        
        return {
            'random_forest': rf_metrics,
            'neural_network': nn_metrics,
            'xgboost': xgb_metrics,
            'lightgbm': lgbm_metrics,
            'catboost': catboost_metrics,
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble models not trained yet")
        
        # Collect predictions from all available models
        predictions = []
        active_weights = []
        
        # Always available
        predictions.append(self.rf_model.predict(X))
        active_weights.append(self.weights[0])
        
        predictions.append(self.nn_model.predict(X))
        active_weights.append(self.weights[1])
        
        # Optional models
        weight_idx = 2
        if self.xgb_model is not None:
            predictions.append(self.xgb_model.predict(X))
            active_weights.append(self.weights[weight_idx])
            weight_idx += 1
        
        if self.lgbm_model is not None:
            predictions.append(self.lgbm_model.predict(X))
            active_weights.append(self.weights[weight_idx])
            weight_idx += 1
        
        if self.catboost_model is not None:
            predictions.append(self.catboost_model.predict(X))
            active_weights.append(self.weights[weight_idx])
            weight_idx += 1
        
        # Normalize weights to sum to 1
        total_weight = sum(active_weights)
        normalized_weights = [w / total_weight for w in active_weights]
        
        # Weighted average
        ensemble_pred = sum(
            w * pred for w, pred in zip(normalized_weights, predictions)
        )
        
        # Ensure probabilities in [0, 1]
        ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
        
        return ensemble_pred
    
    def save(self, directory: str) -> bool:
        """
        Save all models to directory.
        
        Args:
            directory: Directory path
            
        Returns:
            True if successful
        """
        os.makedirs(directory, exist_ok=True)
        
        success = True
        success &= self.rf_model.save(os.path.join(directory, 'rf_model.pkl'))
        success &= self.nn_model.save(os.path.join(directory, 'nn_model.pkl'))
        
        if self.xgb_model is not None:
            success &= self.xgb_model.save(os.path.join(directory, 'xgb_model.pkl'))
        
        if self.lgbm_model is not None:
            success &= self.lgbm_model.save(os.path.join(directory, 'lgbm_model.pkl'))
        
        if self.catboost_model is not None:
            success &= self.catboost_model.save(os.path.join(directory, 'catboost_model.pkl'))
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'is_trained': self.is_trained,
            'model_names': self.model_names,
        }
        joblib.dump(metadata, os.path.join(directory, 'ensemble_meta.pkl'))
        
        return success
    
    def load(self, directory: str) -> bool:
        """
        Load all models from directory.
        
        Args:
            directory: Directory path
            
        Returns:
            True if successful
        """
        try:
            self.rf_model.load(os.path.join(directory, 'rf_model.pkl'))
            self.nn_model.load(os.path.join(directory, 'nn_model.pkl'))
            
            if self.xgb_model is not None and os.path.exists(
                os.path.join(directory, 'xgb_model.pkl')
            ):
                self.xgb_model.load(os.path.join(directory, 'xgb_model.pkl'))
            
            if self.lgbm_model is not None and os.path.exists(
                os.path.join(directory, 'lgbm_model.pkl')
            ):
                self.lgbm_model.load(os.path.join(directory, 'lgbm_model.pkl'))
            
            if self.catboost_model is not None and os.path.exists(
                os.path.join(directory, 'catboost_model.pkl')
            ):
                self.catboost_model.load(os.path.join(directory, 'catboost_model.pkl'))
            
            # Load ensemble metadata
            metadata = joblib.load(os.path.join(directory, 'ensemble_meta.pkl'))
            self.weights = metadata['weights']
            self.is_trained = metadata['is_trained']
            if 'model_names' in metadata:
                self.model_names = metadata['model_names']
            
            return True
        except Exception as e:
            print(f"Error loading ensemble: {e}")
            return False


def test_models():
    """Test all prediction models."""
    print("Testing NFL Prediction Models")
    print("=" * 60)
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Synthetic target (win probabilities)
    # Based on linear combination + noise
    true_weights = np.random.randn(n_features)
    y = 1 / (1 + np.exp(-X.dot(true_weights)))  # Sigmoid for [0,1]
    y = pd.Series(y)
    
    print(f"\nTraining data: {n_samples} samples, {n_features} features")
    
    # Test Random Forest
    print("\n1. Testing Random Forest:")
    rf = RandomForestPredictor(n_estimators=50)
    rf_metrics = rf.train(X, y, validation_split=0.2)
    print(f"   Val MAE: {rf_metrics['val_mae']:.4f}")
    print(f"   Val R²: {rf_metrics['val_r2']:.4f}")
    
    # Test Neural Network
    print("\n2. Testing Neural Network:")
    nn = NeuralNetworkPredictor(hidden_layers=(50, 25), max_iter=200)
    nn_metrics = nn.train(X, y, validation_split=0.2)
    print(f"   Val MAE: {nn_metrics['val_mae']:.4f}")
    print(f"   Val R²: {nn_metrics['val_r2']:.4f}")
    
    # Test XGBoost if available
    if XGBOOST_AVAILABLE:
        print("\n3. Testing XGBoost:")
        xgb_model = XGBoostPredictor(n_estimators=50)
        xgb_metrics = xgb_model.train(X, y, validation_split=0.2)
        print(f"   Val MAE: {xgb_metrics['val_mae']:.4f}")
        print(f"   Val R²: {xgb_metrics['val_r2']:.4f}")
    
    # Test Ensemble
    print("\n4. Testing Ensemble:")
    ensemble = EnsemblePredictor()
    ensemble_metrics = ensemble.train(X, y, validation_split=0.2)
    print("   All models trained successfully")
    
    # Test predictions
    X_test = X.iloc[:5]
    predictions = ensemble.predict(X_test)
    print(f"\n   Sample predictions: {predictions}")
    
    print("\n✓ All model tests complete")


if __name__ == "__main__":
    test_models()
