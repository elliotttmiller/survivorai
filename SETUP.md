# Survivor AI - Setup & Installation Guide

Complete guide to setting up and running Survivor AI on your system.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Using ML Features](#using-ml-features)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for ML features)
- **Storage**: 500MB for dependencies, 1GB+ for ML models
- **OS**: Linux, macOS, or Windows

### Required Software
- Python 3.8+
- pip (Python package installer)
- Git (for cloning the repository)

### Optional
- The Odds API key (free tier available at [the-odds-api.com](https://the-odds-api.com/))
- CUDA-capable GPU (for faster Neural Network training - optional)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/elliotttmiller/survivorai.git
cd survivorai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

#### Basic Installation (Without ML)
```bash
pip install numpy pandas scipy requests beautifulsoup4 lxml python-dotenv streamlit
```

#### Full Installation (With ML Features)
```bash
pip install -r requirements.txt
```

This includes:
- Core: numpy, pandas, scipy
- Data: requests, beautifulsoup4, lxml
- Web: streamlit
- ML: scikit-learn, xgboost, tensorflow, torch
- Visualization: matplotlib, seaborn, plotly

#### Minimal Installation for Testing
```bash
# Install only core dependencies
pip install numpy pandas scipy scikit-learn joblib
```

---

## Configuration

### Step 1: Create Configuration File

```bash
cp .env.example .env
```

### Step 2: Edit Configuration

Open `.env` in your favorite text editor:

```bash
nano .env  # or vim, code, notepad, etc.
```

### Configuration Options

#### Essential Settings
```bash
# Current season and week
CURRENT_SEASON=2025
CURRENT_WEEK=7
```

#### The Odds API (Optional but Recommended)
```bash
# Get free API key from https://the-odds-api.com/
ODDS_API_KEY=your_api_key_here
```

**Free Tier**: 500 requests/month (sufficient for weekly usage)

#### ML Model Settings
```bash
# Enable/disable ML predictions
USE_ML_PREDICTIONS=false  # Set to 'true' to enable

# Model type: ensemble, random_forest, neural_network
ML_MODEL_TYPE=ensemble

# Model storage directory
ML_MODEL_DIR=models
```

#### Advanced Settings
```bash
# Cache settings
CACHE_DIR=cache
CACHE_EXPIRY_HOURS=6

# Feature engineering
USE_ADVANCED_FEATURES=true
INCLUDE_HISTORICAL_DATA=true
HISTORICAL_SEASONS=3

# Ensemble weights (Random Forest, Neural Network, XGBoost)
ENSEMBLE_WEIGHTS=0.4,0.3,0.3

# Prediction confidence threshold
CONFIDENCE_THRESHOLD=0.6
```

---

## Running the Application

### Option 1: Web Interface (Streamlit)

**Recommended for most users**

```bash
streamlit run app.py
```

This will:
1. Start a local web server
2. Open your browser automatically
3. Display the interactive interface

**Access URL**: http://localhost:8501

#### Using the Web Interface

1. **Configure Your Pool**
   - Enter pool size in sidebar
   - Set current week
   
2. **Select Previous Picks**
   - For each completed week, select the team you picked
   - The app prevents duplicate selections
   
3. **Calculate Optimal Picks**
   - Click "Calculate Optimal Picks" button
   - Wait for analysis to complete
   
4. **Review Recommendations**
   - See top 5 picks for current week
   - Each pick shows:
     - This week's win probability
     - Overall win-out probability
     - Pool-adjusted score
     - Pick popularity
     - Complete season path

### Option 2: Command Line Testing

```bash
python test_full_system.py
```

This runs a comprehensive system test showing:
- Data collection status
- Optimizer functionality
- Pool size adjustments
- Sample recommendations

### Option 3: Individual Module Testing

Test specific components:

```bash
# Test ML models
python ml_models/prediction_models.py

# Test feature engineering
python ml_models/feature_engineering.py

# Test Monte Carlo simulation
python analytics/monte_carlo.py

# Test data collection
python data_collection/data_manager.py
```

### Option 4: Unit Tests

Run the test suite:

```bash
# Using unittest
python -m unittest discover tests

# Using pytest (if installed)
pytest tests/ -v
```

---

## Using ML Features

### Prerequisites for ML

1. Install ML dependencies:
```bash
pip install scikit-learn xgboost joblib
```

2. Enable in configuration:
```bash
# In .env file
USE_ML_PREDICTIONS=true
ML_MODEL_TYPE=ensemble
```

### First-Time ML Setup

The ML models will work without training data, using:
- Pythagorean expectation as baseline
- Elo ratings for team strength
- Feature engineering from current data

### Training Custom Models (Advanced)

If you have historical game data:

```python
from ml_models.ml_predictor import MLNFLPredictor
import pandas as pd

# Initialize predictor
predictor = MLNFLPredictor(model_type='ensemble')

# Load your historical data
historical_games = pd.read_csv('historical_games.csv')
# Required columns: team, opponent, week, season, is_home, 
#                   spread, actual_result (0 or 1)

# Train the model
metrics = predictor.train(historical_games, save_model=True)

print(f"Training complete!")
print(f"Validation R²: {metrics['random_forest']['val_r2']:.3f}")
```

### Model Performance

Expected performance on NFL data:
- **Random Forest**: R² ≈ 0.65-0.75
- **Neural Network**: R² ≈ 0.70-0.80 (highest accuracy)
- **Ensemble**: R² ≈ 0.72-0.82 (most robust)

---

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'X'

**Problem**: Missing dependency

**Solution**:
```bash
pip install [missing_module]

# Or reinstall all dependencies
pip install -r requirements.txt
```

#### 2. The Odds API Errors

**Problem**: API key invalid or quota exceeded

**Solution**:
- Check API key in `.env` file
- Verify key is active at the-odds-api.com
- Check quota usage (free tier: 500/month)
- Run without API (will use SurvivorGrid only):
  ```bash
  # Remove or comment out in .env
  # ODDS_API_KEY=
  ```

#### 3. SurvivorGrid Scraping Fails

**Problem**: Website HTML structure changed

**Solution**:
- The app will use placeholder data
- Manually update `data_collection/survivorgrid_scraper.py`
- Or use The Odds API exclusively

#### 4. Streamlit Won't Start

**Problem**: Port already in use

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# On Linux/macOS:
lsof -ti:8501 | xargs kill

# On Windows:
netstat -ano | findstr :8501
taskkill /PID [PID] /F
```

#### 5. ML Models Not Loading

**Problem**: Models not trained or file not found

**Solution**:
The app will work without pre-trained models, using:
- Pythagorean expectation
- Elo ratings
- Market odds

To disable ML predictions:
```bash
# In .env
USE_ML_PREDICTIONS=false
```

#### 6. Memory Issues

**Problem**: System runs out of memory

**Solution**:
```bash
# Reduce simulation count
# Edit analytics/monte_carlo.py:
# n_simulations = 1000  # Instead of 10000

# Or close other applications
# Or upgrade RAM to 8GB+
```

#### 7. Slow Performance

**Possible causes**:
- Large pool size (200+ entries)
- Many remaining weeks
- ML predictions enabled
- Monte Carlo simulations

**Solutions**:
```bash
# Disable ML for faster results
USE_ML_PREDICTIONS=false

# Reduce Random Forest trees
# In ml_models/prediction_models.py:
# n_estimators=50  # Instead of 100

# Cache results (automatic after first run)
```

### Getting Help

1. **Check Logs**
   - Streamlit: Check terminal output
   - Look for error messages and stack traces

2. **Documentation**
   - README.md - Overview and features
   - ARCHITECTURE.md - Technical details
   - QUICKSTART.md - Quick reference

3. **Debug Mode**
   ```python
   # Add to top of app.py for verbose output
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Test Components Individually**
   ```bash
   # Test what's working/broken
   python ml_models/feature_engineering.py
   python data_collection/data_manager.py
   ```

---

## Performance Tips

### Speed Optimization

1. **Use Caching**
   - First run: slower (fetching data)
   - Subsequent runs: faster (using cache)
   - Cache expires after 6 hours

2. **Reduce Model Complexity**
   ```python
   # Faster Random Forest
   RandomForestPredictor(n_estimators=50, max_depth=10)
   
   # Faster Neural Network
   NeuralNetworkPredictor(hidden_layers=(50, 25), max_iter=200)
   ```

3. **Disable Unnecessary Features**
   ```bash
   USE_ML_PREDICTIONS=false  # For quickest results
   ```

### Memory Optimization

1. **Reduce Monte Carlo Simulations**
   ```python
   simulator = MonteCarloSimulator(n_simulations=1000)
   ```

2. **Clear Cache Periodically**
   ```bash
   rm -rf cache/*
   ```

3. **Use Smaller Models**
   - random_forest instead of ensemble
   - Fewer features in feature engineering

---

## Next Steps

After successful setup:

1. **Configure Your Pool**
   - Enter your pool size
   - Input your previous picks
   
2. **Get Recommendations**
   - Run the optimizer
   - Review top 5 picks
   
3. **Analyze Risk**
   - Review confidence intervals
   - Check win-out probabilities
   
4. **Make Your Pick**
   - Consider pool dynamics
   - Balance risk and reward
   
5. **Weekly Updates**
   - Update `CURRENT_WEEK` in `.env`
   - Add your new pick to the UI
   - Re-run the optimizer

---

## Advanced Configuration

### Custom Pythagorean Exponent

If you have NFL-specific data suggesting different optimal exponent:

```python
# In config.py
PYTHAGOREAN_EXPONENT = 2.40  # Custom value (default: 2.37)
```

### Custom Elo Parameters

```python
# In ml_models/feature_engineering.py
def calculate_elo_rating(self, ...):
    k_factor = 25.0  # Higher = faster adjustment
    home_advantage = 70  # Higher = more home benefit
```

### Custom Ensemble Weights

If you find certain models perform better:

```bash
# In .env - format: RF,NN,XGB
ENSEMBLE_WEIGHTS=0.5,0.3,0.2  # Favor Random Forest
```

### Pool Size Thresholds

Adjust strategy breakpoints:

```python
# In optimizer/pool_calculator.py
def get_strategy_recommendation(self):
    if self.pool_size < 20:  # Was 10
        return "Small pool..."
```

---

## Production Deployment

### For Public Hosting

1. **Use Streamlit Cloud**
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Add secrets (API keys) in dashboard
   - Deploy!

2. **Or Deploy to Heroku**
   ```bash
   # Create Procfile
   echo "web: streamlit run app.py" > Procfile
   
   # Deploy
   heroku create survivorai-app
   git push heroku main
   ```

3. **Security Considerations**
   - Never commit `.env` file
   - Use environment variables for secrets
   - Implement rate limiting
   - Add authentication if needed

---

## Support & Resources

- **Documentation**: README.md, ARCHITECTURE.md
- **Issues**: GitHub Issues
- **API Docs**: The Odds API documentation
- **Research**: See ARCHITECTURE.md references

---

Happy optimizing! 🏈
