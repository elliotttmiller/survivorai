"""Configuration settings for NFL Survivor Pool Optimizer."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'

# NFL Configuration
NFL_TEAMS = [
    'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
    'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
    'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
    'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
    'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
    'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
    'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
    'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
]

# Team abbreviation mappings (for different data sources)
TEAM_ABBREV_MAP = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}

# Season Configuration
CURRENT_SEASON = int(os.getenv('CURRENT_SEASON', 2025))
CURRENT_WEEK = int(os.getenv('CURRENT_WEEK', 7))
TOTAL_WEEKS = 18  # Regular season weeks

# Data Sources
SURVIVORGRID_URL = 'https://www.survivorgrid.com/'

# Cache settings
CACHE_DIR = 'cache'
CACHE_EXPIRY_HOURS = 6

# ML Model Configuration
ML_MODEL_DIR = os.getenv('ML_MODEL_DIR', 'models')
ML_MODEL_TYPE = os.getenv('ML_MODEL_TYPE', 'ensemble')  # Options: random_forest, neural_network, xgboost, ensemble
USE_ML_PREDICTIONS = os.getenv('USE_ML_PREDICTIONS', 'false').lower() == 'true'

# Feature Engineering
USE_ADVANCED_FEATURES = os.getenv('USE_ADVANCED_FEATURES', 'true').lower() == 'true'
INCLUDE_HISTORICAL_DATA = os.getenv('INCLUDE_HISTORICAL_DATA', 'true').lower() == 'true'
HISTORICAL_SEASONS = int(os.getenv('HISTORICAL_SEASONS', '3'))

# Prediction Configuration
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
ENSEMBLE_WEIGHTS = tuple(map(float, os.getenv('ENSEMBLE_WEIGHTS', '0.4,0.3,0.3').split(',')))

# Pythagorean Expectation (NFL-optimized)
PYTHAGOREAN_EXPONENT = 2.37  # Research-backed optimal value for NFL
