# 🏈 Survivor AI - Google Colab Notebook Guide

## Quick Start

### Option 1: Open Directly in Colab

Click this badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elliotttmiller/survivorai/blob/main/SurvivorAI_Colab_Notebook.ipynb)

### Option 2: Manual Upload

1. Download `SurvivorAI_Colab_Notebook.ipynb` from this repository
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click `File` → `Upload notebook`
4. Select the downloaded `.ipynb` file

---

## 🚀 Features

This notebook provides a **100% automated workflow**:

✅ **Auto-Setup** - Clones repository, installs dependencies  
✅ **Real-Time Data** - Fetches live NFL odds and predictions  
✅ **Auto-Detection** - Detects current NFL season and week  
✅ **ML Predictions** - Ensemble models (R² > 0.75)  
✅ **Optimization** - Hungarian algorithm for optimal picks  
✅ **Risk Analysis** - Monte Carlo simulation (10,000+ runs)  
✅ **Recommendations** - Top 5 picks with season paths  

---

## 📋 Prerequisites

### Required

- **Google Account** - To use Google Colab (free)
- **Internet Connection** - For data fetching

### Optional but Recommended

- **The Odds API Key** - For real-time betting odds
  - Get free at [the-odds-api.com](https://the-odds-api.com/)
  - Free tier: 500 requests/month
  - Without key: System uses SurvivorGrid + ML predictions only

---

## 🎯 How to Use

### Step 1: Open the Notebook

Click the "Open in Colab" badge above or upload the notebook manually.

### Step 2: Run All Cells

Click `Runtime` → `Run all` in the Colab menu.

The notebook will automatically:
1. Clone the Survivor AI repository
2. Install all dependencies (~2-3 minutes first time)
3. Set up the environment
4. Detect current NFL season and week

### Step 3: Configure

When prompted, enter:

1. **The Odds API Key** (optional)
   - Paste your API key, or press Enter to skip
   - System works without it using fallback data

2. **Pool Size**
   - Enter your pool size (e.g., `50`)

3. **Previous Picks**
   - Enter teams you've already used, one per line
   - Press Enter twice when done

### Step 4: View Results

The notebook will:
- Fetch real-time NFL data
- Run ML predictions
- Optimize team selections
- Generate top 5 recommendations
- Show complete season paths
- Provide risk analysis

### Step 5: Export (Optional)

Run the export cell to save results as CSV for reference.

---

## ⏱️ Performance

- **First Run**: 3-5 minutes (setup + dependencies)
- **Subsequent Runs**: < 1 minute (cached)
- **Data Fetching**: 10-30 seconds (real-time)
- **Optimization**: 5-15 seconds
- **Total**: < 2 minutes (after first setup)

---

## 📊 What You Get

### For Each Recommendation

- **This Week Win %** - Probability of winning this week
- **Win-Out Probability** - Chance of surviving entire season
- **Pool-Adjusted Score** - Strategic score based on pool size
- **Pick Popularity** - What % of pools are picking this team
- **Complete Season Path** - Week-by-week picks through Week 18

### Additional Analysis

- **Monte Carlo Simulation** - 10,000 scenarios, confidence intervals
- **Week-by-Week Survival** - Survival rates for each week
- **Risk Assessment** - Variance and confidence metrics
- **Strategy Guidance** - Tailored to your pool size

---

## 🔧 Configuration

### Data Sources

The notebook uses multiple data sources:

1. **The Odds API** (if key provided)
   - Real-time betting odds
   - Most accurate for current week
   - Updated hourly

2. **SurvivorGrid** (always used)
   - Consensus picks
   - Future week projections
   - Pick percentages

3. **ML Models** (always used)
   - Ensemble predictions
   - Feature engineering
   - 70% ML + 30% market blend

### Auto-Detection

The system automatically detects:
- **Current Season** - Based on today's date
- **Current Week** - Calculated from season start
- **Available Teams** - Based on your previous picks

---

## 🔄 Weekly Workflow

### Every Week

1. **Open the Notebook** - Same notebook, fresh data
2. **Run All Cells** - Auto-detects new week
3. **Add New Pick** - Include your latest pick
4. **Get Recommendations** - Fresh analysis

### Season Management

- Week detection is automatic
- No manual configuration needed
- System updates with real-time data

---

## 🛠️ Troubleshooting

### "Runtime disconnected"

**Solution**: Reconnect runtime and run all cells again. Your code is saved.

### "No data retrieved"

**Causes**:
- No API key provided (expected)
- SurvivorGrid temporarily unavailable
- Rate limit exceeded (free tier: 500/month)

**Solution**: System uses ML predictions and fallback data automatically.

### "Module not found"

**Solution**: Re-run the installation cell. Dependencies are required.

### Slow Performance

**Tips**:
- First run is slower (setup)
- Cached data speeds up subsequent runs
- GPU not required (CPU is fine)

---

## 💡 Tips for Best Results

### API Key

- Get free key for best results
- Monitor usage (500 requests/month free)
- Key only needed weekly (not daily)

### Pool Size

- Be accurate with pool size
- Strategy changes based on size:
  - Small (<50): Safety-first
  - Medium (50-200): Balanced
  - Large (200+): Contrarian

### Previous Picks

- Enter exact team names
- One team per line
- Order matters (chronological)

### Recommendations

- Review all 5 options
- Consider risk tolerance
- Factor in pool dynamics
- Don't always pick #1 (sometimes #2 or #3 is better)

---

## 📚 Additional Resources

### Documentation

- **README.md** - Project overview
- **ARCHITECTURE.md** - Technical details
- **SETUP.md** - Local installation guide
- **FEATURES.md** - Complete feature list

### Links

- **GitHub**: [elliotttmiller/survivorai](https://github.com/elliotttmiller/survivorai)
- **The Odds API**: [the-odds-api.com](https://the-odds-api.com/)
- **Google Colab**: [colab.research.google.com](https://colab.research.google.com/)

---

## 🎓 How It Works

### 1. Feature Engineering

Extracts 35+ features per game:
- Pythagorean expectation (NFL-optimized)
- Elo ratings
- Offensive/defensive metrics
- Recent form and momentum
- Rest advantages

### 2. ML Predictions

Ensemble of 3 models:
- Random Forest (40% weight)
- Neural Network (30% weight)
- XGBoost (30% weight)

### 3. Optimization

Hungarian algorithm:
- O(n³) complexity
- Guarantees global optimum
- Maximizes win-out probability

### 4. Pool Strategy

Size-aware adjustments:
- Expected value calculation
- Contrarian vs. consensus
- Dynamic weighting

### 5. Risk Analysis

Monte Carlo simulation:
- 10,000+ iterations
- 95% confidence intervals
- Sensitivity analysis
- Variance metrics

---

## ⚡ Advanced Usage

### Custom Configuration

Edit environment variables in the configuration cell:

```python
# Example customizations
os.environ['ML_MODEL_TYPE'] = 'random_forest'  # Use single model
os.environ['ENSEMBLE_WEIGHTS'] = '0.5,0.3,0.2'  # Custom weights
os.environ['CACHE_EXPIRY_HOURS'] = '12'  # Longer cache
```

### Export Results

Save recommendations for tracking:
- CSV export included
- Download from Colab files panel
- Compare week-over-week

### Multiple Runs

Try different scenarios:
- Various pool sizes
- Different previous picks
- What-if analysis

---

## 🔒 Privacy & Security

### Data Handling

- All computation in your Colab instance
- No data stored on external servers
- API key only used for data fetching
- Results stay in your Colab session

### API Key Safety

- Never share your API key
- Don't commit key to Git
- Use environment variables
- Regenerate if exposed

---

## 📞 Support

### Issues

Report bugs or issues on [GitHub Issues](https://github.com/elliotttmiller/survivorai/issues)

### Questions

- Check documentation first
- Review troubleshooting section
- Search existing issues
- Open new issue if needed

---

## 🏆 Success Tips

1. **Weekly Discipline** - Run every week consistently
2. **Risk Management** - Don't always pick highest probability
3. **Pool Awareness** - Consider what others are picking
4. **Long-Term Thinking** - Optimize for full season, not just this week
5. **Backup Plans** - Review multiple options before deciding

---

**Built with cutting-edge research and production-quality code** 🏈

*Survivor AI - Maximize your chances of winning your survivor pool*

---

## 📝 Version History

- **v1.0** - Initial release with automated workflow
- Full ML pipeline integration
- Real-time data fetching
- Auto-detection of season/week
- Monte Carlo risk analysis
- Complete optimization suite

---

**Ready to dominate your survivor pool?** Click the badge above to get started! 🏈🏆
