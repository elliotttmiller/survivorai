# 🏈 Survivor AI - Google Colab Notebook Guide (v2.0 Enterprise)

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

This notebook provides an **enterprise-grade, fully automated workflow**:

✅ **Pool Size Configuration** - Prominent input with automatic strategy selection  
✅ **Auto-Setup** - Clones repository, installs dependencies  
✅ **Real-Time Data** - Fetches live NFL odds and predictions  
✅ **Auto-Detection** - Detects current NFL season and week  
✅ **ML Predictions** - Ensemble models (Random Forest, XGBoost, Neural Networks)  
✅ **Optimization** - Hungarian algorithm for mathematically optimal picks  
✅ **Risk Analysis** - Monte Carlo simulation (10,000+ iterations)  
✅ **Recommendations** - Top 5 picks with complete season paths  
✅ **State Persistence** - Configuration saves between runs  
✅ **Weekly Updates** - Seamless workflow for recurring use  

---

## 📋 Prerequisites

### Required

- **Google Account** - To use Google Colab (free)
- **Internet Connection** - For data fetching

### Optional but Recommended

- **The Odds API Key** - For real-time betting odds
  - Get free at [the-odds-api.com](https://the-odds-api.com/)
  - Free tier: 500 requests/month
  - Without key: System uses ML predictions and SurvivorGrid data

---

## 🎯 How to Use

### First-Time Setup (5 minutes)

#### Step 1: Configure API Key (Optional but Recommended)

1. Visit [The Odds API](https://the-odds-api.com/) and sign up
2. Copy your API key
3. In Colab, click the 🔑 icon in the left sidebar
4. Click "Add new secret"
5. Name: `ODDS_API_KEY`
6. Value: Paste your API key
7. Enable "Notebook access"

This only needs to be done once - the key persists across sessions!

#### Step 2: Configure Pool Size

1. Run the **first cell** (Configuration section)
2. Enter your pool size (e.g., `150`)
3. The system automatically determines optimal strategy based on size:
   - Small (< 50): Conservative approach
   - Medium (50-200): Balanced strategy
   - Large (200-1000): Contrarian picks
   - Very Large (> 1000): Highly contrarian

Configuration is saved and persists between runs!

#### Step 3: Run All Cells

Click `Runtime` → `Run all` in the Colab menu.

The notebook will automatically:
1. Clone/update the Survivor AI repository
2. Install dependencies (~2-3 minutes first time, cached thereafter)
3. Set up the environment
4. Load your API key from secrets (if configured)
5. Detect current NFL season and week
6. Fetch real-time data
7. Load your used teams
8. Run optimization
9. Display recommendations

#### Step 4: Manage Used Teams

**Option A: File-based (Recommended)**
1. Open file browser (📁 icon in left sidebar)
2. Navigate to `/content/survivorai/used_teams.json`
3. Edit the file to add teams as you use them
4. Format: `"used_teams": ["Team Name 1", "Team Name 2"]`

**Option B: Interactive Cell**
- Use the "Weekly Update Helper" cell (Section 8)
- Enter team name when prompted
- File updates automatically

### Weekly Workflow (< 30 seconds)

1. **Open Notebook**: Click the Colab link or open from saved notebooks
2. **Run All**: `Runtime` → `Run all` (auto-detects current week)
3. **Review**: Check recommendations in results section
4. **Decide**: Choose your pick based on analysis
5. **Update**: Add team to `used_teams.json` or use update cell
6. **Export** (Optional): Download CSV for your records

No need to reconfigure - your pool size and API key are remembered!

---

## ⏱️ Performance

- **First Run**: 3-5 minutes (setup + dependencies)
- **Subsequent Runs**: 15-30 seconds (cached)
- **Data Fetching**: 5-10 seconds (real-time)
- **Optimization**: 5-15 seconds
- **Total**: ~30 seconds for weekly updates

---

## 📊 What You Get

### Summary Table
Quick overview of top 5 picks showing:
- Rank
- Team recommendation
- This week win probability
- Win-out probability (survive entire season)
- Pick percentage (popularity)

### Detailed Analysis (Top 3)
For each top recommendation:
- **Current Week Details**: Win probability, opponent, pick percentage
- **Pool-Adjusted Score**: Strategic score based on your pool size
- **Season Outlook**: Win-out probability, expected value
- **Risk Metrics**: Monte Carlo simulation results (if available)
- **Season Path**: Complete week-by-week recommendations through Week 18

### Export Options
- CSV export with all recommendations
- Season path export for planning
- Downloadable for offline review

---

## 🔧 Configuration Details

### Data Sources

The notebook intelligently combines multiple data sources:

1. **The Odds API** (if API key provided)
   - Real-time betting odds
   - Most accurate for current week
   - Hourly updates

2. **SurvivorGrid** (always used)
   - Consensus picks and percentages
   - Future week projections
   - Community insights

3. **ML Ensemble Models** (always used)
   - Random Forest, XGBoost, Neural Networks
   - Advanced feature engineering (49+ features)
   - Elo ratings, Pythagorean expectation
   - Historical performance data

### Auto-Detection Features

The system automatically:
- **Detects Current Season & Week** - Based on today's date
- **Identifies Available Teams** - Excludes your used teams
- **Selects Optimal Strategy** - Based on your pool size
- **Caches Data** - Reduces API calls and speeds up reruns
- **Validates Inputs** - Ensures data quality and consistency

---

## 🎯 Notebook Sections

### 1. Configuration & Pool Setup
- Pool size input with form-based UI
- Automatic strategy selection
- Configuration persistence

### 2. Setup & Installation
- Repository cloning/updating
- Dependency installation
- Environment configuration

### 3. Real-Time Data Collection
- NFL matchup data
- Betting odds (if API key available)
- Consensus picks and projections

### 4. Used Teams Management
- Load from JSON file
- File-based editing
- Interactive update helper

### 5. Optimization & Analysis
- Hungarian algorithm optimization
- Pool size strategy adjustment
- Monte Carlo risk analysis

### 6. Results & Recommendations
- Summary table
- Detailed analysis of top picks
- Season path visualization

### 7. Export & Save
- CSV export of recommendations
- Season path export
- Download instructions

### 8. Weekly Update Helper
- Quick team addition
- File management
- Status tracking

---

## 🔄 Advanced Usage

### Changing Pool Size

If your pool size changes mid-season:
1. Edit the first cell (Configuration section)
2. Update the `pool_size` parameter
3. Re-run all cells
4. New strategy will be applied automatically

### Managing Multiple Pools

For multiple survivor pools:
1. Create separate used_teams files: `used_teams_pool1.json`, etc.
2. Edit the `user_input` cell to load the appropriate file
3. Run optimization separately for each pool
4. Compare recommendations across pools

### Custom Team Constraints

To exclude teams temporarily (injuries, bye weeks, etc.):
1. Add them to `used_teams.json` for this week
2. Remove after the week passes
3. Or manually filter results

---

## 🛠️ Troubleshooting

### "Runtime disconnected"

**Solution**: Reconnect runtime and run all cells again. Your code is saved.

### "No data retrieved"

**Causes**:
- No API key provided (works without it using ML models)
- SurvivorGrid temporarily unavailable (fallback data used)
- Rate limit exceeded (free tier: 500 requests/month)

**Solution**: System automatically uses ML predictions and cached data.

### "Module not found"

**Solution**: Re-run the installation cell (Section 2). Dependencies are required.

### Slow Performance

**Tips**:
- First run is slower (3-5 minutes for setup)
- Subsequent runs use cached data (15-30 seconds)
- GPU not required (CPU is sufficient)
- Clear runtime if it becomes sluggish: `Runtime` → `Restart runtime`

### Used Teams Not Loading

**Solution**:
1. Check file path: `/content/survivorai/used_teams.json`
2. Verify JSON format is valid
3. Use the interactive update cell as backup
4. File is recreated if missing

---

## 💡 Tips for Best Results

### API Key Management

- Get free key from The Odds API for best accuracy
- Monitor usage (500 requests/month on free tier)
- Key only needed once per week
- Store in Colab secrets for security and convenience

### Pool Size Configuration

- Be accurate with your pool size
- Strategy automatically adjusts:
  - **Small (< 50)**: Conservative, high win probability
  - **Medium (50-200)**: Balanced approach
  - **Large (200-1000)**: Contrarian picks for differentiation
  - **Very Large (> 1000)**: Highly contrarian strategy

### Team Name Format

- Use exact NFL team names (e.g., "Kansas City Chiefs")
- Full names, not abbreviations
- Consistent spelling matters
- System validates against NFL teams list

### Making Your Decision

- Review all 5 recommendations, not just #1
- Consider your risk tolerance
- Factor in pool dynamics and competition
- Sometimes #2 or #3 is strategically better
- Balance win probability with pick percentage

---

## 📚 Additional Resources

### Documentation

- **README.md** - Project overview and quick start
- **ARCHITECTURE.md** - Technical architecture details
- **SETUP.md** - Local installation guide
- **FEATURES.md** - Complete feature list
- **NOTEBOOK_CHANGES.md** - Recent improvements

### External Links

- **GitHub**: [elliotttmiller/survivorai](https://github.com/elliotttmiller/survivorai)
- **The Odds API**: [the-odds-api.com](https://the-odds-api.com/)
- **Google Colab**: [colab.research.google.com](https://colab.research.google.com/)
- **Research Paper**: [NFL Win Prediction with ML](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1638446/full)

---

## 🎓 How It Works

### 1. Feature Engineering (49+ Features)

The system extracts comprehensive features per team/game:
- **Pythagorean Expectation**: NFL-optimized (exponent 2.37)
- **Elo Ratings**: Dynamic team strength with K-factor 20
- **Offensive Metrics**: Points/yards per game, efficiency
- **Defensive Metrics**: Points/yards allowed, success rate
- **Recent Form**: Last 3-5 games, momentum indicators
- **Rest Advantages**: Days since last game, bye weeks
- **Advanced Stats**: EPA, DVOA-proxy, explosive play rates

### 2. ML Ensemble Predictions

State-of-the-art ensemble of 5 models:
- **Random Forest**: Robust, feature importance analysis
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast, efficient gradient boosting
- **CatBoost**: Categorical feature handling
- **Neural Network**: Deep learning for pattern recognition

Models are weighted based on validation performance (R² > 0.75)

### 3. Hungarian Algorithm Optimization

Mathematically optimal team assignment:
- **Complexity**: O(n³) time with linear sum assignment
- **Guarantee**: Global optimum solution
- **Objective**: Maximize overall win-out probability
- **Constraint**: Each team used exactly once

### 4. Pool Size Strategy

Dynamic adjustments based on your pool:
- **Expected Value**: Calculates EV for each pick
- **Contrarian vs Consensus**: Balances safety with differentiation
- **Dynamic Weighting**: Adjusts scoring formula
- **Risk/Reward**: Optimizes for your pool dynamics

### 5. Monte Carlo Risk Analysis

Comprehensive simulation for confidence:
- **Iterations**: 10,000+ scenarios per path
- **Confidence Intervals**: 95% bounds on outcomes
- **Sensitivity Analysis**: How robust is each pick?
- **Variance Metrics**: Standard deviation and risk measures

---

## ⚡ Advanced Usage

### Custom Configuration

Edit parameters in cells for fine-tuning:

```python
# In configuration cell
pool_size = 250  # Your actual pool size

# In API config cell
os.environ['CACHE_EXPIRY_HOURS'] = '12'  # Longer cache period
```

### Scenario Analysis

Test different strategies:
1. Run with current used teams
2. Compare with alternative past picks
3. Evaluate "what if" scenarios
4. Test different pool sizes

### Export and Tracking

Maintain records across weeks:
- Export CSV each week
- Track actual vs predicted results
- Monitor model performance
- Adjust strategy based on patterns

---

## 🔒 Privacy & Security

### Data Handling

- **Local Computation**: All processing in your Colab instance
- **No External Storage**: Data not stored on external servers
- **API Key Security**: Only used for authorized data fetching
- **Session Privacy**: Results stay in your Colab environment

### API Key Best Practices

- Store in Colab secrets, never in code
- Never commit keys to version control
- Don't share keys with others
- Regenerate if compromised
- Monitor usage on The Odds API dashboard

---

## 📞 Support & Community

### Getting Help

1. **Check Documentation**: README, guides, and troubleshooting
2. **Review Examples**: Sample runs and expected outputs
3. **Search Issues**: Check if problem already addressed
4. **Open Issue**: Report bugs or request features on [GitHub](https://github.com/elliotttmiller/survivorai/issues)

### Contributing

Contributions welcome! See repository for:
- Code improvements
- Documentation updates
- Bug fixes
- Feature suggestions

---

## 🏆 Pro Tips for Success

### Strategic Principles

1. **Weekly Discipline**: Run analysis every week before games
2. **Risk Management**: Balance safety with differentiation
3. **Pool Awareness**: Monitor popular picks in your pool
4. **Long-Term Focus**: Optimize for full season, not just current week
5. **Backup Plans**: Always review multiple options

### Common Strategies

**Conservative (Small Pools)**
- Prioritize high win probability
- Save strong teams for later
- Avoid upset picks early
- Build confidence first

**Balanced (Medium Pools)**
- Mix safety with value
- Time contrarian picks strategically
- Monitor consensus picks
- Adjust week-to-week

**Contrarian (Large Pools)**
- Differentiate early and often
- Target low-owned quality teams
- Accept higher short-term risk
- Survive, then compete

---

## 📝 Version History

### v2.0 Enterprise (Current)
- ✅ Pool size configuration with automatic strategy
- ✅ Enhanced state management and persistence
- ✅ Improved notebook structure (8 clear sections)
- ✅ Better error handling and validation
- ✅ Professional output formatting
- ✅ Weekly update helper cell
- ✅ Comprehensive documentation
- ✅ Form-based interactive UI

### v1.0
- Initial automated workflow release
- ML pipeline integration (3 models)
- Real-time data fetching
- Auto-detection of season/week
- Basic Monte Carlo analysis
- Hungarian algorithm optimization

---

**Ready to dominate your survivor pool?** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elliotttmiller/survivorai/blob/main/SurvivorAI_Colab_Notebook.ipynb)

Click the badge above to get started! 🏈🏆

---

*Built with enterprise-grade code, cutting-edge ML research, and NFL analytics expertise*

**Survivor AI v2.0 Enterprise** - Maximize your chances of winning your survivor pool

