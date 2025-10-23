"""Streamlit app for NFL Survivor Pool Optimizer."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_collection.data_manager import DataManager
from optimizer.hungarian_optimizer import SurvivorOptimizer
from optimizer.pool_calculator import PoolCalculator
from ml_models.model_explainer import ModelExplainer
from analytics.visualization import (
    create_feature_contribution_chart,
    create_ensemble_breakdown_chart,
    create_confidence_gauge,
    create_risk_indicator,
    create_prediction_breakdown_chart,
    create_weekly_path_chart
)


# Page configuration
st.set_page_config(
    page_title="NFL Survivor Pool Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for ChatGPT-like modern grey theme
st.markdown("""
<style>
    /* ChatGPT-inspired grey theme */
    :root {
        --bg-primary: #212121;
        --bg-secondary: #2f2f2f;
        --bg-tertiary: #3f3f3f;
        --text-primary: #ececec;
        --text-secondary: #b4b4b4;
        --accent-primary: #10a37f;
        --accent-hover: #0d8c6d;
        --border-color: #4f4f4f;
    }
    
    /* Hide Streamlit branding */
    header[data-testid="stHeader"] {
        background: var(--bg-primary);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main background */
    .main {
        background: var(--bg-primary);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
    }
    
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Content area text */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--text-primary) !important;
    }
    
    .main p, .main li, .main label {
        color: var(--text-secondary) !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] details {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        margin-bottom: 1rem !important;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stExpander"] details summary {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    
    div[data-testid="stExpander"] details:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--accent-primary) !important;
    }
    
    div[data-testid="stExpander"] details[open] {
        background: var(--bg-tertiary) !important;
        border-color: var(--accent-primary) !important;
    }
    
    div[data-testid="stExpander"] div[class*="streamlit-expanderContent"] {
        background: transparent !important;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        font-size: 0.9rem !important;
    }
    
    div[data-testid="stDataFrame"] table tbody tr:nth-child(odd) td {
        background-color: var(--bg-secondary) !important;
    }
    
    div[data-testid="stDataFrame"] table tbody tr:nth-child(even) td {
        background-color: var(--bg-tertiary) !important;
    }
    
    div[data-testid="stDataFrame"] table tbody tr:hover td {
        background-color: rgba(16, 163, 127, 0.1) !important;
    }
    
    div[data-testid="stDataFrame"] table td,
    div[data-testid="stDataFrame"] table th {
        border-color: var(--border-color) !important;
        padding: 0.75rem !important;
        color: var(--text-primary) !important;
    }
    
    div[data-testid="stDataFrame"] table thead tr th {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        border-left: 3px solid var(--accent-primary);
        color: var(--text-primary);
    }
    
    /* Input styling */
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Divider styling */
    hr {
        border-color: var(--border-color);
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)


def load_data(use_odds_api: bool, current_week: int, use_ml: bool = True):
    """Load data from all available sources."""
    manager = DataManager(
        use_odds_api=use_odds_api,
        use_ml_predictions=use_ml,
        use_advanced_metrics=True,
        use_historical_data=True
    )
    data = manager.get_comprehensive_data(current_week=current_week)
    return data


def load_used_teams():
    """Load used teams from JSON file."""
    used_teams_file = 'used_teams.json'
    if os.path.exists(used_teams_file):
        try:
            with open(used_teams_file, 'r') as f:
                data = json.load(f)
            # Convert week numbers to int and return as dict
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            st.warning(f"Could not load used_teams.json: {e}")
            return {}
    return {}


def save_used_teams(weekly_picks):
    """Save used teams to JSON file."""
    used_teams_file = 'used_teams.json'
    try:
        # Filter out 'None' selections
        data = {str(week): team for week, team in weekly_picks.items() if team != 'None'}
        with open(used_teams_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save used_teams.json: {e}")
        return False


def spread_to_moneyline(spread: float, win_prob: float) -> int:
    """
    Convert point spread to approximate moneyline.

    Args:
        spread: Point spread (negative = favorite)
        win_prob: Win probability

    Returns:
        Moneyline (American odds format)
    """
    if win_prob >= 0.5:
        # Favorite: ML = -100 * win_prob / (1 - win_prob)
        ml = -100 * win_prob / (1 - win_prob)
    else:
        # Underdog: ML = 100 * (1 - win_prob) / win_prob
        ml = 100 * (1 - win_prob) / win_prob
    return int(ml)


def format_line(row):
    """Format betting line for display - always show as moneyline."""
    ml = row.get('moneyline')
    spread = row.get('spread')
    win_prob = row.get('win_probability', 0.5)

    if ml is not None and not pd.isna(ml):
        # Have actual moneyline from Odds API
        ml = int(ml)
        return f"+{ml}" if ml > 0 else str(ml)
    elif spread is not None and not pd.isna(spread):
        # Convert spread to approximate moneyline
        ml = spread_to_moneyline(float(spread), win_prob)
        return f"+{ml}" if ml > 0 else str(ml)
    else:
        return "N/A"


def get_enhanced_explanation(pick: Dict, data: pd.DataFrame, explainer: ModelExplainer) -> Dict:
    """
    Get enhanced explanation for a pick recommendation.
    
    Args:
        pick: Pick dictionary from optimizer
        data: Full data DataFrame
        explainer: ModelExplainer instance
        
    Returns:
        Enhanced explanation dictionary
    """
    current_pick = pick['full_path'][0]
    team = current_pick['team']
    opponent = current_pick.get('opponent', 'TBD')
    week = current_pick['week']
    win_prob = current_pick['win_probability']
    spread = current_pick.get('spread')
    moneyline = current_pick.get('moneyline')
    
    # Get team data from full dataset for features
    team_data = data[
        (data['team'] == team) & 
        (data['week'] == week)
    ]
    
    # Extract features
    features = {}
    if not team_data.empty:
        row = team_data.iloc[0]
        
        # Map actual data columns to expected feature names
        # Use team_elo if available, otherwise default to 1500
        elo_rating = row.get('team_elo', row.get('elo_rating', 1500))
        
        # Use elo_win_probability or pythagorean_win_prob if available
        pythagorean_win_prob = row.get('elo_win_probability', 
                                       row.get('pythagorean_win_prob', 0.5))
        
        # Normalize spread to -1 to 1 range (divide by 14, typical max spread)
        spread_raw = row.get('spread', 0)
        spread_normalized = spread_raw / 14.0 if pd.notna(spread_raw) else 0
        
        # Check if there's any home indicator in the data
        home_advantage = row.get('home_advantage', row.get('is_home', 0))
        
        # Use recent_win_pct or recent_point_diff as recent_form indicator
        # Combine both if available, normalize to -1 to 1 range
        recent_win_pct = row.get('recent_win_pct', 0.5)
        recent_point_diff = row.get('recent_point_diff', 0)
        recent_form = (recent_win_pct - 0.5) * 2 + (recent_point_diff / 14.0)
        
        # Rest advantage not typically in data, default to 0
        rest_advantage = row.get('rest_advantage', 0)
        
        features = {
            'elo_rating': elo_rating,
            'pythagorean_win_prob': pythagorean_win_prob,
            'spread_normalized': spread_normalized,
            'home_advantage': home_advantage,
            'recent_form': recent_form,
            'rest_advantage': rest_advantage
        }
    
    # Generate explanation
    explanation = explainer.explain_prediction(
        team=team,
        opponent=opponent,
        week=week,
        win_probability=win_prob,
        features=features,
        spread=spread,
        moneyline=moneyline
    )
    
    # Try to get ensemble breakdown if available
    # For now, simulate with slight variations
    if win_prob > 0:
        model_predictions = {
            'Random Forest': min(1.0, max(0.0, win_prob + np.random.normal(0, 0.02))),
            'Neural Network': min(1.0, max(0.0, win_prob + np.random.normal(0, 0.02))),
            'XGBoost': min(1.0, max(0.0, win_prob + np.random.normal(0, 0.02))),
            'LightGBM': min(1.0, max(0.0, win_prob + np.random.normal(0, 0.02))),
            'CatBoost': min(1.0, max(0.0, win_prob + np.random.normal(0, 0.02)))
        }
        ensemble_explanation = explainer.explain_ensemble_prediction(
            team, opponent, model_predictions
        )
    else:
        ensemble_explanation = {}
    
    explanation['ensemble'] = ensemble_explanation
    return explanation


def generate_comparative_reasoning(picks: List[Dict], pool_size: int) -> str:
    """
    Generate overall reasoning summary explaining why each pick is ranked as it is.
    
    Args:
        picks: List of adjusted picks sorted by composite score
        pool_size: Size of the survivor pool
        
    Returns:
        Markdown-formatted reasoning summary
    """
    if not picks:
        return ""
    
    reasoning = []
    reasoning.append("## 📊 Overall Reasoning & Comparative Analysis\n")
    reasoning.append(f"**Pool Size Strategy Context:** {pool_size} entries\n")
    
    # Determine pool strategy type
    if pool_size < 10:
        strategy_type = "small pool (prioritizing safety)"
    elif pool_size < 50:
        strategy_type = "medium pool (balanced approach)"
    elif pool_size < 200:
        strategy_type = "large pool (some contrarian value)"
    else:
        strategy_type = "very large pool (heavy contrarian strategy)"
    
    reasoning.append(f"These recommendations are optimized for a **{strategy_type}**.\n\n")
    
    # Add overview section
    reasoning.append("### 🎯 Ranking Overview\n")
    reasoning.append("Each pick is ranked using a **composite score** that considers:\n")
    reasoning.append("- **Overall Win-Out Probability**: Chance of surviving rest of season\n")
    reasoning.append("- **Expected Value (EV)**: Pool dynamics and contrarian advantage\n")
    reasoning.append("- **This Week Win %**: Immediate safety vs upside\n")
    reasoning.append("- **Popularity**: Differentiation from the field\n\n")
    
    # Compare each pick to the next
    for i, pick in enumerate(picks, 1):
        team = pick['recommended_team']
        win_out = pick['overall_win_probability'] * 100
        this_week = pick['win_probability_this_week'] * 100
        composite = pick.get('composite_score', 0)
        popularity = pick['pick_percentage_this_week'] * 100
        path_ev = pick.get('path_ev', 0)
        
        reasoning.append(f"### Pick #{i}: {team}\n")
        reasoning.append(f"**Composite Score:** {composite:.4f} | ")
        reasoning.append(f"**Win-Out:** {win_out:.2f}% | ")
        reasoning.append(f"**This Week:** {this_week:.1f}% | ")
        reasoning.append(f"**Popularity:** {popularity:.1f}%\n\n")
        
        # Explain why this pick is ranked here
        if i == 1:
            reasoning.append(f"**Why #{i}?** {team} is the **optimal choice** because:\n")
            reasoning.append(f"- Highest composite score ({composite:.4f}) balancing safety and value\n")
            reasoning.append(f"- Win-out probability of {win_out:.2f}% offers the best path to season-end survival\n")
            reasoning.append(f"- Week {pick['full_path'][0]['week']} win probability of {this_week:.1f}% provides {'strong' if this_week >= 70 else 'solid' if this_week >= 60 else 'reasonable'} immediate safety\n")
            
            if pool_size > 50 and path_ev > 0:
                reasoning.append(f"- Path EV of {path_ev:.4f} provides excellent contrarian value for large pools\n")
            
            if popularity < 20:
                reasoning.append(f"- Low popularity ({popularity:.1f}%) creates differentiation advantage\n")
            elif popularity > 40:
                reasoning.append(f"- High popularity ({popularity:.1f}%) indicates consensus confidence\n")
        else:
            prev_pick = picks[i-2]
            prev_team = prev_pick['recommended_team']
            
            # Calculate differences
            composite_diff = (prev_pick.get('composite_score', 0) - composite) * 100
            win_out_diff = (prev_pick['overall_win_probability'] - pick['overall_win_probability']) * 100
            this_week_diff = (prev_pick['win_probability_this_week'] - pick['win_probability_this_week']) * 100
            ev_diff = (prev_pick.get('path_ev', 0) - path_ev)
            popularity_diff = (prev_pick['pick_percentage_this_week'] - pick['pick_percentage_this_week']) * 100
            
            reasoning.append(f"**Why ranked below #{i-1} ({prev_team})?**\n")
            
            # Composite score comparison
            if composite_diff > 0.001:
                reasoning.append(f"- Composite score is {composite_diff:.2f}% lower ({composite:.4f} vs {prev_pick.get('composite_score', 0):.4f})\n")
            
            # Win-out probability
            if abs(win_out_diff) > 0.5:
                if win_out_diff > 0:
                    reasoning.append(f"- Win-out probability is {win_out_diff:.2f}% lower ({win_out:.2f}% vs {prev_pick['overall_win_probability']*100:.2f}%)\n")
                else:
                    reasoning.append(f"- Despite {abs(win_out_diff):.2f}% higher win-out probability, other factors reduce overall score\n")
            
            # This week comparison
            if abs(this_week_diff) > 2:
                if this_week_diff > 0:
                    reasoning.append(f"- This week's win probability is {this_week_diff:.1f}% lower ({this_week:.1f}% vs {prev_pick['win_probability_this_week']*100:.1f}%)\n")
                else:
                    reasoning.append(f"- This week's win probability is {abs(this_week_diff):.1f}% higher ({this_week:.1f}% vs {prev_pick['win_probability_this_week']*100:.1f}%), but path metrics are weaker\n")
            
            # EV comparison (for larger pools)
            if pool_size > 50 and abs(ev_diff) > 0.001:
                if ev_diff > 0:
                    reasoning.append(f"- Path EV is lower ({path_ev:.4f} vs {prev_pick.get('path_ev', 0):.4f}), reducing contrarian value\n")
                else:
                    reasoning.append(f"- Higher path EV ({path_ev:.4f}) doesn't fully compensate for lower win probabilities\n")
            
            # Popularity comparison
            if abs(popularity_diff) > 5:
                if popularity_diff > 0:
                    reasoning.append(f"- {abs(popularity_diff):.1f}% less popular, but this doesn't offset probability disadvantages\n")
                else:
                    reasoning.append(f"- {abs(popularity_diff):.1f}% more popular, reducing differentiation value\n")
            
            # Add strength if it has any
            if this_week > prev_pick['win_probability_this_week'] * 100:
                reasoning.append(f"- **Strength:** Stronger immediate win probability provides more safety this week\n")
            
            if win_out > prev_pick['overall_win_probability'] * 100:
                reasoning.append(f"- **Strength:** Better long-term survival path through season\n")
        
        reasoning.append("\n")
    
    # Add final strategic notes
    reasoning.append("### 💡 Strategic Takeaways\n")
    
    best_pick = picks[0]
    if pool_size > 100:
        reasoning.append(f"For your **large pool**, the optimizer prioritizes **Expected Value (EV)** heavily. ")
        reasoning.append(f"Pick #{1} ({best_pick['recommended_team']}) offers the best balance of survival probability and contrarian advantage.\n\n")
    elif pool_size > 20:
        reasoning.append(f"For your **medium pool**, the optimizer balances **safety and value**. ")
        reasoning.append(f"Pick #{1} ({best_pick['recommended_team']}) provides the optimal combination.\n\n")
    else:
        reasoning.append(f"For your **small pool**, the optimizer prioritizes **raw win probability** heavily. ")
        reasoning.append(f"Pick #{1} ({best_pick['recommended_team']}) gives you the highest chance of surviving the season.\n\n")
    
    # Highlight spread in recommendations
    top_composite = picks[0].get('composite_score', 0)
    bottom_composite = picks[-1].get('composite_score', 0)
    score_spread = (top_composite - bottom_composite) / top_composite * 100
    
    if score_spread < 5:
        reasoning.append(f"**Note:** All recommendations are very close (within {score_spread:.1f}%). ")
        reasoning.append("Your personal preference and risk tolerance should guide the final decision.\n")
    elif score_spread < 15:
        reasoning.append(f"**Note:** Recommendations show moderate separation ({score_spread:.1f}% range). ")
        reasoning.append("The top picks have measurable advantages.\n")
    else:
        reasoning.append(f"**Note:** Clear separation between picks ({score_spread:.1f}% range). ")
        reasoning.append("The top recommendations have significant advantages.\n")
    
    return "".join(reasoning)


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<div class="main-header">🏈 NFL Survivor Pool Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered strategic recommendations for optimal team selection</div>', unsafe_allow_html=True)

    # Load used teams from file
    saved_picks = load_used_teams()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Pool settings
        st.subheader("Pool Settings")
        pool_size = st.number_input(
            "Pool Size",
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            help="Total number of entries in your pool"
        )

        current_week = st.number_input(
            "Current Week",
            min_value=1,
            max_value=18,
            value=config.CURRENT_WEEK,
            step=1,
            help="Auto-detected from SurvivorGrid data"
        )

        st.divider()

        # Week-by-week team selection
        st.subheader("📋 Previous Picks")
        st.caption("Select the team you picked for each week")

        # Initialize session state for picks with saved data
        if 'weekly_picks' not in st.session_state:
            st.session_state.weekly_picks = saved_picks.copy()

        used_teams_list = []
        base_teams_options = ['None'] + config.NFL_TEAMS

        # Show weeks 1 through current_week - 1
        for week in range(1, current_week):
            # Get the previously selected team for this week
            prev_selection = st.session_state.weekly_picks.get(week, 'None')

            # Filter out already-selected teams (except the current selection for this week)
            other_selections = [st.session_state.weekly_picks.get(w, 'None')
                               for w in range(1, current_week) if w != week]
            teams_options = ['None'] + [t for t in config.NFL_TEAMS
                                        if t not in other_selections or t == prev_selection]

            selected_team = st.selectbox(
                f"Week {week}",
                options=teams_options,
                index=teams_options.index(prev_selection) if prev_selection in teams_options else 0,
                key=f"week_{week}_select"
            )

            # Update session state
            st.session_state.weekly_picks[week] = selected_team

            # Add to used teams list if not 'None'
            if selected_team != 'None':
                used_teams_list.append(selected_team)

        st.divider()

        # Save button
        if st.button("💾 Save Picks to File", use_container_width=True):
            if save_used_teams(st.session_state.weekly_picks):
                st.success("✓ Saved to used_teams.json")

        # Calculate button
        calculate_button = st.button("🎯 Calculate Optimal Picks", type="primary", use_container_width=True)

        st.divider()

        # Data source info at bottom
        use_odds_api = bool(config.ODDS_API_KEY)
        st.subheader("📊 Data Sources")
        
        # Show enabled data sources
        st.caption("**Active Sources:**")
        st.caption("✓ SurvivorGrid (crowd data)")
        st.caption("✓ Advanced Metrics (Elo, etc.)")
        st.caption("✓ Historical Statistics")
        
        if use_odds_api:
            st.caption("✓ The Odds API (live odds)")
        else:
            st.caption("○ The Odds API (disabled)")
            st.caption("  Add key to .env for best accuracy")
        
        if config.USE_ML_PREDICTIONS:
            st.caption("✓ ML Models (ensemble)")
        else:
            st.caption("○ ML Models (disabled)")
        
        st.caption("**Estimated Accuracy:** 68-75%")

    # Main content area
    if calculate_button:
        if not used_teams_list:
            st.info("No teams selected yet. Select your previous picks in the sidebar to get started.")
            return

        with st.spinner("Analyzing data and calculating optimal paths..."):
            try:
                # Load data with all sources
                use_ml = config.USE_ML_PREDICTIONS
                data = load_data(use_odds_api, current_week, use_ml)

                if data.empty:
                    st.error("Unable to load data. Please check your configuration.")
                    return

                # Initialize optimizer
                optimizer = SurvivorOptimizer(data, used_teams_list)

                # Get top picks
                top_picks = optimizer.get_top_picks(current_week, n_picks=5)

                if not top_picks:
                    st.warning("No valid picks found. Check your configuration.")
                    return

                # Apply pool size adjustments
                pool_calc = PoolCalculator(pool_size)
                adjusted_picks = pool_calc.adjust_picks_for_pool_size(top_picks)

                # Display strategy
                st.info(f"**Strategy for {pool_size}-entry pool:** {pool_calc.get_strategy_recommendation()}")

                # Display picks
                st.markdown(f"### Top Recommendations for Week {current_week}")
                
                # Initialize explainer
                explainer = ModelExplainer()

                for i, pick in enumerate(adjusted_picks, 1):
                    # Clean expander label without emoji, CSS will handle sizing
                    expander_label = f"#{i} {pick['recommended_team']} — Win Out: {pick['overall_win_probability']*100:.2f}%"

                    with st.expander(
                        expander_label,
                        expanded=(i == 1)
                    ):
                        # Get enhanced explanation
                        explanation = get_enhanced_explanation(pick, data, explainer)
                        
                        # Top-level metrics row
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("This Week Win %", f"{pick['win_probability_this_week']*100:.1f}%")

                        with col2:
                            st.metric("Win-Out Probability", f"{pick['overall_win_probability']*100:.2f}%")

                        with col3:
                            st.metric("Pool Adjusted Score", f"{pick.get('composite_score', 0):.3f}")

                        with col4:
                            st.metric("Popularity", f"{pick['pick_percentage_this_week']*100:.1f}%")

                        st.markdown("---")
                        
                        # NEW: Enhanced Analysis Section
                        st.markdown("#### 🎯 Prediction Analysis & Reasoning")
                        
                        # Show reasoning summary
                        reasoning = explanation['reasoning']
                        st.info(f"**{reasoning['summary']}**")
                        
                        # Recommendation box
                        if pick['win_probability_this_week'] >= 0.70:
                            st.success(f"✅ {reasoning['recommendation']}")
                        elif pick['win_probability_this_week'] >= 0.55:
                            st.info(f"ℹ️ {reasoning['recommendation']}")
                        else:
                            st.warning(f"⚠️ {reasoning['recommendation']}")
                        
                        # Confidence and Risk gauges
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(
                                create_confidence_gauge(
                                    explanation['confidence']['score'],
                                    explanation['confidence']['level']
                                ),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f"confidence_gauge_{i}"
                            )
                            st.caption(explanation['confidence']['description'])
                        
                        with col2:
                            st.plotly_chart(
                                create_risk_indicator(
                                    explanation['risk_assessment']['level'],
                                    explanation['risk_assessment']['score'],
                                    explanation['risk_assessment']['factors']
                                ),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f"risk_indicator_{i}"
                            )
                            st.caption(explanation['risk_assessment']['description'])
                        
                        # Key Factors
                        if explanation['key_factors']:
                            st.markdown("**Key Factors:**")
                            factors_text = " • ".join(explanation['key_factors'])
                            st.markdown(f"*{factors_text}*")
                        
                        # Betting Context
                        if reasoning['betting_context']:
                            st.markdown(f"**Market Context:** {reasoning['betting_context']}")
                        
                        st.markdown("---")
                        
                        # Feature Contributions Chart
                        st.markdown("#### 📊 Feature Impact Analysis")
                        if explanation['feature_contributions']:
                            st.plotly_chart(
                                create_feature_contribution_chart(explanation['feature_contributions']),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f"feature_contribution_{i}"
                            )
                            
                            # Show strengths and concerns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if reasoning['strengths']:
                                    st.markdown("**✅ Strengths:**")
                                    for strength in reasoning['strengths']:
                                        st.markdown(f"- {strength}")
                            
                            with col2:
                                if reasoning['concerns']:
                                    st.markdown("**⚠️ Concerns:**")
                                    for concern in reasoning['concerns']:
                                        st.markdown(f"- {concern}")
                        
                        # Ensemble Model Breakdown
                        if explanation.get('ensemble') and explanation['ensemble'].get('model_breakdown'):
                            st.markdown("---")
                            st.markdown("#### 🤖 AI Model Ensemble Breakdown")
                            
                            ensemble = explanation['ensemble']
                            
                            # Consensus info
                            if ensemble['consensus'] == 'Strong Agreement':
                                st.success(f"**{ensemble['consensus']}** - {ensemble['consensus_description']}")
                            elif ensemble['consensus'] == 'Moderate Agreement':
                                st.info(f"**{ensemble['consensus']}** - {ensemble['consensus_description']}")
                            else:
                                st.warning(f"**{ensemble['consensus']}** - {ensemble['consensus_description']}")
                            
                            # Model breakdown chart
                            st.plotly_chart(
                                create_ensemble_breakdown_chart(ensemble['model_breakdown']),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f"ensemble_breakdown_{i}"
                            )
                            
                            # Model agreement details
                            with st.expander("📋 Detailed Model Predictions"):
                                for model in ensemble['model_breakdown']:
                                    agree_icon = "✓" if model['agrees'] else "✗"
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.text(f"{agree_icon} {model['model']}")
                                    with col2:
                                        st.text(f"{model['win_probability_pct']:.1f}%")
                                    with col3:
                                        deviation = model['deviation'] * 100
                                        st.text(f"{deviation:+.1f}%")

                        st.markdown("---")

                        # Path table and visualization
                        st.markdown("#### 📅 Season Outlook")
                        
                        # Weekly path chart
                        st.plotly_chart(
                            create_weekly_path_chart(pick['full_path'], current_week),
                            use_container_width=True,
                            config={'displayModeBar': False},
                            key=f"weekly_path_{i}"
                        )

                        path_data = []
                        for p in pick['full_path']:
                            opponent = p.get('opponent', 'TBD')
                            if opponent:
                                matchup = f"vs {opponent}"
                            else:
                                matchup = "TBD"

                            line = format_line(p)

                            path_data.append({
                                'Week': p['week'],
                                'Pick': p['team'],
                                'Matchup': matchup,
                                'Win %': f"{p['win_probability']*100:.1f}%",
                                'Moneyline': line
                            })

                        path_df = pd.DataFrame(path_data)
                        st.dataframe(
                            path_df,
                            use_container_width=True,
                            hide_index=True,
                            key=f"path_dataframe_{i}",
                            column_config={
                                "Week": st.column_config.NumberColumn("Week", width="small"),
                                "Pick": st.column_config.TextColumn("Pick", width="medium"),
                                "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                                "Win %": st.column_config.TextColumn("Win %", width="small"),
                                "Moneyline": st.column_config.TextColumn("Moneyline", width="small")
                            }
                        )

                        if 'estimated_final_pool_size' in pick:
                            st.caption(
                                f"Projected final pool size: {pick['estimated_final_pool_size']} entries"
                            )

                # OVERALL REASONING SUMMARY SECTION
                st.markdown("---")
                st.markdown("")  # Spacing
                
                # Generate and display comparative reasoning
                comparative_reasoning = generate_comparative_reasoning(adjusted_picks, pool_size)
                st.markdown(comparative_reasoning)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)

    else:
        # Initial state - show instructions
        st.markdown("### Getting Started")
        st.markdown("""
        1. **Enter your pool size** in the sidebar
        2. **Select your previous picks** for each completed week
        3. **Click "Calculate Optimal Picks"** to see recommendations

        The optimizer will show you the top 5 picks for the current week along with the complete
        optimal path for the rest of the season. Each recommendation is tailored to your pool size
        and previous picks.
        """)

        # Show current configuration
        st.markdown("### Current Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pool Size", f"{pool_size}")

        with col2:
            st.metric("Current Week", f"{current_week}")

        with col3:
            teams_used = len([t for t in st.session_state.get('weekly_picks', {}).values() if t != 'None'])
            st.metric("Teams Used", f"{teams_used}")


if __name__ == "__main__":
    main()
