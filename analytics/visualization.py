"""
Visualization components for model predictions and explanations.

Creates interactive charts and visual displays for:
- Feature importance and contributions
- Model ensemble breakdown
- Confidence and risk indicators
- Prediction reasoning
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def create_feature_contribution_chart(
    feature_contributions: List[Dict],
    max_features: int = 8
) -> go.Figure:
    """
    Create horizontal bar chart of feature contributions.
    
    Args:
        feature_contributions: List of feature contribution dicts
        max_features: Maximum number of features to display
        
    Returns:
        Plotly figure
    """
    # Take top features
    features = feature_contributions[:max_features]
    
    if not features:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No feature data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract data
    names = [f['feature'] for f in features]
    contributions = [f['contribution'] for f in features]
    colors = ['#10a37f' if f['impact'] == 'positive' else '#ef4444' 
              for f in features]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names[::-1],  # Reverse for top-to-bottom display
        x=contributions[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{c:+.2%}" for c in contributions[::-1]],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Feature Impact on Prediction',
            'font': {'size': 16, 'color': '#ececec'}
        },
        xaxis_title='Contribution to Win Probability',
        yaxis_title='',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=max(300, len(features) * 45),
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2,
            tickformat='.0%'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
        )
    )
    
    return fig


def create_ensemble_breakdown_chart(
    model_breakdown: List[Dict]
) -> go.Figure:
    """
    Create chart showing individual model predictions in ensemble.
    
    Args:
        model_breakdown: List of model prediction dicts
        
    Returns:
        Plotly figure
    """
    if not model_breakdown:
        fig = go.Figure()
        fig.add_annotation(
            text="No ensemble data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract data
    models = [m['model'] for m in model_breakdown]
    predictions = [m['prediction'] * 100 for m in model_breakdown]
    agrees = [m['agrees'] for m in model_breakdown]
    
    # Color based on agreement
    colors = ['#10a37f' if a else '#f59e0b' for a in agrees]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=predictions,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{p:.1f}%" for p in predictions],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Win Probability: %{y:.1f}%<extra></extra>'
    ))
    
    # Add average line
    avg_pred = np.mean(predictions)
    fig.add_hline(
        y=avg_pred,
        line_dash="dash",
        line_color="#8b5cf6",
        line_width=2,
        annotation_text=f"Ensemble: {avg_pred:.1f}%",
        annotation_position="right",
        annotation_font_color="#8b5cf6"
    )
    
    fig.update_layout(
        title={
            'text': 'Ensemble Model Breakdown',
            'font': {'size': 16, 'color': '#ececec'}
        },
        xaxis_title='Model',
        yaxis_title='Win Probability (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
        )
    )
    
    return fig


def create_confidence_gauge(
    confidence_score: float,
    confidence_level: str
) -> go.Figure:
    """
    Create gauge chart for confidence visualization.
    
    Args:
        confidence_score: Confidence score (0-1)
        confidence_level: Confidence level string
        
    Returns:
        Plotly figure
    """
    # Determine color based on confidence
    if confidence_score > 0.6:
        color = '#10a37f'  # Green
    elif confidence_score > 0.4:
        color = '#f59e0b'  # Yellow/Orange
    else:
        color = '#ef4444'  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {confidence_level}", 'font': {'size': 16, 'color': '#ececec'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#ececec'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#ececec'},
            'bar': {'color': color},
            'bgcolor': 'rgba(47, 47, 47, 0.5)',
            'borderwidth': 2,
            'bordercolor': '#4f4f4f',
            'steps': [
                {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [20, 40], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [40, 60], 'color': 'rgba(59, 130, 246, 0.3)'},
                {'range': [60, 80], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(16, 163, 127, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': confidence_score * 100
            }
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_risk_indicator(
    risk_level: str,
    risk_score: float,
    risk_factors: List[Dict]
) -> go.Figure:
    """
    Create visual risk indicator.
    
    Args:
        risk_level: Risk level string
        risk_score: Risk score (0-1, lower is better)
        risk_factors: List of risk factor dicts
        
    Returns:
        Plotly figure
    """
    # Invert risk score for display (higher = better)
    display_score = (1 - risk_score) * 100
    
    # Determine color (inverse of risk)
    if risk_score < 0.2:
        color = '#10a37f'  # Green - low risk
    elif risk_score < 0.4:
        color = '#3b82f6'  # Blue - moderate risk
    elif risk_score < 0.6:
        color = '#f59e0b'  # Orange - elevated risk
    else:
        color = '#ef4444'  # Red - high risk
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_level}", 'font': {'size': 16, 'color': '#ececec'}},
        number={'suffix': '', 'font': {'size': 40, 'color': '#ececec'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#ececec'},
            'bar': {'color': color},
            'bgcolor': 'rgba(47, 47, 47, 0.5)',
            'borderwidth': 2,
            'bordercolor': '#4f4f4f',
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(59, 130, 246, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(16, 163, 127, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': display_score
            }
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_prediction_breakdown_chart(
    win_probability: float,
    confidence_score: float,
    risk_score: float
) -> go.Figure:
    """
    Create multi-metric visualization showing prediction breakdown.
    
    Args:
        win_probability: Win probability (0-1)
        confidence_score: Confidence score (0-1)
        risk_score: Risk score (0-1)
        
    Returns:
        Plotly figure
    """
    categories = ['Win<br>Probability', 'Confidence', 'Safety<br>(1-Risk)']
    values = [
        win_probability * 100,
        confidence_score * 100,
        (1 - risk_score) * 100
    ]
    
    # Color based on value
    colors = []
    for val in values:
        if val >= 75:
            colors.append('#10a37f')
        elif val >= 60:
            colors.append('#3b82f6')
        elif val >= 45:
            colors.append('#f59e0b')
        else:
            colors.append('#ef4444')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
        textfont=dict(size=18, color='white'),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
    ))
    
    # Add reference line at 50%
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        line_width=1
    )
    
    fig.update_layout(
        title={
            'text': 'Prediction Overview',
            'font': {'size': 16, 'color': '#ececec'}
        },
        yaxis_title='Score (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=300,
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
        ),
        showlegend=False
    )
    
    return fig


def create_weekly_path_chart(
    path_data: List[Dict],
    current_week: int
) -> go.Figure:
    """
    Create line chart showing win probability across weekly path.
    
    Args:
        path_data: List of weekly pick dicts from optimizer
        current_week: Current week number
        
    Returns:
        Plotly figure
    """
    if not path_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No path data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    weeks = [p['week'] for p in path_data]
    win_probs = [p['win_probability'] * 100 for p in path_data]
    teams = [p['team'] for p in path_data]
    
    # Calculate cumulative win-out probability
    cumulative_prob = [100]
    for prob in win_probs:
        cumulative_prob.append(cumulative_prob[-1] * (prob / 100))
    cumulative_prob = cumulative_prob[1:]
    
    fig = go.Figure()
    
    # Add win probability line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=win_probs,
        mode='lines+markers',
        name='Weekly Win Probability',
        line=dict(color='#10a37f', width=3),
        marker=dict(size=10, color='#10a37f', line=dict(width=2, color='white')),
        hovertemplate='<b>Week %{x}</b><br>%{customdata}<br>Win Prob: %{y:.1f}%<extra></extra>',
        customdata=teams
    ))
    
    # Add cumulative probability line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=cumulative_prob,
        mode='lines+markers',
        name='Cumulative Survival Probability',
        line=dict(color='#8b5cf6', width=2, dash='dash'),
        marker=dict(size=8, color='#8b5cf6'),
        hovertemplate='<b>Week %{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>'
    ))
    
    # Highlight current week
    if current_week in weeks:
        idx = weeks.index(current_week)
        fig.add_vline(
            x=current_week,
            line_dash="dot",
            line_color="rgba(255,255,255,0.5)",
            line_width=2,
            annotation_text="Current Week",
            annotation_position="top"
        )
    
    fig.update_layout(
        title={
            'text': 'Optimal Path Win Probabilities',
            'font': {'size': 16, 'color': '#ececec'}
        },
        xaxis_title='Week',
        yaxis_title='Probability (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ececec', size=12),
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            dtick=1
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def test_visualizations():
    """Test visualization functions."""
    print("Testing Visualization Components")
    print("=" * 60)
    
    # Test feature contribution chart
    print("\n1. Testing Feature Contribution Chart...")
    features = [
        {'feature': 'Elo Rating', 'contribution': 0.08, 'impact': 'positive'},
        {'feature': 'Home Advantage', 'contribution': 0.06, 'impact': 'positive'},
        {'feature': 'Point Spread', 'contribution': 0.05, 'impact': 'positive'},
        {'feature': 'Recent Form', 'contribution': -0.03, 'impact': 'negative'},
    ]
    fig1 = create_feature_contribution_chart(features)
    print("   ✓ Feature chart created")
    
    # Test ensemble breakdown chart
    print("\n2. Testing Ensemble Breakdown Chart...")
    models = [
        {'model': 'Random Forest', 'prediction': 0.68, 'agrees': True},
        {'model': 'Neural Network', 'prediction': 0.72, 'agrees': True},
        {'model': 'XGBoost', 'prediction': 0.65, 'agrees': False},
    ]
    fig2 = create_ensemble_breakdown_chart(models)
    print("   ✓ Ensemble chart created")
    
    # Test confidence gauge
    print("\n3. Testing Confidence Gauge...")
    fig3 = create_confidence_gauge(0.75, 'High')
    print("   ✓ Confidence gauge created")
    
    # Test risk indicator
    print("\n4. Testing Risk Indicator...")
    risk_factors = [
        {'factor': 'Close spread', 'severity': 'medium'}
    ]
    fig4 = create_risk_indicator('Low', 0.15, risk_factors)
    print("   ✓ Risk indicator created")
    
    # Test prediction breakdown
    print("\n5. Testing Prediction Breakdown...")
    fig5 = create_prediction_breakdown_chart(0.72, 0.68, 0.15)
    print("   ✓ Prediction breakdown created")
    
    # Test weekly path chart
    print("\n6. Testing Weekly Path Chart...")
    path = [
        {'week': 7, 'team': 'Chiefs', 'win_probability': 0.72},
        {'week': 8, 'team': 'Bills', 'win_probability': 0.68},
        {'week': 9, 'team': '49ers', 'win_probability': 0.75},
    ]
    fig6 = create_weekly_path_chart(path, 7)
    print("   ✓ Weekly path chart created")
    
    print("\n✓ All visualization tests complete")


if __name__ == "__main__":
    test_visualizations()
