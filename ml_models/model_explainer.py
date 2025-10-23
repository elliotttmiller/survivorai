"""
Model explainability and reasoning module for NFL predictions.

Provides detailed explanations for model predictions including:
- Feature importance analysis
- Confidence scoring
- Risk assessment
- Prediction reasoning
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Explains model predictions with detailed reasoning and analysis.
    
    Provides:
    - Feature contribution analysis
    - Confidence scoring and calibration
    - Risk assessment
    - Human-readable explanations
    """
    
    def __init__(self, predictor=None):
        """
        Initialize model explainer.
        
        Args:
            predictor: MLNFLPredictor instance (optional)
        """
        self.predictor = predictor
        
    def explain_prediction(
        self,
        team: str,
        opponent: str,
        week: int,
        win_probability: float,
        features: Dict[str, Any],
        spread: Optional[float] = None,
        moneyline: Optional[int] = None
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            team: Team name
            opponent: Opponent name
            week: Week number
            win_probability: Predicted win probability
            features: Dictionary of features used in prediction
            spread: Point spread (optional)
            moneyline: Moneyline odds (optional)
            
        Returns:
            Dictionary with explanation components
        """
        # Analyze feature contributions
        feature_contributions = self._analyze_feature_contributions(
            features, win_probability
        )
        
        # Calculate confidence metrics
        confidence = self._calculate_confidence_metrics(
            win_probability, features
        )
        
        # Assess risk factors
        risk_assessment = self._assess_risk_factors(
            win_probability, features, spread
        )
        
        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(
            team, opponent, week, win_probability,
            feature_contributions, confidence, risk_assessment,
            spread, moneyline
        )
        
        return {
            'team': team,
            'opponent': opponent,
            'week': week,
            'win_probability': win_probability,
            'confidence': confidence,
            'feature_contributions': feature_contributions,
            'risk_assessment': risk_assessment,
            'reasoning': reasoning,
            'key_factors': self._identify_key_factors(feature_contributions)
        }
    
    def _analyze_feature_contributions(
        self,
        features: Dict[str, Any],
        win_probability: float
    ) -> List[Dict]:
        """
        Analyze which features contributed most to the prediction.
        
        Args:
            features: Feature dictionary
            win_probability: Predicted win probability
            
        Returns:
            List of feature contributions sorted by importance
        """
        contributions = []
        
        # Extract key features and their impact
        # Positive contributions (favor team winning)
        if 'elo_rating' in features:
            elo = features.get('elo_rating', 1500)
            elo_contribution = (elo - 1500) / 100 * 0.05  # ~5% per 100 Elo
            contributions.append({
                'feature': 'Elo Rating',
                'value': elo,
                'contribution': elo_contribution,
                'impact': 'positive' if elo_contribution > 0 else 'negative',
                'description': f"Team strength rating: {elo:.0f}"
            })
        
        if 'pythagorean_win_prob' in features:
            pyth = features.get('pythagorean_win_prob', 0.5)
            pyth_contribution = (pyth - 0.5) * 0.3  # Pythagorean heavily weighted
            contributions.append({
                'feature': 'Pythagorean Expectation',
                'value': pyth,
                'contribution': pyth_contribution,
                'impact': 'positive' if pyth_contribution > 0 else 'negative',
                'description': f"Expected win rate based on scoring: {pyth*100:.1f}%"
            })
        
        if 'spread_normalized' in features and features.get('spread_normalized') is not None:
            spread_norm = features.get('spread_normalized', 0)
            spread_contribution = -spread_norm * 0.15  # Negative spread = favorite
            contributions.append({
                'feature': 'Point Spread',
                'value': spread_norm * 14,  # Denormalize
                'contribution': spread_contribution,
                'impact': 'positive' if spread_contribution > 0 else 'negative',
                'description': f"Betting line: {spread_norm * 14:+.1f}"
            })
        
        if 'home_advantage' in features:
            home = features.get('home_advantage', 0)
            home_contribution = home * 0.06  # ~6% home field advantage
            if home > 0:
                contributions.append({
                    'feature': 'Home Field Advantage',
                    'value': home,
                    'contribution': home_contribution,
                    'impact': 'positive',
                    'description': "Playing at home (historical ~58% win rate)"
                })
        
        if 'rest_advantage' in features:
            rest = features.get('rest_advantage', 0)
            rest_contribution = rest * 0.04
            if rest != 0:
                contributions.append({
                    'feature': 'Rest Advantage',
                    'value': rest,
                    'contribution': rest_contribution,
                    'impact': 'positive' if rest_contribution > 0 else 'negative',
                    'description': f"Extra rest days: {rest}"
                })
        
        if 'recent_form' in features:
            form = features.get('recent_form', 0)
            form_contribution = form * 0.08
            contributions.append({
                'feature': 'Recent Form',
                'value': form,
                'contribution': form_contribution,
                'impact': 'positive' if form_contribution > 0 else 'negative',
                'description': f"Recent performance trend: {form:+.2f}"
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return contributions
    
    def _calculate_confidence_metrics(
        self,
        win_probability: float,
        features: Dict[str, Any]
    ) -> Dict:
        """
        Calculate confidence metrics for the prediction.
        
        Args:
            win_probability: Predicted win probability
            features: Feature dictionary
            
        Returns:
            Dictionary with confidence metrics
        """
        # Distance from 50-50 indicates confidence
        certainty = abs(win_probability - 0.5) * 2  # Scale to 0-1
        
        # Determine confidence level
        if certainty > 0.6:
            level = 'Very High'
            description = 'Strong indicators favor this prediction'
        elif certainty > 0.4:
            level = 'High'
            description = 'Clear advantage indicated by multiple factors'
        elif certainty > 0.25:
            level = 'Moderate'
            description = 'Some advantages but not overwhelming'
        elif certainty > 0.1:
            level = 'Low'
            description = 'Close matchup with slight edge'
        else:
            level = 'Very Low'
            description = 'Toss-up game, minimal predictive edge'
        
        # Check data quality indicators
        data_quality = 'Good'
        has_spread = features.get('spread_normalized') is not None
        has_elo = 'elo_rating' in features
        has_pyth = 'pythagorean_win_prob' in features
        
        data_sources = sum([has_spread, has_elo, has_pyth])
        if data_sources < 2:
            data_quality = 'Limited'
        
        return {
            'level': level,
            'score': certainty,
            'description': description,
            'data_quality': data_quality,
            'certainty_percentage': certainty * 100
        }
    
    def _assess_risk_factors(
        self,
        win_probability: float,
        features: Dict[str, Any],
        spread: Optional[float] = None
    ) -> Dict:
        """
        Assess risk factors for the prediction.
        
        Args:
            win_probability: Predicted win probability
            features: Feature dictionary
            spread: Point spread
            
        Returns:
            Dictionary with risk assessment
        """
        risk_factors = []
        risk_score = 0.0  # Lower is better
        
        # Check if it's a close game
        if spread is not None and abs(spread) < 3:
            risk_factors.append({
                'factor': 'Close Spread',
                'severity': 'medium',
                'description': 'Game expected to be decided by less than a field goal'
            })
            risk_score += 0.15
        
        # Check win probability uncertainty
        if 0.45 <= win_probability <= 0.55:
            risk_factors.append({
                'factor': 'Coin Flip Game',
                'severity': 'high',
                'description': 'Near 50-50 probability indicates high uncertainty'
            })
            risk_score += 0.25
        elif 0.40 <= win_probability <= 0.60:
            risk_factors.append({
                'factor': 'Moderate Uncertainty',
                'severity': 'medium',
                'description': 'Some uncertainty in outcome'
            })
            risk_score += 0.12
        
        # Check recent form volatility
        if 'recent_form' in features:
            form = features.get('recent_form', 0)
            if form < -0.2:
                risk_factors.append({
                    'factor': 'Recent Struggles',
                    'severity': 'medium',
                    'description': 'Team has been underperforming recently'
                })
                risk_score += 0.10
        
        # Check if underdog
        if spread is not None and spread > 3:
            risk_factors.append({
                'factor': 'Underdog Pick',
                'severity': 'low',
                'description': 'Betting markets favor opponent'
            })
            risk_score += 0.08
        
        # Determine overall risk level
        if risk_score < 0.10:
            risk_level = 'Low'
            risk_description = 'Strong pick with minimal concerns'
        elif risk_score < 0.25:
            risk_level = 'Moderate'
            risk_description = 'Good pick with some risk factors'
        elif risk_score < 0.40:
            risk_level = 'Elevated'
            risk_description = 'Uncertain outcome, proceed with caution'
        else:
            risk_level = 'High'
            risk_description = 'High-risk pick, consider alternatives'
        
        return {
            'level': risk_level,
            'score': risk_score,
            'description': risk_description,
            'factors': risk_factors
        }
    
    def _generate_reasoning(
        self,
        team: str,
        opponent: str,
        week: int,
        win_probability: float,
        feature_contributions: List[Dict],
        confidence: Dict,
        risk_assessment: Dict,
        spread: Optional[float] = None,
        moneyline: Optional[int] = None
    ) -> Dict:
        """
        Generate human-readable reasoning for the prediction.
        
        Args:
            team: Team name
            opponent: Opponent name
            week: Week number
            win_probability: Predicted win probability
            feature_contributions: Feature contribution analysis
            confidence: Confidence metrics
            risk_assessment: Risk assessment
            spread: Point spread
            moneyline: Moneyline odds
            
        Returns:
            Dictionary with reasoning text components
        """
        # Main summary
        win_pct = win_probability * 100
        
        if win_probability >= 0.70:
            outlook = "strong favorite"
        elif win_probability >= 0.60:
            outlook = "moderate favorite"
        elif win_probability >= 0.55:
            outlook = "slight favorite"
        elif win_probability >= 0.45:
            outlook = "evenly matched"
        else:
            outlook = "underdog"
        
        summary = (
            f"{team} is projected as a {outlook} against {opponent} in Week {week} "
            f"with a {win_pct:.1f}% win probability."
        )
        
        # Key strengths
        strengths = []
        for contrib in feature_contributions[:3]:  # Top 3 positive
            if contrib['impact'] == 'positive' and contrib['contribution'] > 0.03:
                strengths.append(contrib['description'])
        
        # Key concerns
        concerns = []
        for contrib in feature_contributions[:3]:
            if contrib['impact'] == 'negative' and abs(contrib['contribution']) > 0.03:
                concerns.append(contrib['description'])
        
        # Add risk factors
        for risk_factor in risk_assessment['factors']:
            if risk_factor['severity'] in ['high', 'medium']:
                concerns.append(risk_factor['description'])
        
        # Betting context
        betting_context = ""
        if spread is not None:
            if spread < -7:
                betting_context = f"Vegas has {team} as a heavy favorite ({spread:.1f} point spread)"
            elif spread < -3:
                betting_context = f"Vegas favors {team} ({spread:.1f} point spread)"
            elif spread < -0.5:
                betting_context = f"Vegas slightly favors {team} ({spread:.1f} point spread)"
            elif spread < 0.5:
                betting_context = f"Vegas sees this as a toss-up (pick'em)"
            else:
                betting_context = f"Vegas favors {opponent} ({-spread:.1f} point spread)"
        
        if moneyline is not None:
            if betting_context:
                betting_context += f", moneyline: {moneyline:+d}"
            else:
                betting_context = f"Moneyline: {moneyline:+d}"
        
        # Recommendation
        if win_probability >= 0.75 and confidence['score'] > 0.5:
            recommendation = (
                f"STRONG PICK: High confidence in {team} winning. "
                f"Excellent survivor pool selection for Week {week}."
            )
        elif win_probability >= 0.65 and confidence['score'] > 0.35:
            recommendation = (
                f"GOOD PICK: Solid choice with {team} having clear advantages. "
                f"Recommended for survivor pools."
            )
        elif win_probability >= 0.55:
            recommendation = (
                f"MODERATE PICK: {team} has an edge but not overwhelming. "
                f"Consider your pool strategy and alternative options."
            )
        else:
            recommendation = (
                f"RISKY PICK: Low confidence in this selection. "
                f"Only use if necessary or as a contrarian play."
            )
        
        return {
            'summary': summary,
            'strengths': strengths,
            'concerns': concerns,
            'betting_context': betting_context,
            'recommendation': recommendation,
            'confidence_note': confidence['description']
        }
    
    def _identify_key_factors(
        self,
        feature_contributions: List[Dict]
    ) -> List[str]:
        """
        Identify the most important factors in the prediction.
        
        Args:
            feature_contributions: Feature contribution analysis
            
        Returns:
            List of key factor names
        """
        key_factors = []
        for contrib in feature_contributions[:5]:  # Top 5
            if abs(contrib['contribution']) > 0.05:
                impact_text = "+" if contrib['impact'] == 'positive' else "-"
                key_factors.append(f"{impact_text} {contrib['feature']}")
        
        return key_factors
    
    def explain_ensemble_prediction(
        self,
        team: str,
        opponent: str,
        model_predictions: Dict[str, float]
    ) -> Dict:
        """
        Explain how ensemble models arrived at the prediction.
        
        Args:
            team: Team name
            opponent: Opponent name
            model_predictions: Dict mapping model names to their predictions
            
        Returns:
            Dictionary with ensemble explanation
        """
        if not model_predictions:
            return {
                'model_breakdown': [],
                'consensus': 'Unknown',
                'disagreement': 0.0
            }
        
        predictions = list(model_predictions.values())
        avg_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Calculate disagreement
        disagreement = std_pred / (avg_pred * (1 - avg_pred) + 0.01)  # Normalized
        
        # Model breakdown
        breakdown = []
        for model_name, pred in model_predictions.items():
            deviation = pred - avg_pred
            breakdown.append({
                'model': model_name,
                'prediction': pred,
                'win_probability_pct': pred * 100,
                'deviation': deviation,
                'agrees': abs(deviation) < 0.05
            })
        
        # Sort by prediction
        breakdown.sort(key=lambda x: x['prediction'], reverse=True)
        
        # Determine consensus
        if disagreement < 0.15:
            consensus = 'Strong Agreement'
            consensus_desc = 'All models align closely on this prediction'
        elif disagreement < 0.30:
            consensus = 'Moderate Agreement'
            consensus_desc = 'Models generally agree with some variation'
        else:
            consensus = 'High Disagreement'
            consensus_desc = 'Models show significant divergence - higher uncertainty'
        
        return {
            'model_breakdown': breakdown,
            'consensus': consensus,
            'consensus_description': consensus_desc,
            'disagreement': disagreement,
            'average_prediction': avg_pred,
            'std_deviation': std_pred,
            'prediction_range': (min(predictions), max(predictions))
        }


def test_explainer():
    """Test the model explainer functionality."""
    print("Testing Model Explainer")
    print("=" * 60)
    
    explainer = ModelExplainer()
    
    # Test single prediction explanation
    features = {
        'elo_rating': 1620,
        'pythagorean_win_prob': 0.68,
        'spread_normalized': -0.3,  # -4.2 point favorite
        'home_advantage': 1,
        'recent_form': 0.15,
        'rest_advantage': 0
    }
    
    explanation = explainer.explain_prediction(
        team='Kansas City Chiefs',
        opponent='Buffalo Bills',
        week=7,
        win_probability=0.72,
        features=features,
        spread=-4.5,
        moneyline=-190
    )
    
    print("\nPrediction Explanation:")
    print(f"Team: {explanation['team']}")
    print(f"Win Probability: {explanation['win_probability']*100:.1f}%")
    print(f"\nConfidence: {explanation['confidence']['level']} ({explanation['confidence']['certainty_percentage']:.1f}%)")
    print(f"Risk Level: {explanation['risk_assessment']['level']}")
    
    print("\nReasoning:")
    print(f"  {explanation['reasoning']['summary']}")
    print(f"  {explanation['reasoning']['recommendation']}")
    
    print("\nKey Factors:")
    for factor in explanation['key_factors']:
        print(f"  - {factor}")
    
    print("\nFeature Contributions:")
    for contrib in explanation['feature_contributions'][:5]:
        print(f"  {contrib['feature']}: {contrib['contribution']:+.3f} ({contrib['impact']})")
    
    # Test ensemble explanation
    model_preds = {
        'Random Forest': 0.71,
        'Neural Network': 0.74,
        'XGBoost': 0.69,
        'LightGBM': 0.73,
        'CatBoost': 0.72
    }
    
    ensemble_exp = explainer.explain_ensemble_prediction(
        'Kansas City Chiefs',
        'Buffalo Bills',
        model_preds
    )
    
    print("\n\nEnsemble Explanation:")
    print(f"Consensus: {ensemble_exp['consensus']}")
    print(f"Average Prediction: {ensemble_exp['average_prediction']*100:.1f}%")
    print(f"\nModel Breakdown:")
    for model in ensemble_exp['model_breakdown']:
        agreement = "✓" if model['agrees'] else "✗"
        print(f"  {agreement} {model['model']}: {model['win_probability_pct']:.1f}%")
    
    print("\n✓ Model explainer test complete")


if __name__ == "__main__":
    test_explainer()
