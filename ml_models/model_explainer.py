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
        
        # Enhanced comprehensive summary
        summary = self._build_comprehensive_summary(
            team, opponent, week, win_probability, outlook,
            feature_contributions, confidence, risk_assessment, spread, moneyline
        )
        
        # Key strengths - more comprehensive
        strengths = []
        for contrib in feature_contributions:
            if contrib['impact'] == 'positive' and contrib['contribution'] > 0.02:
                # Add detailed explanation
                strength_detail = self._explain_feature_strength(contrib, team)
                strengths.append(strength_detail)
        
        # Key concerns - more comprehensive
        concerns = []
        for contrib in feature_contributions:
            if contrib['impact'] == 'negative' and abs(contrib['contribution']) > 0.02:
                # Add detailed explanation
                concern_detail = self._explain_feature_concern(contrib, team, opponent)
                concerns.append(concern_detail)
        
        # Add risk factors with detailed context
        for risk_factor in risk_assessment['factors']:
            if risk_factor['severity'] in ['high', 'medium']:
                concerns.append(f"{risk_factor['description']} (Impact: {risk_factor['severity']})")
        
        # Enhanced betting context
        betting_context = self._build_betting_context(
            team, opponent, spread, moneyline, win_probability
        )
        
        # Enhanced recommendation with more context
        recommendation = self._build_comprehensive_recommendation(
            team, opponent, week, win_probability, confidence,
            risk_assessment, strengths, concerns
        )
        
        return {
            'summary': summary,
            'strengths': strengths,
            'concerns': concerns,
            'betting_context': betting_context,
            'recommendation': recommendation,
            'confidence_note': confidence['description']
        }
    
    def _build_comprehensive_summary(
        self,
        team: str,
        opponent: str,
        week: int,
        win_probability: float,
        outlook: str,
        feature_contributions: List[Dict],
        confidence: Dict,
        risk_assessment: Dict,
        spread: Optional[float],
        moneyline: Optional[int]
    ) -> str:
        """Build a comprehensive prediction summary."""
        win_pct = win_probability * 100
        
        # Base summary
        summary_parts = [
            f"{team} is projected as a {outlook} against {opponent} in Week {week} "
            f"with a {win_pct:.1f}% win probability."
        ]
        
        # Add key driver explanation
        top_factors = [c for c in feature_contributions[:3] if abs(c['contribution']) > 0.04]
        if top_factors:
            primary_driver = top_factors[0]
            driver_text = (
                f" This projection is primarily driven by "
                f"{primary_driver['feature'].lower()} ({primary_driver['description'].lower()})"
            )
            if primary_driver['impact'] == 'positive':
                driver_text += f", which significantly favors {team}."
            else:
                driver_text += f", which presents a notable challenge."
            summary_parts.append(driver_text)
        
        # Add confidence context
        if confidence['level'] in ['Very High', 'High']:
            summary_parts.append(
                f" Our analysis shows {confidence['level'].lower()} confidence in this prediction, "
                f"with multiple indicators aligning in the same direction."
            )
        elif confidence['level'] in ['Low', 'Very Low']:
            summary_parts.append(
                f" This matchup shows {confidence['level'].lower()} confidence due to "
                f"closely matched teams and competing factors."
            )
        
        # Add market alignment
        if spread is not None:
            market_alignment = self._assess_market_alignment(win_probability, spread)
            if market_alignment:
                summary_parts.append(f" {market_alignment}")
        
        return "".join(summary_parts)
    
    def _assess_market_alignment(self, win_probability: float, spread: float) -> str:
        """Assess how our prediction aligns with betting markets."""
        # Convert spread to approximate win probability
        # Rule of thumb: each point ≈ 3.3% win probability shift from 50%
        if spread is not None:
            market_implied_prob = 0.5 + (spread * -0.033)
            market_implied_prob = max(0.1, min(0.9, market_implied_prob))
            
            diff = abs(win_probability - market_implied_prob)
            
            if diff < 0.05:
                return "Our model closely aligns with Vegas betting markets."
            elif diff < 0.10:
                if win_probability > market_implied_prob:
                    return "Our model is slightly more bullish than Vegas markets suggest."
                else:
                    return "Our model is slightly more conservative than Vegas markets."
            elif win_probability > market_implied_prob:
                return "Our model sees significantly more value here than Vegas markets are pricing in."
            else:
                return "Vegas markets are more optimistic than our analytical models suggest."
        return ""
    
    def _explain_feature_strength(self, contrib: Dict, team: str) -> str:
        """Explain why a positive feature is a strength."""
        feature = contrib['feature']
        value = contrib.get('value', 0)
        contribution = contrib['contribution']
        
        explanations = {
            'Elo Rating': (
                f"Elite team strength rating ({value:.0f}) places {team} in top tier, "
                f"adding ~{contribution*100:.1f} percentage points to win probability"
            ),
            'Pythagorean Expectation': (
                f"Scoring efficiency metrics ({value*100:.1f}% expected win rate) indicate "
                f"dominant offensive/defensive balance, contributing {contribution*100:.1f}pp"
            ),
            'Point Spread': (
                f"Betting markets heavily favor {team} ({value:+.1f} spread), "
                f"reflecting professional oddsmaker confidence worth {contribution*100:.1f}pp"
            ),
            'Home Field Advantage': (
                f"Playing at home provides significant edge with crowd support and familiarity, "
                f"historically worth ~6% win probability boost ({contribution*100:.1f}pp here)"
            ),
            'Recent Form': (
                f"Strong recent performance trajectory (form score: {value:+.2f}) shows "
                f"momentum and improvement, adding {contribution*100:.1f}pp confidence"
            ),
            'Rest Advantage': (
                f"Extra rest days ({value}) provide recovery and preparation advantage, "
                f"contributing {contribution*100:.1f}pp to win probability"
            )
        }
        
        return explanations.get(feature, contrib['description'])
    
    def _explain_feature_concern(self, contrib: Dict, team: str, opponent: str) -> str:
        """Explain why a negative feature is a concern."""
        feature = contrib['feature']
        value = contrib.get('value', 0)
        contribution = abs(contrib['contribution'])
        
        explanations = {
            'Elo Rating': (
                f"Lower team strength rating ({value:.0f}) indicates quality disadvantage, "
                f"reducing win probability by ~{contribution*100:.1f} percentage points"
            ),
            'Pythagorean Expectation': (
                f"Scoring efficiency below opponent ({value*100:.1f}% expected rate) reveals "
                f"structural weaknesses, costing {contribution*100:.1f}pp"
            ),
            'Point Spread': (
                f"Betting markets favor {opponent} ({value:+.1f} spread against), "
                f"reflecting market concern worth -{contribution*100:.1f}pp"
            ),
            'Recent Form': (
                f"Poor recent performance (form score: {value:+.2f}) suggests declining "
                f"momentum and potential issues, reducing confidence by {contribution*100:.1f}pp"
            ),
            'Rest Advantage': (
                f"Rest disadvantage ({value} days) means less recovery time, "
                f"reducing win probability by {contribution*100:.1f}pp"
            )
        }
        
        return explanations.get(feature, contrib['description'])
    
    def _build_betting_context(
        self,
        team: str,
        opponent: str,
        spread: Optional[float],
        moneyline: Optional[int],
        win_probability: float
    ) -> str:
        """Build comprehensive betting market context."""
        context_parts = []
        
        if spread is not None:
            # Spread context with interpretation
            if spread < -10:
                context_parts.append(
                    f"Vegas has {team} as a dominant favorite ({spread:.1f} spread), "
                    f"expecting a decisive victory by double-digits"
                )
            elif spread < -7:
                context_parts.append(
                    f"Vegas has {team} as a heavy favorite ({spread:.1f} spread), "
                    f"projected to win by more than a touchdown"
                )
            elif spread < -3.5:
                context_parts.append(
                    f"Vegas favors {team} ({spread:.1f} spread) to win by roughly a field goal or more"
                )
            elif spread < -1:
                context_parts.append(
                    f"Vegas slightly favors {team} ({spread:.1f} spread) in what's expected to be close"
                )
            elif spread < 1:
                context_parts.append(
                    f"Vegas sees this as a toss-up (pick'em), indicating an evenly matched game"
                )
            elif spread < 3.5:
                context_parts.append(
                    f"Vegas slightly favors {opponent} ({-spread:.1f} spread), making {team} a small underdog"
                )
            else:
                context_parts.append(
                    f"Vegas favors {opponent} ({-spread:.1f} spread), positioning {team} as the underdog"
                )
        
        # Add moneyline context
        if moneyline is not None:
            ml_text = f"Moneyline: {moneyline:+d}"
            if moneyline < -300:
                ml_text += " (heavy favorite, low payout)"
            elif moneyline < -150:
                ml_text += " (solid favorite)"
            elif moneyline < -110:
                ml_text += " (slight favorite)"
            elif moneyline < 110:
                ml_text += " (pick'em)"
            elif moneyline < 200:
                ml_text += " (underdog value)"
            else:
                ml_text += " (significant underdog)"
            
            if context_parts:
                context_parts.append(f". {ml_text}")
            else:
                context_parts.append(ml_text)
        
        # Add implied probability comparison
        if spread is not None:
            market_prob = 0.5 + (spread * -0.033)
            market_prob = max(0.1, min(0.9, market_prob))
            our_prob = win_probability
            
            if abs(our_prob - market_prob) > 0.10:
                diff_text = f"{abs(our_prob - market_prob)*100:.1f}%"
                if our_prob > market_prob:
                    context_parts.append(
                        f". Our model sees {diff_text} more value than markets are pricing, "
                        f"suggesting potential market inefficiency"
                    )
                else:
                    context_parts.append(
                        f". Markets are {diff_text} more optimistic than our models, "
                        f"warranting additional caution"
                    )
        
        return "".join(context_parts) if context_parts else ""
    
    def _build_comprehensive_recommendation(
        self,
        team: str,
        opponent: str,
        week: int,
        win_probability: float,
        confidence: Dict,
        risk_assessment: Dict,
        strengths: List[str],
        concerns: List[str]
    ) -> str:
        """Build comprehensive recommendation with detailed reasoning."""
        win_pct = win_probability * 100
        
        if win_probability >= 0.75 and confidence['score'] > 0.5:
            rec = (
                f"STRONG PICK: High confidence in {team} winning ({win_pct:.1f}% probability). "
                f"Multiple factors strongly favor this outcome, making it an excellent survivor pool "
                f"selection for Week {week}. "
            )
            if len(strengths) >= 3:
                rec += f"With {len(strengths)} significant advantages identified, this pick offers "
                rec += "both safety and reliability. "
            if risk_assessment['level'] == 'Low':
                rec += "Risk profile is minimal, suitable for conservative strategies."
            
        elif win_probability >= 0.65 and confidence['score'] > 0.35:
            rec = (
                f"GOOD PICK: Solid choice with {team} having clear advantages ({win_pct:.1f}% win probability). "
                f"Recommended for survivor pools with {confidence['level'].lower()} confidence. "
            )
            if len(concerns) > 0:
                rec += f"While there are {len(concerns)} concern(s) to monitor, the overall "
                rec += "profile remains favorable. "
            rec += "Represents a balanced risk/reward selection."
            
        elif win_probability >= 0.55:
            rec = (
                f"MODERATE PICK: {team} has an edge ({win_pct:.1f}% probability) but not overwhelming. "
            )
            if risk_assessment['level'] in ['Elevated', 'High']:
                rec += f"With {risk_assessment['level'].lower()} risk level, this pick requires careful "
                rec += "consideration of your pool strategy and remaining alternatives. "
            else:
                rec += "Consider your pool position, remaining teams, and alternative options. "
            
            if len(strengths) > len(concerns):
                rec += f"The {len(strengths)} identified strengths outweigh {len(concerns)} concerns, "
                rec += "but margin of safety is tighter."
                
        else:
            rec = (
                f"RISKY PICK: Low confidence in this selection ({win_pct:.1f}% probability). "
            )
            if win_probability < 0.50:
                rec += f"{team} is actually projected as the underdog against {opponent}. "
            
            rec += (
                f"With {confidence['level'].lower()} confidence and {risk_assessment['level'].lower()} risk, "
                f"only use if absolutely necessary due to previous picks, or as a deliberate "
                f"contrarian play in large pools where differentiation is crucial."
            )
        
        return rec
    
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
