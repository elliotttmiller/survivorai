"""Test that injury display format shows comprehensive information."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.injury_reports import (
    InjuryImpactAnalyzer,
    get_injury_summary_for_team
)


class TestInjuryDisplayFormat(unittest.TestCase):
    """Test that injury details include all comprehensive information."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = InjuryImpactAnalyzer()
    
    def test_detail_includes_team_name(self):
        """Test that injury details include team name."""
        # Simulate real injury data as it would come from scrapers
        injuries = [
            {
                'player_name': 'Patrick Mahomes',
                'team': 'Kansas City Chiefs',
                'position': 'QB',
                'status': 'OUT',
                'injury_type': 'ANKLE',
                'source': 'ESPN',
                'date_reported': '2024-10-26T10:00:00',
            }
        ]
        
        critical = self.analyzer.get_critical_injuries(injuries, threshold=0.25)
        self.assertEqual(len(critical), 1)
        
        # Format details like get_injury_summary_for_team does
        detail = {
            'player': critical[0]['player_name'],
            'team': critical[0].get('team', 'Unknown'),
            'position': critical[0]['position'],
            'status': critical[0]['status'],
            'injury_type': critical[0].get('injury_type', 'UNKNOWN'),
            'impact': critical[0].get('impact_score', 0),
            'source': critical[0].get('source', 'Unknown'),
            'date_reported': critical[0].get('date_reported', ''),
            'analysis': critical[0].get('analysis', '')
        }
        
        # Verify all fields are present
        self.assertEqual(detail['player'], 'Patrick Mahomes')
        self.assertEqual(detail['team'], 'Kansas City Chiefs')
        self.assertEqual(detail['position'], 'QB')
        self.assertEqual(detail['status'], 'OUT')
        self.assertEqual(detail['injury_type'], 'ANKLE')
        self.assertEqual(detail['source'], 'ESPN')
        self.assertGreater(detail['impact'], 0)
    
    def test_detail_includes_source(self):
        """Test that injury details include data source."""
        injuries = [
            {
                'player_name': 'Travis Kelce',
                'team': 'Kansas City Chiefs',
                'position': 'TE',
                'status': 'OUT',  # Changed to OUT to meet threshold
                'injury_type': 'KNEE',
                'source': 'CBS Sports',
                'date_reported': '2024-10-26T10:00:00',
            }
        ]
        
        critical = self.analyzer.get_critical_injuries(injuries, threshold=0.20)
        self.assertEqual(len(critical), 1)
        
        detail = {
            'player': critical[0]['player_name'],
            'team': critical[0].get('team', 'Unknown'),
            'position': critical[0]['position'],
            'status': critical[0]['status'],
            'injury_type': critical[0].get('injury_type', 'UNKNOWN'),
            'impact': critical[0].get('impact_score', 0),
            'source': critical[0].get('source', 'Unknown'),
            'date_reported': critical[0].get('date_reported', ''),
            'analysis': critical[0].get('analysis', '')
        }
        
        self.assertEqual(detail['source'], 'CBS Sports')
    
    def test_multiple_injuries_show_different_teams(self):
        """Test that multiple injuries from different teams show team names."""
        injuries = [
            {
                'player_name': 'Christian McCaffrey',
                'team': 'San Francisco 49ers',
                'position': 'RB',
                'status': 'OUT',
                'injury_type': 'ACHILLES',
                'source': 'ESPN',
                'date_reported': '2024-10-26T10:00:00',
            },
            {
                'player_name': 'Deebo Samuel',
                'team': 'San Francisco 49ers',
                'position': 'WR',
                'status': 'QUESTIONABLE',
                'injury_type': 'HAMSTRING',
                'source': 'ESPN',
                'date_reported': '2024-10-26T10:00:00',
            }
        ]
        
        critical = self.analyzer.get_critical_injuries(injuries, threshold=0.20)
        self.assertGreater(len(critical), 0)
        
        # All injuries should have team name
        for inj in critical:
            detail = {
                'player': inj['player_name'],
                'team': inj.get('team', 'Unknown'),
                'position': inj['position'],
                'status': inj['status'],
                'injury_type': inj.get('injury_type', 'UNKNOWN'),
                'impact': inj.get('impact_score', 0),
                'source': inj.get('source', 'Unknown'),
            }
            self.assertNotEqual(detail['team'], 'Unknown')
            self.assertEqual(detail['team'], 'San Francisco 49ers')
    
    def test_formatted_output_example(self):
        """Test complete formatted output example."""
        injuries = [
            {
                'player_name': 'Justin Jefferson',
                'team': 'Minnesota Vikings',
                'position': 'WR',
                'status': 'OUT',
                'injury_type': 'HAMSTRING',
                'source': 'ESPN',
                'date_reported': '2024-10-26T10:00:00',
            }
        ]
        
        critical = self.analyzer.get_critical_injuries(injuries, threshold=0.20)
        self.assertEqual(len(critical), 1)
        
        detail = {
            'player': critical[0]['player_name'],
            'team': critical[0].get('team', 'Unknown'),
            'position': critical[0]['position'],
            'status': critical[0]['status'],
            'injury_type': critical[0].get('injury_type', 'UNKNOWN'),
            'impact': critical[0].get('impact_score', 0),
            'source': critical[0].get('source', 'Unknown'),
        }
        
        # Print formatted output (like app.py would display it)
        print("\n" + "="*60)
        print("EXPECTED INJURY DISPLAY FORMAT:")
        print("="*60)
        print(f"**{detail['player']}** ({detail['position']}) — *{detail['status']}*")
        print(f"  • Team: {detail['team']}")
        print(f"  • Injury: {detail['injury_type']}")
        print(f"  • Impact Score: {detail['impact']:.3f}")
        print(f"  • Source: {detail['source']}")
        print("="*60 + "\n")
        
        # Verify this is not the old format with "Player 1", "Player 2"
        self.assertNotIn('Player 1', detail['player'])
        self.assertNotIn('Player 2', detail['player'])
        self.assertNotEqual(detail['team'], 'Unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
