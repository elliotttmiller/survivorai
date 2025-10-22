"""Test suite for week detection and data collection fixes."""
import unittest
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.season_detector import NFLSeasonDetector


class TestWeekDetection(unittest.TestCase):
    """Test week detection logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = NFLSeasonDetector()
    
    def test_week_transitions_on_tuesday(self):
        """Test that weeks transition on Tuesday after Monday Night Football."""
        # Week 7 Monday (games still ongoing)
        monday = datetime(2025, 10, 20)
        _, week = self.detector.get_season_info(monday)
        self.assertEqual(week, 7, "Monday should still be week 7")
        
        # Week 8 Tuesday (planning for next week)
        tuesday = datetime(2025, 10, 21)
        _, week = self.detector.get_season_info(tuesday)
        self.assertEqual(week, 8, "Tuesday should be week 8")
        
        # Week 8 Wednesday
        wednesday = datetime(2025, 10, 22)
        _, week = self.detector.get_season_info(wednesday)
        self.assertEqual(week, 8, "Wednesday should be week 8")
    
    def test_season_detection_2024(self):
        """Test season detection for 2024 dates."""
        # September 2024 - start of 2024 season
        sept_2024 = datetime(2024, 9, 6)
        season, week = self.detector.get_season_info(sept_2024)
        self.assertEqual(season, 2024)
        self.assertGreaterEqual(week, 1)
        
        # December 2024 - mid season
        dec_2024 = datetime(2024, 12, 25)
        season, week = self.detector.get_season_info(dec_2024)
        self.assertEqual(season, 2024)
        self.assertGreaterEqual(week, 15)
        
        # January 2025 - still 2024 season
        jan_2025 = datetime(2025, 1, 15)
        season, _ = self.detector.get_season_info(jan_2025)
        self.assertEqual(season, 2024)
    
    def test_season_detection_2025(self):
        """Test season detection for 2025 season."""
        # September 2025 - start of 2025 season
        sept_2025 = datetime(2025, 9, 5)
        season, week = self.detector.get_season_info(sept_2025)
        self.assertEqual(season, 2025)
        self.assertGreaterEqual(week, 1)
    
    def test_week_cap_at_18(self):
        """Test that week is capped at 18."""
        # Far future date (well past week 18)
        future = datetime(2026, 2, 1)
        _, week = self.detector.get_season_info(future)
        self.assertLessEqual(week, 18, "Week should be capped at 18")
    
    def test_preseason_returns_week_1(self):
        """Test that pre-season dates return week 1."""
        # Before season start
        preseason = datetime(2025, 8, 1)
        _, week = self.detector.get_season_info(preseason)
        self.assertEqual(week, 1, "Pre-season should return week 1")
    
    def test_season_status_messages(self):
        """Test season status message generation."""
        # Current season
        status = self.detector.get_season_status(datetime(2025, 10, 22))
        self.assertIn("2025 NFL Season", status)
        self.assertIn("Week", status)
        
        # Pre-season
        preseason_status = self.detector.get_season_status(datetime(2025, 8, 1))
        self.assertIn("Pre-season", preseason_status)
    
    def test_auto_detect_config(self):
        """Test auto-detect configuration function."""
        from utils.season_detector import auto_detect_season_config
        config = auto_detect_season_config()
        
        self.assertIn('CURRENT_SEASON', config)
        self.assertIn('CURRENT_WEEK', config)
        self.assertIn('STATUS', config)
        self.assertIn('DETECTED_AT', config)
        
        self.assertIsInstance(config['CURRENT_SEASON'], int)
        self.assertIsInstance(config['CURRENT_WEEK'], int)
        self.assertGreaterEqual(config['CURRENT_WEEK'], 1)
        self.assertLessEqual(config['CURRENT_WEEK'], 18)


class TestWeekTransitionScenarios(unittest.TestCase):
    """Test specific week transition scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = NFLSeasonDetector()
    
    def test_thursday_game_week(self):
        """Test week during Thursday night game."""
        # Thursday of week 7 (games starting)
        thursday = datetime(2025, 10, 16)
        _, week = self.detector.get_season_info(thursday)
        # Thursday through Sunday should be next week (planning phase)
        self.assertEqual(week, 8)
    
    def test_sunday_game_week(self):
        """Test week during Sunday games."""
        sunday = datetime(2025, 10, 19)
        _, week = self.detector.get_season_info(sunday)
        # Sunday is still planning for upcoming week
        self.assertEqual(week, 8)
    
    def test_monday_night_football(self):
        """Test week during Monday Night Football."""
        monday = datetime(2025, 10, 20)
        _, week = self.detector.get_season_info(monday)
        # Monday games are for the current week (week 7)
        self.assertEqual(week, 7)


if __name__ == '__main__':
    unittest.main()
