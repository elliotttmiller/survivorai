"""
Utility to auto-detect current NFL season and week based on current date.
Provides real-time season context without manual configuration.
"""
from datetime import datetime, timedelta
from typing import Tuple, Optional


class NFLSeasonDetector:
    """Auto-detect current NFL season and week from current date."""
    
    # NFL season typically starts first Thursday of September
    # 2024 season started September 5, 2024
    # 2025 season will start ~September 4, 2025
    
    SEASON_START_DATES = {
        2024: datetime(2024, 9, 5),
        2025: datetime(2025, 9, 4),
        2026: datetime(2026, 9, 10),
    }
    
    @classmethod
    def detect_current_season(cls, reference_date: Optional[datetime] = None) -> int:
        """
        Detect current NFL season based on date.
        
        Args:
            reference_date: Date to check (defaults to now)
            
        Returns:
            Current NFL season year
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # NFL season runs from September to February
        # If we're in Jan-Feb, it's the previous year's season
        # If we're in Mar-Aug, it's the upcoming season
        # If we're in Sep-Dec, it's the current year's season
        
        month = reference_date.month
        year = reference_date.year
        
        if month <= 2:  # Jan-Feb: previous year's season
            return year - 1
        elif month >= 9:  # Sep-Dec: current year's season
            return year
        else:  # Mar-Aug: upcoming season (use current year)
            return year
    
    @classmethod
    def detect_current_week(cls, reference_date: Optional[datetime] = None) -> int:
        """
        Detect current NFL week based on date.
        
        Args:
            reference_date: Date to check (defaults to now)
            
        Returns:
            Current NFL week (1-18), or 1 if pre-season
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        season = cls.detect_current_season(reference_date)
        
        # Get season start date
        if season in cls.SEASON_START_DATES:
            season_start = cls.SEASON_START_DATES[season]
        else:
            # Estimate: first Thursday of September
            season_start = cls._estimate_season_start(season)
        
        # If before season start, return week 1
        if reference_date < season_start:
            return 1
        
        # Calculate weeks since season start
        days_since_start = (reference_date - season_start).days
        week = (days_since_start // 7) + 1
        
        # Cap at week 18 (regular season)
        return min(18, max(1, week))
    
    @classmethod
    def _estimate_season_start(cls, year: int) -> datetime:
        """
        Estimate season start date for a year.
        
        NFL typically starts first Thursday of September.
        
        Args:
            year: Season year
            
        Returns:
            Estimated season start date
        """
        # Start with September 1st
        sep_first = datetime(year, 9, 1)
        
        # Find first Thursday (weekday 3)
        days_until_thursday = (3 - sep_first.weekday()) % 7
        if days_until_thursday == 0 and sep_first.weekday() != 3:
            days_until_thursday = 7
        
        first_thursday = sep_first + timedelta(days=days_until_thursday)
        return first_thursday
    
    @classmethod
    def get_season_info(cls, reference_date: Optional[datetime] = None) -> Tuple[int, int]:
        """
        Get current season and week.
        
        Args:
            reference_date: Date to check (defaults to now)
            
        Returns:
            Tuple of (season, week)
        """
        season = cls.detect_current_season(reference_date)
        week = cls.detect_current_week(reference_date)
        return season, week
    
    @classmethod
    def get_season_status(cls, reference_date: Optional[datetime] = None) -> str:
        """
        Get human-readable season status.
        
        Args:
            reference_date: Date to check (defaults to now)
            
        Returns:
            Season status string
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        season = cls.detect_current_season(reference_date)
        week = cls.detect_current_week(reference_date)
        
        if season in cls.SEASON_START_DATES:
            season_start = cls.SEASON_START_DATES[season]
        else:
            season_start = cls._estimate_season_start(season)
        
        if reference_date < season_start:
            days_until_start = (season_start - reference_date).days
            return f"Pre-season: {days_until_start} days until {season} season starts"
        elif week <= 18:
            return f"{season} NFL Season - Week {week}"
        else:
            return f"{season} NFL Season - Post-season"


def auto_detect_season_config() -> dict:
    """
    Auto-detect current season configuration.
    
    Returns:
        Dictionary with season and week
    """
    detector = NFLSeasonDetector()
    season, week = detector.get_season_info()
    
    return {
        'CURRENT_SEASON': season,
        'CURRENT_WEEK': week,
        'STATUS': detector.get_season_status(),
        'DETECTED_AT': datetime.now().isoformat()
    }


def test_season_detector():
    """Test season detection functionality."""
    detector = NFLSeasonDetector()
    
    print("NFL Season Detector Test")
    print("=" * 60)
    
    # Test current date
    season, week = detector.get_season_info()
    status = detector.get_season_status()
    
    print(f"\nCurrent Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Detected Season: {season}")
    print(f"Detected Week: {week}")
    print(f"Status: {status}")
    
    # Test specific dates
    test_dates = [
        datetime(2024, 9, 5),   # Start of 2024 season
        datetime(2024, 12, 25), # Mid-season
        datetime(2025, 2, 1),   # End of 2024 season
        datetime(2025, 9, 4),   # Start of 2025 season
    ]
    
    print("\n" + "-" * 60)
    print("Test Dates:")
    for test_date in test_dates:
        season, week = detector.get_season_info(test_date)
        print(f"{test_date.strftime('%Y-%m-%d')}: Season {season}, Week {week}")
    
    print("\n✓ Season detector test complete")


if __name__ == "__main__":
    test_season_detector()
