# Survivor AI - Examples

This directory contains example scripts demonstrating various features of Survivor AI.

## Available Examples

### injury_analysis_example.py

Comprehensive examples of the injury analysis system (v3.0).

**Run the example:**
```bash
python examples/injury_analysis_example.py
```

**What you'll learn:**
1. Basic injury impact analysis
2. Win probability adjustments based on injuries
3. Game data enrichment with injury features
4. ML feature engineering integration
5. Position impact weights and calculations

**Topics Covered:**
- Position-based impact scoring (QB=1.0, RB=0.35, etc.)
- Injury severity classification (OUT, DOUBTFUL, QUESTIONABLE)
- Win probability adjustments (±15% max)
- Critical injury identification
- Integration with ML prediction pipeline

## Running Examples

All examples can be run from the repository root:

```bash
# Run from repository root
cd /path/to/survivorai
python examples/injury_analysis_example.py
```

## Example Output

The injury analysis example produces detailed output showing:
- Impact scores for different injury scenarios
- Win probability adjustments
- Feature integration with ML models
- Position weight explanations

Expected runtime: < 1 second

## Documentation

For detailed technical documentation, see:
- [INJURY_ANALYSIS.md](../INJURY_ANALYSIS.md) - Complete injury analysis docs
- [README.md](../README.md) - Main project documentation
- [FEATURES.md](../FEATURES.md) - Feature list and descriptions

## Requirements

Examples use the same dependencies as the main project:
- pandas
- numpy

No additional installation required if you've already installed requirements.txt

## Contributing

To add new examples:
1. Create a new Python file in this directory
2. Follow the existing example structure
3. Include clear docstrings and comments
4. Update this README
5. Test thoroughly before committing

## Support

For questions or issues with examples:
- Check the main documentation
- Review the example code comments
- Open an issue on GitHub
