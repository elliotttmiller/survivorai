# Used Teams Configuration

This file (`used_teams.json`) tracks which NFL teams you've already selected in your Survivor Pool for each week.

## Purpose

The `used_teams.json` file allows you to:
- **Avoid manual data entry** - No need to type teams every time you run the notebook
- **Maintain consistency** - Single source of truth for your picks
- **Easy updates** - Simple JSON format for quick editing
- **Automatic loading** - Notebook reads this file automatically

## File Format

The file uses a simple JSON structure with week numbers as keys and team names as values:

```json
{
  "1": "Denver Broncos",
  "2": "Dallas Cowboys",
  "3": "Tampa Bay Buccaneers",
  "4": "Detroit Lions",
  "5": "Indianapolis Colts",
  "6": "Green Bay Packers",
  "7": "Carolina Panthers"
}
```

## How to Update

### Option 1: Edit Directly in Google Colab

1. In the Colab notebook, click the 📁 Files icon in the left sidebar
2. Navigate to `/content/survivorai/used_teams.json`
3. Right-click and select "Open in editor"
4. Make your changes
5. Save (Ctrl+S or Cmd+S)

### Option 2: Edit on GitHub

1. Go to your repository: `github.com/elliotttmiller/survivorai`
2. Navigate to `used_teams.json`
3. Click the ✏️ pencil icon to edit
4. Make your changes
5. Commit the changes
6. Re-run the notebook setup cells to pull the latest version

### Option 3: Manual JSON Edit

Edit the file directly with any text editor, following the format:

```json
{
  "week_number": "Team Full Name",
  "1": "Kansas City Chiefs",
  "2": "Buffalo Bills"
}
```

## Team Name Format

**Important**: Use the official NFL team names exactly as they appear below:

### AFC East
- Buffalo Bills
- Miami Dolphins
- New England Patriots
- New York Jets

### AFC North
- Baltimore Ravens
- Cincinnati Bengals
- Cleveland Browns
- Pittsburgh Steelers

### AFC South
- Houston Texans
- Indianapolis Colts
- Jacksonville Jaguars
- Tennessee Titans

### AFC West
- Denver Broncos
- Kansas City Chiefs
- Las Vegas Raiders
- Los Angeles Chargers

### NFC East
- Dallas Cowboys
- New York Giants
- Philadelphia Eagles
- Washington Commanders

### NFC North
- Chicago Bears
- Detroit Lions
- Green Bay Packers
- Minnesota Vikings

### NFC South
- Atlanta Falcons
- Carolina Panthers
- New Orleans Saints
- Tampa Bay Buccaneers

### NFC West
- Arizona Cardinals
- Los Angeles Rams
- San Francisco 49ers
- Seattle Seahawks

## Examples

### Starting Fresh (Week 1)
```json
{}
```

### Mid-Season (Week 8)
```json
{
  "1": "Kansas City Chiefs",
  "2": "San Francisco 49ers",
  "3": "Buffalo Bills",
  "4": "Dallas Cowboys",
  "5": "Miami Dolphins",
  "6": "Detroit Lions",
  "7": "Philadelphia Eagles"
}
```

### Late Season (Week 15)
```json
{
  "1": "Kansas City Chiefs",
  "2": "San Francisco 49ers",
  "3": "Buffalo Bills",
  "4": "Dallas Cowboys",
  "5": "Miami Dolphins",
  "6": "Detroit Lions",
  "7": "Philadelphia Eagles",
  "8": "Baltimore Ravens",
  "9": "Cincinnati Bengals",
  "10": "Jacksonville Jaguars",
  "11": "Los Angeles Chargers",
  "12": "Seattle Seahawks",
  "13": "Minnesota Vikings",
  "14": "Tampa Bay Buccaneers"
}
```

## Tips

1. **Keep it Updated**: Update this file immediately after making each week's selection
2. **Backup Your Data**: Consider keeping a backup copy of this file
3. **Verify Names**: Double-check team names match the official list above
4. **Sequential Weeks**: While not required, keeping weeks in numerical order helps readability

## Troubleshooting

### "Team not found" Error
- Verify the team name exactly matches the official list
- Check for typos or extra spaces
- Ensure proper capitalization

### "Invalid JSON" Error
- Ensure all strings are in double quotes, not single quotes
- Verify commas between entries (but not after the last entry)
- Check that braces `{}` are properly matched
- Use a JSON validator: https://jsonlint.com/

### File Not Loading
- Ensure the file is named exactly `used_teams.json`
- Verify it's in the correct directory: `/content/survivorai/`
- Check file permissions (should be readable)

## Integration with Notebook

The notebook automatically:
1. Checks for the existence of `used_teams.json`
2. Loads and parses the JSON data
3. Displays your previous picks in a formatted table
4. Uses this data to exclude teams from recommendations
5. Creates a default file with sample data if none exists

## Security Note

This file contains no sensitive information - it's just a list of team names and week numbers. It's safe to:
- Commit to version control
- Share with others
- Store in public repositories

---

**Last Updated**: October 2025  
**SurvivorAI Version**: 2.0
