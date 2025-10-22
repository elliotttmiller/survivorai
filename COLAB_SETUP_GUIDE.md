# Google Colab Setup Guide for SurvivorAI v2.0

Quick start guide for using the enhanced notebook with automatic configuration.

## 🚀 Quick Setup (5 Minutes)

### Step 1: Set Up API Key in Colab Secrets
1. Open the notebook in Google Colab
2. Click the **🔑 key icon** in the left sidebar
3. Click **"+ Add new secret"**
4. Enter:
   - **Name**: `ODDS_API_KEY`
   - **Value**: Your API key from [theoddsapi.com](https://the-odds-api.com/)
5. Toggle **"Notebook access"** to **ON**
6. ✅ Done! The notebook will automatically load your key

**Get a free API key:**
- Visit https://the-odds-api.com/
- Sign up (free tier: 500 requests/month)
- Copy your API key

### Step 2: Configure Your Used Teams
The notebook includes a `used_teams.json` file with default teams. Update it with your actual picks:

**Option A: Edit in Colab** (Easiest)
1. Click the **📁 Files icon** in the left sidebar
2. Navigate to `/content/survivorai/used_teams.json`
3. Right-click → **"Open in editor"**
4. Update your teams (format below)
5. Save with **Ctrl+S** (Windows/Linux) or **Cmd+S** (Mac)

**Option B: Edit on GitHub**
1. Go to your fork: `github.com/YOUR_USERNAME/survivorai`
2. Click `used_teams.json`
3. Click the **✏️ pencil icon** to edit
4. Update your teams
5. **Commit changes**
6. Re-run notebook setup cells to pull updates

**File Format:**
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

### Step 3: Run the Notebook
1. Click **Runtime** → **Run all**
2. When prompted, enter your **pool size** (e.g., 150)
3. That's it! View your professional recommendations

## 📊 What You'll See

### Professional Output Features

**Automatic Configuration:**
```
⚙️ System Configuration
============================================================

🔑 The Odds API Configuration
   Loading API key from Colab secrets...
   ✅ API key loaded successfully from secrets!
   Key length: 32 characters
```

**Beautiful Tables:**
```
┌─────────┬─────────────────────────────────────────┐
│  Week   │  Team Selected                          │
├─────────┼─────────────────────────────────────────┤
│ Week 1  │ Denver Broncos                          │
│ Week 2  │ Dallas Cowboys                          │
│ Week 3  │ Tampa Bay Buccaneers                    │
└─────────┴─────────────────────────────────────────┘
```

**Professional Dashboard:**
```
╔════════════════════════════════════════════════════════════╗
║              📊 OPTIMAL RECOMMENDATIONS                    ║
╚════════════════════════════════════════════════════════════╝

🏆 RECOMMENDATION #1

📍 Week 8 Selection:
   ╔══════════════════════════════════════════════╗
   ║  🏈 PICK: Kansas City Chiefs                 ║
   ║  🆚 VS:   Denver Broncos                     ║
   ║  📊 WIN PROBABILITY:  85.3%                  ║
   ╚══════════════════════════════════════════════╝
```

## 🔧 Troubleshooting

### "API key not found" message
- Check that you added the secret in Colab (🔑 icon)
- Verify the name is exactly `ODDS_API_KEY` (case-sensitive)
- Make sure "Notebook access" is toggled ON
- If still issues, notebook will use demo mode automatically

### "used_teams.json not found"
- The notebook creates this file automatically on first run
- If you deleted it, just re-run the cell to recreate
- Default teams are provided as examples

### Teams not loading correctly
- Check JSON syntax (use https://jsonlint.com/)
- Ensure team names match exactly (see `USED_TEAMS_README.md`)
- Make sure quotes are double quotes `"`, not single `'`
- Verify commas between entries (no comma after last entry)

### Notebook doesn't update after editing files
- Re-run the specific cell that loads the file
- Or click **Runtime** → **Restart and run all**

## 📝 Updating Your Teams Each Week

After making your weekly pick:

1. Open `used_teams.json` in Colab file browser
2. Add new entry for the current week:
   ```json
   "8": "Kansas City Chiefs"
   ```
3. Save the file
4. Re-run optimization cells

**Or edit on GitHub and pull changes next run.**

## 🎯 Weekly Workflow

1. ✅ **Monday-Tuesday**: Check when notebook was last run
2. ✅ **Wednesday**: Run notebook for latest odds and recommendations
3. ✅ **Thursday-Sunday**: Make your pick before game time
4. ✅ **After pick**: Update `used_teams.json` with your selection
5. ✅ **Next week**: Repeat!

## 💡 Pro Tips

### Tip 1: Keep a Backup
Save a copy of your `used_teams.json` file outside Colab. If the session ends, you can quickly restore it.

### Tip 2: Run Early in Week
Odds and data are most accurate Wednesday-Thursday. Run the notebook then for best results.

### Tip 3: Compare Recommendations
The notebook shows 5 recommendations. Review all before deciding, especially in large pools.

### Tip 4: Check Pool Size
Make sure your pool size is current. People get eliminated each week, so the effective pool size shrinks.

### Tip 5: Monitor API Usage
Free tier = 500 requests/month. Each run uses ~10-20 requests. You can run ~25-50 times per month.

## 📚 Additional Resources

- **Full Documentation**: `USED_TEAMS_README.md` - Complete team names list
- **Change Log**: `NOTEBOOK_CHANGES.md` - Detailed before/after comparison
- **Architecture**: `ARCHITECTURE.md` - Technical details
- **Research**: `RESEARCH_REPORT.md` - Model selection rationale

## 🆘 Need Help?

1. Check the documentation files above
2. Review error messages carefully
3. Try the troubleshooting section
4. Open an issue on GitHub with:
   - Error message
   - What you were doing
   - Screenshots if possible

---

**Last Updated**: October 22, 2025  
**Version**: SurvivorAI v2.0 Enhanced  
**Notebook**: Professional Edition with Auto-Configuration

🏈 Good luck with your survivor pool! 🏆
