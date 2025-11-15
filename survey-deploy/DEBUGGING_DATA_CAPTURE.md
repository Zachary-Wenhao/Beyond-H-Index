# Survey Data Capture - Debugging Guide

## ✅ Issue Fixed: Missing "Overall Confidence" Field

### What Was Wrong

The survey was collecting the "Overall Confidence" question (final-q3) but **not sending it to Google Sheets**. 

The code was:
- ✅ Saving it locally in the browser
- ✅ Including it in downloaded JSON files
- ❌ **NOT including it in the Google Sheets submission**

### What Was Fixed

Added the missing field to the Google Sheets submission:

```javascript
formattedResponses['final-q3'] = responses.final?.overallConfidence || '';
```

Now all 4 final questions are being sent:
- `final-q1`: Top 3 factors that influenced rankings (array of checkboxes)
- `final-q2`: Additional information needed (text area)
- `final-q3`: Overall confidence level (radio button) **← THIS WAS MISSING**
- `career-stage`: Career stage (radio button)

---

## 🔍 How to Test & Verify

### Step 1: Deploy the Fixed Version

```bash
cd /Users/divyangoyal/Desktop/Beyond-H-Index-deploy
cp /Users/divyangoyal/Desktop/H-Index/survey-deploy/index.html survey-deploy/
cp /Users/divyangoyal/Desktop/H-Index/survey-deploy/survey_china.html survey-deploy/
git add survey-deploy/
git commit -m "Fix: Add missing final-q3 (overall confidence) to Google Sheets submission"
git push origin main
```

Wait 1-2 minutes for GitHub Pages to update.

### Step 2: Test Data Submission

1. **Open the survey in an incognito/private window**:
   ```
   https://zachary-wenhao.github.io/Beyond-H-Index/survey-deploy/
   ```

2. **Open browser console** (F12 or Cmd+Option+J)

3. **Complete a quick test survey** (you can use dummy data)

4. **At the final submission**, check the console output. You should see:
   ```javascript
   Final questions in submission: {
     final-q1: ["total-citations", "career-length", ...],  // array of selected factors
     final-q2: "Some text here...",                        // text response
     final-q3: "very-confident",                           // ← THIS SHOULD NOW APPEAR!
     career-stage: "graduate-student"                      // career stage
   }
   ```

### Step 3: Verify in Google Sheets

After submission, check your Google Sheets to confirm these columns are populated:
- `final-q1` (should show comma-separated values)
- `final-q2` (should show text)
- `final-q3` (should show confidence level) **← CHECK THIS ONE**
- `career-stage` (should show career stage)

---

## 🔧 Google Apps Script Backend

### Check Your Apps Script

Make sure your Google Apps Script backend is configured to receive these fields.

**Your Apps Script URL:**
```
https://script.google.com/macros/s/AKfycbwkkflMcmFgsMQnVEIU4j-kGHO9Kg3ZFGm8Lw1aGTaUb28qyuouHKYpj83BzGDWQNLv/exec
```

### Expected Column Names in Sheet

Your Google Sheet should have these columns for final questions:
- `final-q1` or `Top Factors`
- `final-q2` or `Additional Info`
- `final-q3` or `Overall Confidence` **← Make sure this column exists!**
- `career-stage` or `Career Stage`

### Sample Apps Script Code

If you need to update your Apps Script, here's the relevant part:

```javascript
function doPost(e) {
  const data = JSON.parse(e.postData.contents);
  const responses = data.responses;
  
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  
  // Add row with all data including final questions
  sheet.appendRow([
    data.respondent_id,
    data.timestamp,
    // ... comparison data ...
    Array.isArray(responses['final-q1']) ? responses['final-q1'].join(', ') : '',
    responses['final-q2'] || '',
    responses['final-q3'] || '',  // ← Make sure this is included!
    responses['career-stage'] || ''
  ]);
  
  return ContentService.createTextOutput(JSON.stringify({status: 'success'}));
}
```

---

## 📊 Data Format Reference

### Final Question Field Names & Values

**final-q1** (Top 3 factors - checkbox array):
- `total-citations`
- `citations-per-paper`
- `career-length`
- `recent-activity`
- `influential-percentage`
- `publication-count`
- `field-expertise`
- `career-trajectory`

**final-q2** (Additional info - text):
- Free text string

**final-q3** (Overall confidence - radio):
- `very-confident`
- `confident`
- `somewhat-confident`
- `not-very-confident`
- `uncertain`

**career-stage** (Career stage - radio):
- `graduate-student`
- `postdoc`
- `early-career-faculty`
- `mid-career-faculty`
- `senior-faculty`
- `industry-researcher`
- `industry-engineer`
- `industry-data-scientist`
- `other`

---

## 🐛 Troubleshooting

### "Still not seeing final-q3 in sheets"

1. **Clear browser cache** and reload the survey
2. **Check console** - is final-q3 showing in the console.log?
3. **Check Apps Script** - does it have a column for final-q3?
4. **Check Sheet columns** - is there a column header for final-q3?

### "Console shows final-q3 but sheet doesn't"

The issue is in your **Google Apps Script**, not the survey HTML. You need to:
1. Go to your Apps Script editor
2. Add `responses['final-q3']` to the sheet.appendRow() call
3. Redeploy the script

### "How do I check the Apps Script?"

1. Go to your Google Sheet
2. Extensions → Apps Script
3. Review the `doPost` function
4. Make sure it includes all 4 final question fields

---

## ✨ Summary

**What changed:**
- Added `formattedResponses['final-q3']` to both `index.html` and `survey_china.html`
- Updated console.log to show final-q3 for debugging

**What you need to do:**
1. Deploy the updated files
2. Test a submission and check console
3. Verify final-q3 appears in your Google Sheet
4. Update Apps Script if needed (add final-q3 column)

---

**Last Updated:** November 7, 2025

