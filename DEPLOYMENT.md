# Deployment Guide

## Quick Deploy to GitHub Pages

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something professional like: `research-impact-survey` or `cmu-impact-study`
3. Make it **public** (required for free GitHub Pages)
4. Don't initialize with README (we already have one)

### Step 2: Deploy from Terminal

```bash
# Navigate to survey directory
cd /Users/divyangoyal/Desktop/H-Index/survey-deploy

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial survey deployment"

# Connect to your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (in left sidebar)
3. Under "Source", select:
   - Branch: `main`
   - Folder: `/ (root)`
4. Click **Save**
5. Wait 1-2 minutes for deployment

Your survey will be live at:
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

## Making Updates After Deployment

```bash
# Make your changes to the HTML files
# Then commit and push:

git add .
git commit -m "Updated survey questions"
git push

# Changes go live in 1-2 minutes
```

## Testing Before Sharing

1. Open the live URL in an **incognito/private window**
2. Complete a full test run
3. Check that responses download correctly
4. Test on mobile devices if needed

## Custom Domain (Optional)

If you want `yoursurvey.com` instead of `github.io`:
1. Purchase domain from Namecheap/Google Domains
2. In GitHub Settings → Pages, add custom domain
3. Update DNS settings as instructed

## Professional URL Tips

- Use a descriptive repository name: `research-impact-survey`
- Create GitHub organization account for cleaner URLs
- Consider: `cmu-nlp-impact-study`, `researcher-evaluation-survey`, etc.

## Troubleshooting

**Survey not loading?**
- Wait 2-3 minutes after enabling Pages
- Check that `index.html` is in root directory
- Clear browser cache

**Changes not showing?**
- Wait 1-2 minutes after push
- Clear browser cache (Cmd+Shift+R)
- Check GitHub Actions for deployment status

## Data Collection

Participants download their responses as JSON files. To analyze:
1. Collect JSON files via email
2. Use Python/R to parse and aggregate
3. See main project for analysis scripts

---

**Need Help?** Check GitHub Pages documentation or open an issue in the repository.


