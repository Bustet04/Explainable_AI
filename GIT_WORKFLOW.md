# Git Workflow Guide

## Current Repository Structure

Your Git repo is now organized with a clean structure. Here's what to commit:

### ✅ Ready to Commit (New Clean Structure)
```
.gitignore           # Excludes data, models, old Code/ folder
README.md            # Comprehensive project documentation
requirements.txt     # Python dependencies
src/                 # Source code modules
  ├── preprocessing.py
  ├── features.py
  ├── model.py
  ├── train_autoencoder.py (TBD - copy from notebook)
  ├── clustering.py (TBD - copy from notebook)
  └── explainability.py (TBD - copy from notebook)
docs/
  └── report.md      # Analysis report template
```

### ❌ Excluded from Git (via .gitignore)
```
Code/                # Old folder structure (kept locally)
Excercises/
Mental_Disorders/
Research/
Skript/
Dokumentation/
data/processed/*.npy # Large data files
models/*.joblib      # Large model files
.venv/               # Virtual environment
__pycache__/         # Python cache
```

---

## Committing Workflow

### Step 1: Stage Files You Want to Commit

**Option A: Stage everything ready to commit**
```powershell
git add .gitignore README.md requirements.txt
git add src/
git add docs/
```

**Option B: Stage specific files one by one**
```powershell
git add .gitignore
git add README.md
git add requirements.txt
git add src/preprocessing.py
git add src/features.py
git add src/model.py
git add docs/report.md
```

### Step 2: Check What's Staged
```powershell
git status
```

You should see files in "Changes to be committed" section.

### Step 3: Commit with Message
```powershell
git commit -m "Initial commit: project structure with src modules and documentation"
```

### Step 4: Push to GitHub
```powershell
git push -u origin master
```

(You may be prompted for GitHub authentication)

---

## Recommended Commit Strategy

**Commit 1: Project Structure**
```powershell
git add .gitignore README.md requirements.txt
git commit -m "Add project structure: README, requirements, gitignore"
git push -u origin master
```

**Commit 2: Source Code Modules**
```powershell
git add src/
git commit -m "Add source code modules: preprocessing, features, model"
git push
```

**Commit 3: Documentation**
```powershell
git add docs/
git commit -m "Add analysis report template"
git push
```

**Future Commit: Notebooks**
```powershell
# After copying notebooks to notebooks/ folder
git add notebooks/
git commit -m "Add Jupyter notebooks for analysis pipeline"
git push
```

---

## Next Steps

### 1. Copy Your Notebooks
Move your important notebooks to the `notebooks/` folder:
```powershell
Copy-Item "Code\Projekt\train_unsupervised_sleep_clustering.ipynb" "notebooks\autoencoder_clustering.ipynb"
Copy-Item "Code\Projekt\train_sleep_classifier.ipynb" "notebooks\" -ErrorAction SilentlyContinue
```

### 2. Create Python Scripts from Notebooks
Extract code from notebooks into `src/` modules:
- `src/train_autoencoder.py` - Training script
- `src/clustering.py` - PCA + K-Means
- `src/explainability.py` - XAI analysis

### 3. Fill in Report Template
After running analysis, update `docs/report.md` with actual results.

---

## Git Best Practices

✅ **DO:**
- Commit often with clear messages
- Use meaningful commit messages (e.g., "Add XAI analysis for cluster explainability")
- Test code before committing
- Keep commits focused (one feature/fix per commit)

❌ **DON'T:**
- Commit large data files (they're in .gitignore)
- Commit model weights (also in .gitignore)
- Commit broken/untested code
- Use vague messages like "update" or "fix"

---

## Useful Git Commands

```powershell
# See what changed
git status
git diff

# View commit history
git log --oneline

# Undo staged file (before commit)
git reset HEAD <file>

# Undo changes to file
git checkout -- <file>

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout master
```

---

## GitHub Repository

Your repo: https://github.com/Bustet04/Explainable_AI

After pushing, you'll be able to:
- View code online
- Share with collaborators
- Track issues
- Create pull requests
- Enable GitHub Pages for documentation

---

**Ready to commit!** Start with:
```powershell
git add .gitignore README.md requirements.txt src/ docs/
git commit -m "Initial commit: project structure and core modules"
git push -u origin master
```
