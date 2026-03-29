#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# git_init.sh — Initialize Git repo and push to GitHub
#
# Usage:
#   chmod +x scripts/git_init.sh
#   ./scripts/git_init.sh
#
# You'll be prompted for your GitHub username and repo name.
# Make sure you've already created an empty repo on GitHub first.
# ─────────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}CreditIQ — Git Setup${NC}\n"

# Check git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Install it first."
    exit 1
fi

# Prompt for GitHub details
read -p "GitHub username: " GH_USER
read -p "Repository name (e.g. creditiq): " REPO_NAME
read -p "Your name (for git config): " GIT_NAME
read -p "Your email (for git config): " GIT_EMAIL

REMOTE_URL="https://github.com/${GH_USER}/${REPO_NAME}.git"

echo -e "\n${YELLOW}Setting up Git...${NC}"

# Configure git identity
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

# Initialize repo (if not already)
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✓ Git repo initialized${NC}"
else
    echo "Git repo already initialized"
fi

# Initial commit
git add .
git commit -m "feat: initial project structure

- Full folder layout: data, notebooks, models, api, dashboard, tests
- requirements.txt with all dependencies pinned
- config.py with pydantic-settings
- .env.example template
- .gitignore (data and model files excluded)
- docker-compose.yml (API + Dashboard + PostgreSQL + Redis)
- Dockerfile.api and Dockerfile.dashboard
- GitHub Actions CI/CD workflow
- README.md with architecture and quick start"

# Add remote and push
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
git branch -M main
git push -u origin main

echo -e "\n${GREEN}✓ Pushed to: ${REMOTE_URL}${NC}"
echo -e "${GREEN}✓ Done! Your CreditIQ repo is live on GitHub.${NC}\n"
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. Fill in .env with your API keys"
echo "  3. Move Home Credit CSVs to data/raw/"
echo "  4. jupyter notebook"
