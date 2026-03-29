"""
CreditIQ — One-shot environment setup script.
Run this once after cloning the repo:

    python setup.py

It will:
1. Create a virtual environment
2. Install all dependencies
3. Copy .env.example → .env
4. Verify the folder structure
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(msg, color=GREEN):
    print(f"{color}{msg}{RESET}")


def run(cmd, cwd=ROOT):
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        log(f"  Command failed: {cmd}", RED)
        sys.exit(1)


def check_structure():
    required = [
        "data/raw", "data/processed", "data/external",
        "notebooks", "models/xgboost", "models/finbert",
        "models/lstm", "models/ensemble",
        "api/routers", "api/schemas", "api/services", "api/db",
        "dashboard/pages", "dashboard/components",
        "tests/api", "tests/models",
        "scripts", "docs", "mlflow_runs", ".github/workflows",
    ]
    all_ok = True
    for folder in required:
        path = ROOT / folder
        if path.exists():
            print(f"  {GREEN}✓{RESET}  {folder}")
        else:
            print(f"  {RED}✗{RESET}  {folder}  ← MISSING, creating...")
            path.mkdir(parents=True, exist_ok=True)
            (path / ".gitkeep").touch()
            all_ok = False
    return all_ok


def main():
    print(f"\n{BOLD}╔══════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║      CreditIQ — Project Setup        ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════╝{RESET}\n")

    # 1. Check Python version
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        log(f"Python 3.10+ required. You have {major}.{minor}", RED)
        sys.exit(1)
    log(f"[1/5] Python {major}.{minor} — OK")

    # 2. Create virtual environment
    venv_path = ROOT / "venv"
    if not venv_path.exists():
        log("[2/5] Creating virtual environment...")
        run(f"{sys.executable} -m venv venv")
    else:
        log("[2/5] Virtual environment already exists — skipping")

    # 3. Install dependencies
    log("[3/5] Installing dependencies (this may take a few minutes)...")
    pip = str(venv_path / "bin" / "pip") if os.name != "nt" else str(venv_path / "Scripts" / "pip")
    run(f'"{pip}" install --upgrade pip')
    run(f'"{pip}" install -r requirements.txt')

    # 4. Copy .env.example → .env
    env_src = ROOT / ".env.example"
    env_dst = ROOT / ".env"
    if not env_dst.exists():
        shutil.copy(env_src, env_dst)
        log("[4/5] Created .env from .env.example")
        log("      ⚠  Open .env and fill in your API keys before running!", YELLOW)
    else:
        log("[4/5] .env already exists — skipping")

    # 5. Verify folder structure
    log("[5/5] Verifying project structure...")
    all_ok = check_structure()

    # Done
    print()
    if all_ok:
        log("══════════════════════════════════════")
        log("  Setup complete! Next steps:")
        log("══════════════════════════════════════")
    print(f"""
  1. Activate your virtual environment:
     {YELLOW}source venv/bin/activate{RESET}          (Mac/Linux)
     {YELLOW}venv\\Scripts\\activate{RESET}             (Windows)

  2. Fill in your API keys:
     {YELLOW}nano .env{RESET}   (or open in VS Code)

  3. Move your Home Credit CSV files to:
     {YELLOW}data/raw/{RESET}

  4. Launch Jupyter to start EDA:
     {YELLOW}jupyter notebook{RESET}

  5. When ready for the API:
     {YELLOW}uvicorn api.main:app --reload{RESET}
""")


if __name__ == "__main__":
    main()
