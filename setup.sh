#!/usr/bin/env bash
# Run ONCE on each machine after git clone / git pull:
#   bash setup.sh
set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PY="$PROJECT_DIR/.venv/bin/python"

# 1) Create the virtual environment if it does not exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# 2) Install dependencies
"$PY" -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  "$PY" -m pip install -r requirements.txt
fi
"$PY" -m pip install jupyter pyyaml ipykernel

# 3) Register the Jupyter kernel (name must match _quarto.yml)
"$PY" -m ipykernel install --user --name tp-ml-ia-fiable

# 4) Point RStudio to this venv's Python, PER PROJECT
#    (writes .Renviron in the project folder, not ~/.Renviron)
echo "QUARTO_PYTHON=$PY" > .Renviron

# 5) Keep .Renviron out of GitHub (it contains machine-specific paths)
touch .gitignore
grep -qxF '.Renviron' .gitignore || echo '.Renviron' >> .gitignore

echo ""
echo "Done. Quit RStudio completely (Cmd+Q), then open the project"
echo "via the .Rproj file (or File > Open Project) and click Render."