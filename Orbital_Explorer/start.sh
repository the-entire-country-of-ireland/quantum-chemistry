#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! python -m streamlit --version >/dev/null 2>&1; then
  echo "Missing dependencies. Install with: python -m pip install -r requirements.txt"
  exit 1
fi
exec python -m streamlit run app.py
