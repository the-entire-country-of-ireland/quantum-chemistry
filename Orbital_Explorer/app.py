"""Orbital Studio: atomic orbitals and molecular MO/GVB-PP bond interpretations."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Orbital Studio", page_icon="⚛", layout="wide")
page = st.navigation([
    st.Page(str(ROOT / "pages" / "atomic.py"), title="Atomic Orbital Explorer", icon="⚛", default=True),
    st.Page(str(ROOT / "pages" / "molecular.py"), title="Molecular Bond Explorer", icon="🧬"),
], position="sidebar")
page.run()
