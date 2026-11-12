"""Orbital Studio: atomic and molecular Hartree-Fock explorers."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Orbital Studio", page_icon="⚛", layout="wide")
page = st.navigation([
    st.Page(str(ROOT / "pages" / "atomic.py"), title="Atomic Orbital Explorer", icon="⚛", default=True),
    st.Page(str(ROOT / "pages" / "molecular.py"), title="Molecular Orbital Explorer", icon="🧬"),
], position="sidebar")
page.run()
