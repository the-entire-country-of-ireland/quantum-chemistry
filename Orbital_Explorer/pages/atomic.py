"""Isolated-atom RHF/ROHF orbital and hybrid exploration."""
from pathlib import Path

import numpy as np
import streamlit as st

from qchem_analysis.atomic_orbitals import (
    make_equivalent_hybrids,
    plot_atomic_orbital_3d,
)
from qchem_analysis.hf_calculations import ELEMENTS
from qchem_analysis.molecular_hf import AtomicLibrary

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "saved_atomic_hf"
BASIS_OPTIONS = ["cc-pVDZ", "cc-pVTZ", "6-31g", "sto-3g"]

st.title("Atomic Orbital Explorer")
st.caption("Explore isolated-atom Hartree–Fock orbitals, valence sp / sp² / sp³ hybrids, radial penetration, and wavefunction phase.")

@st.cache_resource(show_spinner=False)
def get_atom(z: int, basis: str):
    return AtomicLibrary.prepare([z], basis=basis, directory=CACHE).atoms[z]

with st.sidebar:
    st.header("Atomic explorer settings")
    element = st.selectbox(
        "Element", options=list(ELEMENTS),
        format_func=lambda z: f"{ELEMENTS[z][0]} · Z={z}",
        index=5,
        key="atomic_element",
    )
    basis = st.selectbox("Gaussian basis", BASIS_OPTIONS, index=0, key="atomic_basis")

try:
    with st.spinner("Loading saved atomic orbitals or solving isolated-atom HF…"):
        atom = get_atom(element, basis)
except Exception as exc:
    st.error(f"Could not load or calculate atomic orbitals: {exc}")
    st.stop()

shells = dict(atom.shell_orbitals)
hybrids = {}
if atom.principal_n >= 2:
    for count, name in ((2, "sp"), (3, "sp²"), (4, "sp³")):
        try:
            for key, val in make_equivalent_hybrids(atom, count).items():
                hybrids[key.replace("sp2", "sp²").replace("sp3", "sp³")] = val
        except (ValueError, RuntimeError):
            pass

spin_records = atom.spin_orbitals
canonical = {}
for record in spin_records:
    k = record["spatial_orbital"]
    spin = record["spin"]
    status = "occupied" if record["occupied"] else "unoccupied"
    canonical[f"MO {k+1} {spin} · {status}"] = record["coefficients"]

with st.sidebar:
    st.divider()
    family = st.radio(
        "Orbital family", ["Atomic shells", "Valence hybrids", "Canonical spin orbitals"],
        horizontal=False, key="atomic_family",
    )
    options = {"Atomic shells": shells, "Valence hybrids": hybrids,
               "Canonical spin orbitals": canonical}[family]
    if not options:
        st.info("This atom does not have an available valence s/p hybrid set.")
        st.stop()
    orbital_label = st.selectbox("Orbital", list(options), key="atomic_orbital")
    st.divider()
    mode = st.radio("Visualization", ["Probability points", "Isosurface"],
                    key="atomic_render_mode")
    selected_n = (int(orbital_label[0]) if family == "Atomic shells"
                  and orbital_label[0].isdigit() else atom.principal_n)
    suggested_extent = 4.0 if selected_n == 1 else (7.0 if selected_n == 2 else 11.0)
    extent = st.slider("Half-width (Bohr)", 2.0, 20.0, suggested_extent,
                       0.5, key="atomic_extent")
    resolution = st.slider("Grid resolution per axis", 25, 91, 51, 2,
                           key="atomic_resolution")
    if mode == "Probability points":
        point_count = st.slider("Sampled points", 1000, 100000, 30000, 1000,
                                key="atomic_point_count")
        coverage = 0.95
    else:
        coverage_percent = st.slider("Enclosed in-box probability (%)", 60, 99, 95, 1,
                                     key="atomic_probability_coverage")
        coverage = coverage_percent / 100.0
        point_count = 30000
    seed = st.number_input("Sampling seed", min_value=0, max_value=2**31 - 1,
                           value=12345, key="atomic_seed")

st.markdown(f"### {atom.symbol} · {orbital_label}")
left, right, third = st.columns(3)
left.metric("Atomic number", atom.atomic_number)
right.metric("Valence shell", f"n = {atom.principal_n}")
third.metric("Atomic HF basis", basis)

try:
    with st.spinner("Evaluating the wavefunction and rendering 3D orbital…"):
        fig = plot_atomic_orbital_3d(
            atom, np.asarray(options[orbital_label]), orbital_label,
            grid_points=int(resolution), extent_bohr=float(extent),
            render_mode="points" if mode == "Probability points" else "surface",
            probability_coverage=float(coverage), point_count=int(point_count),
            random_seed=int(seed),
        )
    fig.update_layout(height=670)
    st.plotly_chart(fig, use_container_width=True, key="atomic_plot")
except Exception as exc:
    st.error(f"Orbital rendering failed: {exc}")

st.caption("Blue/red encode wavefunction sign, not spin. Point density approximates |ψ(r)|² within the plotting volume. The surface threshold is based on contained grid probability, not peak amplitude.")

with st.expander("Orbital and spin information"):
    if family == "Canonical spin orbitals":
        k = list(canonical).index(orbital_label)
        record = spin_records[k]
        st.write(f"Energy: **{record['energy_hartree']:.6f} Ha** · Spin: **{record['spin']}** · Occupied: **{record['occupied']}**")
        st.caption("For open-shell atoms, ROHF canonical spin records share spatial MO coefficients. Occupation labels follow the saved ROHF convention.")
    else:
        st.write("These are spatial orbitals. Each may be combined with alpha or beta spin functions; occupancy is determined by the atomic electronic configuration.")
    if family == "Valence hybrids":
        st.write("Hybrids are orthonormal combinations of the saved valence s and p orbitals. Their orientation is conventional until aligned to bonds in the molecular calculation.")

filename = CACHE / f"Z{element}_{basis.replace('/', '_')}.npz"
if filename.is_file():
    st.download_button("Download saved atomic HF coefficients (.npz)",
                       data=filename.read_bytes(), file_name=filename.name,
                       mime="application/octet-stream", key="atomic_download")
st.caption("Atomic calculations are cached in saved_atomic_hf/ and reused by the Molecular Orbital Explorer when element and basis match.")
