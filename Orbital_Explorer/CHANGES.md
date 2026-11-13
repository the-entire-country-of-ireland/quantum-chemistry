# Molecular rendering resolution (November 2026)

- Increased the default 3D molecular orbital grid from 33 to 49 samples per axis, for crisper phase isosurfaces and more faithful point-cloud spatial sampling.
- Increased default probability points from 18,000 to 26,000 per orbital. The display toggle applies to MO, both GVB orbitals, and optional VSEPR reference plots.
- Did not change atomic-mode resolution or introduce additional UI knobs. GVB-SCF energies and cached optimized orbitals are unaffected by rendering choices.

# Visualization restoration (November 2026)

- Restored a single molecular **Isosurface / Probability points** toggle.
- Both GVB pair orbitals (and MO and VSEPR reference orbitals) obey the same
  display choice. The point cloud samples `|psi(r)|^2`; blue/red retain the
  meaning of orbital phase, not opposite electron spins.
- Kept grid resolution, point count and phase thresholds as internal defaults.
  Toggling never triggers RHF/IBO recalculation or a new GVB-SCF optimization.
- Atomic explorer unchanged.

# Changes: all-orbital GVB-SCF (November 2026)

- Added `qchem_analysis/gvb_scf.py`: complete molecular-orbital-space, spin-adapted
  GVB-PP orbital optimization including occupied, core, and all external virtual
  mixing. No external/frozen orbital subspace is fixed in the energy objective.
- Retained the earlier active-pair optimizer only for deterministic startup;
  it is no longer used to report final GVB orbitals or energies.
- Added an analytic Coulomb orbital gradient and a stable Cayley-rotation
  parametrization; optimized pair occupations and spatial orbitals together.
- Updated MO/GVB page and exports to correctly represent all-space GVB orbitals.
- Added pair-label retention diagnostics for cases where fully optimized pairs
  move substantially from the original IBO chemical assignment.
- Preserved lone-pair selection and the hydrogen diatomic assignment fix.
- Preserved fixed nuclear geometries, a single RHF calculation, IBO localization,
  and the separate unchanged Atomic Orbital Explorer.
- Added independent finite-difference and Hubbard-dimer numerical tests, plus
  opt-in PySCF H₂ and HF calculations. No claim of real-molecule testing without
  PySCF is made.
