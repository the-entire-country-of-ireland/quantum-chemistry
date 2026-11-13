# VB-centered Quantum Bonding Notes (November 2026)

Primary files:
- `Quantum_Bonding_VB_Centered_2026.tex`: 27-page LaTeX source.
- `Quantum_Bonding_VB_Centered_2026.pdf`: compiled manuscript.
- `figures/`: generated figures, including 2e singlet/triplet VB configuration-space cuts and covalent/ionic mixing.

Reproduce figure generation from this directory:
1. `python make_figures.py`
2. `python make_revised_figures.py`
3. `python make_vb_figures.py`
4. `pdflatex -interaction=nonstopmode Quantum_Bonding_VB_Centered_2026.tex` (run twice)

Scripts depend on numpy, scipy, matplotlib; the optional `rdkit_eht.py` cross-check depends on RDKit.

Model validation:
- The 1D double delta well and discrete Hubbard dimer are exactly solvable within their stated Hamiltonians.
- The 3D full-valence extended Huckel computations capture orbital/bond topology and a qualitative C–O polarization trend. They **do not** compute molecular bond dissociation energies or an optimized VB Hamiltonian.
- The σ/π spin-coupled VB wavefunctions and the covalent/ionic matrix structures are explicit. The polyatomic two-electron VB integrals are specified but **not solved numerically**; values from one-electron localized orbitals are never equated with VB resonance weights.
- The original point-screened-Coulomb 2s/2p example is retained only as a historically informative, superseded approximation in the appendix.
