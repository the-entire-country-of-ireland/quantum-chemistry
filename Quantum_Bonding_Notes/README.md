# From Atomic Orbitals to Quantum Bonds

This bundle contains graduate-level LaTeX lecture notes, a compiled PDF,
reproducible numerical figures, and the exact Python script used to calculate
the figures and tabulated one-electron data.

## Contents

- `Quantum_Bonding_Revised.tex` -- LaTeX source (24-page manuscript).
- `Quantum_Bonding_Revised.pdf` -- compiled manuscript.
- `make_figures.py` -- reproducible Python script (numpy/scipy/matplotlib).
- `numerical_results.json` -- one-electron two-center numerical data.
- `figures/` -- 13 PNG figures included in the manuscript.

## Rebuild

1. `python make_figures.py`
2. `latexmk -pdf -interaction=nonstopmode Quantum_Bonding_Revised.tex`

Tested with Python, NumPy, SciPy, Matplotlib, and pdfLaTeX.

## Scientific scope and limitations

- The double-delta one-electron model is exactly solvable. The displayed
  two-state hopping is a projection onto fixed isolated atomic orbitals.
- The Gaussian example uses Rayleigh--Ritz for the same delta-well Hamiltonian.
- The Hubbard dimer is exactly solvable within its discrete two-site model.
- The H2 / H2+ calculations distinguish the covalent Heitler--London ansatz
  from one-electron LCAO and from controlled second-order superexchange.
- The hydrogenic n=2 and screened-charge examples contain accurately
  integrated ONE-electron two-center matrix elements. Their TWO-electron
  Heitler--London direct/exchange integrals are formulated but not evaluated
  numerically; no fictitious pair binding-energy curves are claimed.
- The C--O / C--H charge models are intentionally simplified and can give
  unphysical polarization because they exclude occupied cores, Pauli
  repulsion and self-consistent screening. This failure is discussed in the
  notes, rather than hidden.
