# Orbital Studio — MO versus fully orbital-optimized GVB-PP

**Molecular explorer:** Two chemical-bonding interpretations at the same fixed
nuclear geometry. **Atomic Orbital Explorer:** unchanged.

- **MO / IBO:** one doubly occupied *intrinsic bond orbital* (or nonbonding
  localized orbital) for each selected pair. The IBOs are occupied-space
  rotations of one RHF calculation, not separately optimized chemical bonds.
- **GVB-PP SCF:** two distinct spatial orbitals for each selected spin-singlet
  electron pair. Pair orbitals and every other molecular spatial orbital are
  self-consistently relaxed against the physical electron Hamiltonian in the
  entire finite Gaussian orbital basis. Bonds and lone pairs can be selected
  together and are optimized *simultaneously*.

Both modes offer a single **Isosurface / Probability points** display toggle
for the selected orbital(s) (including both GVB pair orbitals and VSEPR
reference orbitals). The point cloud samples the spatial density `|psi|^2`
and uses color only for orbital phase, not electron spin. Toggling the display
does **not** recompute the molecule or reoptimize GVB. Both modes also show
signed decomposition into atomic s/p
functions and VSEPR-aligned hybrids, and a geometric VSEPR reference comparison.
The UI has no basis selector, localization selector, geometry optimization,
Bond Formation Lab, Valence Bond Explorer, or default CI/resonance controls.

## Run

Use a Python version supported by your installed PySCF distribution.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For molecular mode choose a built-in geometry or enter explicit XYZ coordinates,
load the molecule, select bonds and/or lone pairs, switch to GVB-PP and click
**Optimize selected singlet pairs**. No GVB optimization occurs in MO mode or
when merely switching the inspected orbital. Computations are cached by
molecule and selected pair indices.

The fixed electronic RHF calculation uses **6-31G**. IBO localization and
signed atomic/hybrid analysis use IAOs built from a valence-complete **STO-3G
minimal atomic reference**, retaining Be 2p orbitals. Minimal IAO projection
coverage is reported, since optimized GVB orbital density may extend beyond
that analysis space.

## What has changed about GVB-SCF

The former active-space GVB-PP implementation froze the occupied spectator
orbitals and all virtual orbitals outside each chosen pair. This release
instead uses `qchem_analysis/gvb_scf.py`:

1. Start from the occupied IBOs and atom-centered trial partners to define the
   initial *assignment* of correlated geminals.
2. Complete those orbitals with **all remaining molecular virtual orbitals**.
3. Evaluate the **full electronic molecular Hamiltonian** (original one-body
   integrals, two-body Coulomb integrals, nuclear repulsion), with **no frozen
   core or frozen spectator orbitals**.
4. Optimize all nonredundant orbital rotations: paired↔paired (including
   interpair mixing), paired↔inactive, paired↔external virtual, and
   inactive↔external virtual, plus the occupation-mixing angle within each
   selected geminal.
5. Return the optimized orbital coefficients (AO basis), pair overlap, natural
   occupations, and total variational energy. The optimized pair's atomic
   population and similarity to the original IBO are checked before making a
   chemical interpretation.

The old active-space solver, `perfect_pairing.py`, remains as an **initial
starting guess and a mathematical regression reference only**. Its energy is
*not* the reported GVB-SCF result.

### Essential scope restrictions

This is a **fully orbital-optimized, spin-adapted GVB perfect-pairing SCF
ansatz**; it is **not** a general multiconfigurational VBSCF calculation.
All orbitals can relax, but the selected pairing topology remains fixed and
pair subspaces remain **strongly orthogonal**. Unselected lone pairs and bonds
stay doubly occupied, though their orbitals relax. No alternative valence-bond
resonance structures, interpair spin recoupling, or unrestricted spin
symmetry breaking are included. Full-basis means full *finite Gaussian AO
basis*, not the complete continuum one-electron space.

A nonlinear orbital optimization may converge to a **local** stationary point.
Results are variational but not globally certified. Check the convergence
indicator and the pair's chemical-label continuity diagnostic. In highly polar
bonds both GVB spatial orbitals may be centered on the electronegative atom;
they are **not** assigned one per atom.

The present dense-integral implementation limits full-space GVB-SCF to **44
spatial orbitals** to control memory and runtime. This limit is an
implementation constraint, not a chemical approximation. For larger systems a
density-fitted or direct-SCF integral/gradient backend would be required.

## Mathematical formulation

For `m` selected pairs, the wavefunction is an antisymmetrized product of
spin-singlet geminals and doubly occupied spectator orbitals:

\[
\Psi=\mathcal A\left[\prod_{c\in\mathrm{inactive}}\!|c^2\rangle
\;\prod_{i=1}^{m}\left(\cos\theta_i|p_i^2\rangle
-\sin\theta_i|q_i^2\rangle\right)\right].
\]

Every spatial orbital, including the `p_i`, `q_i`, spectators and external
virtuals, may rotate in the complete molecular basis, while remaining
orthonormal. The two nonorthogonal **pair orbitals** displayed in the UI are
reconstructed as `u_i = cos(beta_i) p_i + sin(beta_i) q_i`,
`v_i = cos(beta_i) p_i - sin(beta_i) q_i`, with
`tan(beta_i)^2 = tan(theta_i)`. Their symmetrized product equals the chosen
natural-geminal combination (up to normalization).

The complete energy formula and analytic orbital gradient are documented in
`docs/bond_interpretations.md`.

## Tests

```bash
PYTHONPATH=. pytest -q tests
```

For **real molecular PySCF integration tests**:

```bash
ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 PYTHONPATH=. pytest -v tests
```

The integration suite includes H₂ STO-3G, H₂ 6-31G, HF bond polarization, BeH₂ two-pair GVB-SCF,
water lone-pair selection, and methane multiple singlet pairs. The standalone
suite validates all-space orbital/angle derivatives by finite differences,
Hubbard-dimer exact limits, the inactive/external mixing channel, and
consistency with the previous perfect-pairing energy expression.

**Execution limitation:** The packaging environment contains NumPy/SciPy but
not PySCF or Streamlit. Therefore the pure-Python tests and syntax can be
verified here, but actual molecular GVB orbital shapes, convergence times,
and Streamlit interactions must be checked on a PySCF-equipped machine. The
package does not claim that unexecuted integration tests passed.

## Source organization

- `pages/atomic.py` — original unchanged atomic explorer.
- `pages/molecular.py` — MO/IBO and all-space GVB-SCF bond/lone-pair explorer.
- `qchem_analysis/molecular_fixed.py` — fixed geometries, RHF, IAOs and IBOs.
- `qchem_analysis/perfect_pairing.py` — local pair-space initial seeds and old
  fixed-active-space regression formulas.
- `qchem_analysis/gvb_scf.py` — new all-orbital SCF objective, analytic gradient,
  variational minimizer, AO orbital reconstruction.
- `tests/test_gvb_scf.py` — PySCF-independent SCF math and gradient tests.
