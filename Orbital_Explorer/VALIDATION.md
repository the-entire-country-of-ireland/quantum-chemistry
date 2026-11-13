# Validation (all-orbital GVB-SCF)

Mathematical tests check: Hubbard-dimer exact singlet limit, variational RHF
reference, 2e pair normalization, analytic gradients with finite differences,
Cayley-rotation derivatives, all-space inactive→virtual relaxation, pair↔virtual
and interpair rotation availability, and total-energy agreement with the
independent earlier perfect-pairing frozen-core formula.

The opt-in PySCF integration tests include H₂ (STO-3G and 6-31G), HF (polar
bond atomic populations) and methane multi-pair optimization, in addition to
legacy orbital-selection and water lone-pair checks.

**Important:** PySCF/Streamlit are not installed in this build environment.
Actual electronic-structure and browser UI integration tests are not executed
here. Run `ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 PYTHONPATH=. pytest -v tests`
in the target environment. SCF optimizations are local and the strongest test
is comparing full-space energy with the RHF and fixed-active-space baselines.
