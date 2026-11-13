# Molecular bonds: localized MO (IBO) and fully relaxed GVB perfect pairing

## Two electron-pair pictures

The localized RHF molecular-orbital picture uses one occupied spatial orbital
`g_i` per spin-singlet electron pair:

\[
\Psi^{\rm MO}_i(1,2)=g_i(1)g_i(2)\chi_S(1,2).
\]

A GVB perfect-pairing function uses separate, optimized spatial orbitals
`u_i`, `v_i` but the **same spin-singlet state**:

\[
\Psi^{\rm GVB}_i(1,2)=
\frac{u_i(1)v_i(2)+v_i(1)u_i(2)}
 {\sqrt{2(1+|\langle u_i|v_i\rangle|^2)}}\chi_S(1,2).
\]

The two orbitals within a pair may overlap and can both be concentrated on a
single atom for a strongly ionic bond. Neither orbital assigns an electron a
specific spin or nuclear center. Multiple geminals are antisymmetrized
*together*, not optimized as independent two-electron molecules.

## Entire orbital space versus active-space restriction

The molecular RHF calculation starts from the **physical nonrelativistic
molecular Hamiltonian**, including all electron–electron Coulomb terms and
nuclear repulsion at the fixed nuclear geometry. The molecular AO basis is
6-31G, fixed internally, and the occupied space is localized with IBOs.

For selected `m` singlet pairs, build an initial orthonormal spatial frame
ordered as `inactive doubly occupied | (p1,q1),...,(pm,qm) | remaining virtual`.
The inactive orbitals include the core and any unselected occupied bond/lone
pairs. We then optimize a full orthogonal rotation in **all** spatial orbitals,
not merely within each selected pair's initial active subspace. The only
redundant rotations removed are inactive↔inactive and unused virtual↔unused
virtual rotations.

At every iteration the many-electron wavefunction remains the *strongly
orthogonal* perfect-pairing state

\[
|\Psi\rangle=\mathcal A\left[
\prod_{c\in\mathcal C}|c^2\rangle
\prod_{k=1}^m\left(\cos\theta_k|p_k^2\rangle
-\sin\theta_k|q_k^2\rangle\right)\right],
\quad0\le\theta_k\le\frac\pi4.
\]

In the rotated orthonormal natural-orbital frame the pair occupations are
`n(p_k)=2 cos²(theta_k)` and `n(q_k)=2 sin²(theta_k)`. Inactive natural-orbital
occupations remain 2. The spin symmetry stays singlet; this method does **not**
mean UHF or independent alpha/beta orbital optimization.

## Full energy expectation

Let `w_i` be the *pair probability weight*: 1 for each inactive doubly
occupied orbital, `cos²(theta_k)` or `sin²(theta_k)` for the two natural
orbitals of pair `k`, and 0 for external unoccupied orbitals. For chemist's
integrals `J_ij=(ii|jj)` and `K_ij=(ij|ij)`, the total energy is

\[
\begin{aligned}
E={}&E_{NN}+\sum_i w_i\left[2h_{ii}+(ii|ii)\right]\\
&+\sum_{i<j\atop\mathrm{different\ electron\ pairs}}
  w_iw_j\left[4(ii|jj)-2(ij|ij)\right]\\
&-\sum_{k=1}^m\sin(2\theta_k)(p_kq_k|p_kq_k).
\end{aligned}
\]

This includes Coulomb and exchange between *all* electron pairs, core and
valence alike, and the off-diagonal pair-excitation matrix elements. There is
no frozen-core effective potential in the objective. The nuclear energy is a
fixed additive scalar at each geometry.

The orbital rotations are parametrized with the Cayley map
`Q=(I-K/2)^(-1)(I+K/2)`, `K^T=-K`, and optimized with L-BFGS-B. Orbital
and pair-angle gradients are **analytic**; they were independently checked
against numerical derivatives. Multiple deterministic starting points include
RHF and an optimized selected-pair active-space seed. This is a **local** SCF
optimizer, not proof of the global optimum.

## Interpretation and limitations

The optimized pair's two displayed orbitals are reconstructed from its two
natural orbitals. Both are individually normalized and generally
nonorthogonal. For each displayed orbital the IAO projection reports signed
atomic/hybrid coefficients, captured norm, and site populations. For the *pair
itself*, the chemically meaningful atomic electron population uses the
spin-summed one-electron reduced density matrix

\[
\gamma_k=2\cos^2\theta_k\,|p_k\rangle\langle p_k|
+2\sin^2\theta_k\,|q_k\rangle\langle q_k|.
\]

That avoids arbitrarily assigning one of the two indistinguishable electrons
or a pair orbital to each atom. An additional retention diagnostic warns if
optimized pair density has moved far from its initially selected IBO. The
VSEPR hybrid is a *reference*, not a unique observable. Lone-pair hybrids can
be rotated in chemically equivalent subspaces without changing the state.

This is **GVB-PP with full molecular-orbital SCF relaxation**. It is not a
unrestricted spin-polarized Hartree–Fock calculation, a resonance-structure
CI expansion, full VBSCF with all spin recouplings, or continuum-exact quantum
chemistry. Pair assignments are held fixed; spectator pairs remain doubly
occupied (while their orbitals relax). The finite basis imposes its usual
limitations. The interactive dense tensor backend supports up to 44 spatial
orbitals.

## Verification and user workflow

MO mode visualizes IBOs immediately after the RHF reference. GVB-SCF runs only
on explicit request and caches by molecule and selected pairs. The initial
atom-local virtual partner for lone pairs is just a starting guess, not an
ongoing constraint on the final orbitals. Select multiple bonds and lone pairs
to optimize them together. The molecule's nuclear positions never change.

Run `PYTHONPATH=. pytest -q tests` for mathematical checks. With PySCF installed,
run `ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 PYTHONPATH=. pytest -v tests` for
real Hamiltonian calculations including HF polarity and H₂ comparisons with
FCI. The packaging environment lacks PySCF, so actual chemical integration
results are not represented as executed.
