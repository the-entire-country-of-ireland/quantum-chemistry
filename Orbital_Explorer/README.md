# Orbital Studio — atomic and molecular orbital website

This is a browser-based interface for both the atomic-orbital explorer and the molecular RHF workflow previously implemented as Jupyter notebooks. **You do not need to open or edit any Python code or notebook** to use the app.

## Start the site

1. Install Python 3.11 or 3.12 (recommended) and create a virtual environment (optional but recommended).
2. From this folder, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

   On macOS/Linux you can alternatively run `./start.sh` after installing requirements. On Windows you can run `start.bat`.

3. Open the **Local URL** printed in the terminal (typically http://localhost:8501). Leave the terminal running while using the app. Stop it with Ctrl+C.

**All calculations run locally** via PySCF, RDKit, and NumPy. This is a local website, not a hosted publicly accessible service. There is no cloud backend. The UI and results stay in your active Streamlit session; calculated results are cleared when the app process is stopped. Download `.npz` files to retain results.

## Pages

Use the left-hand navigation to switch between:

- **Atomic Orbital Explorer** (default): isolated neutral elements H–Ar, shell orbitals (1s, 2s, 2p, ...), valence sp/sp²/sp³ hybrids, and canonical alpha/beta orbital records; phase-colored 3D isosurfaces or probability-sampled point clouds with adjustable box size, grid resolution, enclosed probability, and sample count. The plot displays *spatial* wavefunction phase; red/blue do not denote spin. Download the saved atomic Hartree–Fock coefficients.
- **Molecular Orbital Explorer**: the previous VSEPR-initialized full/reduced-basis Hartree–Fock optimization, frozen-core option, spin-orbital decompositions, and molecular visualizations.

Both pages use `saved_atomic_hf/` and reuse cached isolated-atom solutions for matching element and basis. Open the Atomic Explorer first to generate an atom; the Molecular Explorer can then load it automatically. For an open-shell isolated atom the atomic solver uses ROHF; the molecular solver remains closed-shell RHF only.

## What the molecular page does

- Choose water, HF, CH4, NH3, CO2, formaldehyde, methanol, ethane, ethene, acetylene, N2, or enter a custom SMILES molecular graph.
- Pick cc-pVDZ, cc-pVTZ, 6-31g, or STO-3G.
- Select the full Gaussian model, the reduced saved-atom/hybrid model, or both. Independently optimize the geometry for each.
- Optionally freeze occupied atomic core subspaces, while retaining their Coulomb and exchange interactions with the valence shell.
- Explore valence alpha and beta spin orbitals (their spatial parts coincide in RHF).
- Compare IBO-localized bonds and lone pairs against canonical HF, Boys and Pipek–Mezey orbitals.
- Inspect **individual signed AO/hybrid contributions** for each localized bond, alongside the complete phase-colored MO isosurface or probability point cloud.
- Change plot extent, grid resolution, point count, selected constituent functions, and ingredient isosurface cutoff.
- Compare **Hybrid amplitudes** and **Atomic s/p amplitudes** in adjacent mini-tabs below the 3D plot, with signed coefficient bars and tables for each. The projected-basis fraction is shown once, independent of the chart representation.
- Occupied-valence IBO localization by default; Boys and Pipek–Mezey alternatives preserve the occupied RHF density and energy. No L1 sparsity regularization is used.
- Inspect VSEPR-seeded and optimized molecular structures, bond lengths, bond angles, SCF/geometry convergence, and energies. Compare RHF single-point energies at **fixed VSEPR seeds** against relaxed geometries for each electronic model.
- Download both model results as NumPy `.npz` files, including explicit alpha and beta valence spin-orbital coefficient records.

## Scope and limitations

- The molecular page is a small-molecule research prototype, **not** a general chemistry server. Molecules should be closed-shell, even-electron, and contain only H–Ar elements. No general CI, UHF, ROHF, solvation, excited-state calculations, or automated conformer enumeration. The optional *single selected-pair* CASCI(2,2) is the only correlated electronic model.
- Molecular charge is a setting; for custom charged SMILES ensure chemical connectivity and charge settings agree.
- Initial geometry uses RDKit embedding with approximate bond-size and VSEPR angle priors. Optimized geometries can be local minima.
- Frozen-core and reduced-basis geometry optimizations use numerical gradients and can require many SCF calculations. Start with HF or H2O, cc-pVDZ, small iteration limits. The interface remains blocked while a calculation executes; this is not asynchronous computation.
- Results from the two model spaces are optimized *separately*, so their energy difference includes geometry effects.
- A saved atomic orbital basis file is generated under `saved_atomic_hf/` for each atomic number and Gaussian basis, and reused later.
- Full-basis orbitals are projected into the saved atomic functions for decomposition. The omitted projection norm is reported; if large, the displayed ingredients are not an exact reconstruction. All coefficient weights are representation dependent. Individual surfaces do not independently show interference; the coherent sum and overlap heatmap reveal it.
- The overlap matrix decomposes orbital **norm**, not bond dissociation energy. Localized diagonal Fock expectations and inter-orbital couplings are representation dependent; they are not single-bond energies.

## Package contents

```
app.py
pages/
    atomic.py
    molecular.py
requirements.txt
start.sh
start.bat
.streamlit/config.toml
qchem_analysis/
    __init__.py
    atomic_orbitals.py
    hf_calculations.py
    molecular_geometry.py
    molecular_hf.py
    molecular_viewer.py
```

The package includes the last frozen-core molecular implementation and revised atomic orbital visualization module. It does not modify your original repository.

## Bond angles and coefficient views

The **Geometry & energy** tab lists every bonded-neighbor angle A–B–C in
both the VSEPR seed and optimized structures. Diatomics have no bond angles.

Two coefficient tabs appear beneath the **individual constituent surfaces**:
**Hybrid amplitudes** and **Atomic s/p amplitudes**, for the same selected
canonical/localized orbital. The viewer no longer offers the earlier atom-summed
isosurfaces, because they obscured the overlap of individual σ and π terms.

## Geometry-aware hybrid decompositions

The hybrid coefficient chart no longer assumes `sp3` whenever an atom has four VSEPR electron domains. Instead, its **complete local basis** is chosen from the molecule's actual bond geometry:

- One sigma bond (e.g. fluorine in HF): bond-aligned `sp` pair with two perpendicular `p` functions. A terminal double bond instead uses `sp2`.
- Two bonds: approximately linear → `sp`; near trigonal angles → `sp2`; strongly bent → `sp3`-style.
- Three bonds: approximately planar → `sp2`; pyramidal → `sp3`.
- Four bonds: `sp3`. Isolated atoms remain in the unhybridized `s/p` representation.

This is a visualization/coordinate convention, not an inference of unique physical hybridization. The four `ns/np` functions are always present when available, even when `sp` or `sp2` is selected: unused perpendicular `p` orbitals are retained. Therefore this change is an orthogonal change of local basis and does not change the Hartree–Fock variational space or energy. The molecular page displays the assigned basis by atom above the coefficient tabs. Restart Streamlit and rerun an optimization to regenerate results with the new labels.

## Bonding-oriented interpretation (new)

- On **Bonding orbitals**, choose **Intrinsic bond orbitals (IBO)** (default), then select an individual σ bond, π bond, or lone pair. Names are matched automatically to candidate bond directions; scores are shown, with *mixed* used when ambiguous. This is a heuristic label, not a guarantee that every localized orbital has a two-center character.
- The left 3D panel displays individually selectable **signed atomic s/p** or **bond-aligned hybrid** functions. Unlike the older atom-summed surfaces, C1:2py and C2:2py appear as two distinct phase-colored overlapping contributions. The right panel shows the same localized RHF MO as a complete isosurface or density-sampled point cloud.
- Initial ingredient selection prioritizes one large contribution from each atom (for a clear two-center bonding view). The full ingredient sum uses all retained local functions regardless of which have been hidden.
- The **Constructive/destructive overlap** tab shows the atom×atom and local-function×local-function matrices `a_i S_ij a_j`; for a unique pair the corresponding norm cross term is `2 a_i S_ij a_j`. Both signed coefficient bases are selectable. Norm reconstructed below one means missing full-basis components. The overlap is not itself the energetic stabilization due to a bond.
- **Fock energies & couplings** reports `<φ_i|F|φ_j>` in eV. Diagonal elements are meaningful **expectation values for the chosen orbital representation**, but they are not electron removal or bond breaking energies.
- **Geometry & energy** lists each model at the initial VSEPR geometry and, if requested, its own relaxed optimized geometry. Geometry and orbital coordinate frames align the principal bond along x and planar secondary directions along y; a linear acetylene molecule therefore lies along x, with π components in y and z.
- Canonical, IBO, Boys and Pipek–Mezey **occupy exactly the same valence occupied subspace**. Occupied rotations preserve the RHF density and total energy to floating-point precision. Core orbitals stay outside the valence localization.

### How to inspect acetylene

Choose **Acetylene (C#C)**, then **Intrinsic bond orbitals**. Select the localized C1–C2 π(y) or π(z) assignment (when confidently matched). In the *Atomic s/p* constituent view select `C1:2py` and `C2:2py` (or pz). Rotate the left isosurfaces to see like-signed lobes overlap side-by-side; inspect the right combined MO for the bonding shape and the signed interatomic overlap term below. Then inspect a C–H σ orbital: it should have a bond-oriented carbon `sp` hybrid and a hydrogen 1s contribution. Labels reflect computed matches, not forced states.

## New: Valence-bond learning workflow

The Molecular Bonding Explorer now provides four complementary tools:

1. **Clickable bond map / bond chooser**: click diamond-shaped bond-center markers in the Plotly component (or use the always-visible bond dropdown) to identify the corresponding IBO-localized sigma and pi orbitals. If an IBO assignment is ambiguous, the dropdown still allows a manual orbital choice. The clickable map uses `streamlit-plotly-events`.
2. **Calculated hybrid character**: directly underneath each selected MO, a per-atom table gives projected s/p fractions, effective `sp^n` exponent, and p-orbital axis. These are derived from the *actual localized orbital* rather than hard-coded ideal VSEPR labels. They depend on the saved atomic reference basis and lose reliability when the atomic projection fraction is poor.
3. **Bond Formation Lab**: choose a bond and a localized orbital, then change the two-center mixing angle and the relative sign to construct normalized trial bonding and antibonding combinations. Compare the resulting 3D orbital with the localized RHF result and its one-electron Fock expectation. These controls change a trial function only; they do not recompute the self-consistent HF total energy.
4. **Selected spin-coupled pair**: for an all-electron **full Gaussian** RHF calculation, the lab can construct one occupied bond orbital and a virtual antibonding partner, then run a **singlet CASCI(2 electrons, 2 orbitals)** calculation. Other occupied orbitals remain doubly occupied but inactive. This produces a total correlated molecular energy, the lowering relative to RHF, and the two active natural occupations. This is a controlled one-pair demonstration, **not** a fully simultaneous GVB perfect-pairing method and **not** an unrestricted bond dissociation calculation. The CAS calculation can fail when the selected occupied bond cannot be paired with a sufficiently independent virtual function; the app reports this explicitly.

The **Fock energies & couplings** tab also shows a bond-by-bond table of localized Fock expectation values. These are mathematically well-defined for the chosen localization, but **not bond dissociation energies**. In particular, they cannot be summed to recover the molecule's total energy. The selected-pair energy lowering is a *difference between total molecular energies*, not the energy required to break a bond. Physical bond dissociation energies require separately specified product states and independently relaxed fragment calculations.

### Recommended first experiment

Run **H2** via custom SMILES `[H][H]`, full Gaussian cc-pVDZ, no frozen core. Select the H-H bond, vary mixing phase/angle in the Bond Formation Lab, and run its CASCI(2,2) singlet. Look for occupations of the two active natural orbitals deviating from exactly 2.0 and 0.0. As the bond is stretched (future geometry scanning enhancement), that deviation would become more pronounced.

For polyatomic bonding, start with acetylene (`C#C`), choose a C-C pi IBO, and inspect its per-carbon `p_y`/`p_z` contributions, calculated p-character, and Fock expectation. The atom-pair heatmap distinguishes overlap contributions from orbital energy expectations.

### Validation

The pure NumPy/scipy and RDKit logic has automated tests, including hybrid fraction extraction, two-center mixing, and assembly of an orthonormal active-space frame. **PySCF/Streamlit execution was not available in the packaging environment**, so actual CASCI, localization, 3D click handling, and page integration require runtime verification after installation.

### Expanded orbital displays

The Bonding orbitals tab no longer has the clickable **Select a chemical bond** map. Choose the localized bonding orbital directly from the dropdown. Both **Signed atomic ingredients** and **Combined molecular orbital** now occupy the full page width in two large, 760-pixel-tall plots. Component selectors, isosurface settings, and related options are tucked into a collapsible **Display settings and atomic component selection** section above the plots. The separate Bond Formation Lab is unchanged.
