"""Occupied-space bond interpretation and overlap/Fock diagnostics.

All reported bond orbitals are orthonormal rotations of a converged RHF
occupied valence space. Orbitals are *not* optimized independently, and
orbital-level Fock expectations are not bond dissociation energies.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

BLUE, RED = '#2878b5', '#d34a4a'
HARTREE_TO_EV = 27.211386245988


def pair_overlap_matrix(coefficients, gram):
    """M_ij = a_i a_j S_ij, whose sum is the norm of the AO projection.

    Both symmetric off-diagonal cells appear in the heatmap. A *pairwise*
    contribution in the table is 2*M_ij, counted only once per unordered pair.
    """
    a = np.asarray(coefficients, dtype=float)
    S = np.asarray(gram, dtype=float)
    if a.ndim != 1 or S.shape != (a.size, a.size):
        raise ValueError('Coefficient vector/Gram matrix shape mismatch')
    return (a[:, None] * S) * a[None, :]


def overlap_analysis(data, top_pairs=40):
    """Report normalization-interference terms in one explicit local basis."""
    labels = list(data['labels'])
    S = np.asarray(data['gram'])
    a = np.asarray(data['coefficients'])
    M = pair_overlap_matrix(a, S)
    terms = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            terms.append({
                'Component A': labels[i], 'Component B': labels[j],
                'Contribution to norm': float(2 * M[i, j]),
                'Bare AO overlap': float(S[i, j]),
                'Coefficient A': float(a[i]), 'Coefficient B': float(a[j]),
                'Interatomic': labels[i].split(':', 1)[0] != labels[j].split(':', 1)[0],
            })
    terms.sort(key=lambda entry: abs(entry['Contribution to norm']), reverse=True)
    return {
        'matrix': M,
        'top_pairs': terms[:top_pairs],
        'top_interatomic': [entry for entry in terms if entry['Interatomic']][:top_pairs],
        'total': float(M.sum()),
        'self_terms': float(M.trace()),
        'cross_terms': float(M.sum() - M.trace()),
    }


def atom_pair_overlap(data):
    """Aggregate M_ij by nucleus; interatomic entries remain symmetric."""
    M = pair_overlap_matrix(data['coefficients'], data['gram'])
    atoms = list(dict.fromkeys(label.split(':', 1)[0] for label in data['labels']))
    indices = [[i for i, name in enumerate(data['labels']) if name.split(':', 1)[0] == atom]
               for atom in atoms]
    grouped = np.array([[M[np.ix_(a, b)].sum() for b in indices] for a in indices])
    if not np.allclose(grouped.sum(), M.sum(), atol=1e-10):
        raise RuntimeError('Atom decomposition does not preserve projection norm')
    return atoms, grouped


def plot_overlap_heatmap(data, grouped=False, max_orbitals=22):
    if grouped:
        labels, matrix = atom_pair_overlap(data)
        title = 'Atom-pair contribution to orbital norm'
    else:
        a = np.asarray(data['coefficients'])
        S = np.asarray(data['gram'])
        # Limit only the visualization, not the totals or the tabular report.
        importance = abs(a) * np.sqrt(np.maximum(np.diag(S), 0))
        selected = np.sort(np.argsort(importance)[::-1][:max_orbitals])
        labels = [data['labels'][i] for i in selected]
        matrix = pair_overlap_matrix(a, S)[np.ix_(selected, selected)]
        title = f'Local-orbital interference ({len(selected)} terms shown)'
    scale = max(float(np.max(abs(matrix))), 1e-8)
    fig = go.Figure(go.Heatmap(
        x=labels, y=labels, z=matrix, zmin=-scale, zmax=scale,
        colorscale='RdBu', reversescale=True, zmid=0,
        colorbar={'title': 'aᵢ Sᵢⱼ aⱼ'},
        hovertemplate='%{y} × %{x}<br>Norm contribution = %{z:+.5f}<extra></extra>',
    ))
    fig.update_layout(title=title, height=max(380, min(710, 20 * len(labels) + 180)),
                      xaxis={'tickangle': -45, 'side': 'top'},
                      margin={'l': 20, 'r': 12, 't': 110, 'b': 25})
    return fig


def effective_fock_matrix(result, valence_ao):
    """Fock expectation/coupling of a selected occupied valence basis.

    Evaluates the *full AO* mean field from the density matrix, also for the
    reduced model whose PySCF get_hcore/get_ovlp are expressed in a custom
    coordinate system. This avoids mixing reduced-basis and AO matrices.
    """
    from pyscf import scf
    mol = result.mol
    Cocc = np.asarray(result.coefficients_ao[:, :result.nocc])
    S = mol.intor('int1e_ovlp')
    D = 2.0 * Cocc @ Cocc.T
    h = scf.hf.get_hcore(mol)
    fock_ao = h + scf.hf.get_veff(mol, D)
    C = np.asarray(valence_ao)
    if not np.allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=2e-5):
        raise ValueError('Localized orbitals are not orthonormal')
    F = C.T @ fock_ao @ C
    return (F + F.T) / 2


def plot_fock_matrix(matrix, names):
    F = np.asarray(matrix)
    energy = F * HARTREE_TO_EV
    fig = go.Figure(go.Heatmap(
        z=energy, x=names, y=names, colorscale='RdBu', reversescale=True,
        zmid=0, colorbar={'title': 'eV'},
        hovertemplate='%{y} / %{x}<br>Fock = %{z:+.4f} eV<extra></extra>',
    ))
    fig.update_layout(title='Occupied-valence Fock matrix (expectations and couplings)',
                      height=max(440, 44 * len(names) + 170),
                      xaxis={'side': 'top', 'tickangle': -50},
                      margin={'l': 15, 'r': 15, 't': 115, 'b': 25})
    return fig


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError('Zero-length bond')
    return vector / norm


def _perpendicular_axes(bond_axis, order, linear_double=False):
    """Transverse axes in the aligned molecular coordinate frame."""
    axis = _unit(bond_axis)
    canonical = np.eye(3)
    transverses = []
    for vector in (canonical[2], canonical[1], canonical[0]):
        t = vector - (vector @ axis) * axis
        for prior in transverses:
            t -= (t @ prior) * prior
        if np.linalg.norm(t) > 1e-7:
            transverses.append(_unit(t))
        if len(transverses) == 2:
            break
    return transverses[:2] if order >= 3 or linear_double else transverses[:1] if order == 2 else []


def bond_trials(result, seed):
    """Candidate bond and lone-pair AO functions for *labelling* localized MOs.

    Trials do not alter the RHF state. Labels are assigned by measured squared
    S-inner products with the final occupied orbitals, not by imposing a
    bonding-orbital representation or constraining variational coefficients.
    """
    from .molecular_hf import _geometry_hybrid_count
    mol = seed.rdkit_mol
    xyz = np.asarray(result.xyz_angstrom)
    S = result.mol.intor('int1e_ovlp')
    atom_cols = dict(zip(result.local_basis.atomic_labels,
                         result.local_basis.B_atomic.T))

    def valence(i):
        atom = mol.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        n = 1 if z <= 2 else 2 if z <= 10 else 3
        return f'{atom.GetSymbol()}{i+1}:{n}'

    def s(i):
        return atom_cols.get(valence(i) + 's')

    def p(i, vector):
        cols = [atom_cols.get(valence(i) + 'p' + axis) for axis in 'xyz']
        if any(x is None for x in cols):
            return None
        return np.column_stack(cols) @ _unit(vector)

    def hybrid(i, direction):
        sv = s(i)
        if sv is None:
            return None
        pv = p(i, direction)
        if pv is None:
            return sv
        m = _geometry_hybrid_count(mol.GetAtomWithIdx(i), xyz)
        m = m if m in (2, 3, 4) else 2
        return np.sqrt(1 / m) * sv + np.sqrt((m - 1) / m) * pv

    def add(left, right):
        if left is None:
            return right
        if right is None:
            return left
        return left + (1 if float(left @ S @ right) >= 0 else -1) * right

    trials = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        u = _unit(xyz[j] - xyz[i])
        root = f'{mol.GetAtomWithIdx(i).GetSymbol()}{i+1}–{mol.GetAtomWithIdx(j).GetSymbol()}{j+1}'
        trials.append((root + ' σ', add(hybrid(i, u), hybrid(j, -u))))
        order = int(round(bond.GetBondTypeAsDouble()))
        # A linear AX2 double-bonded center supports two orthogonal pi axes.
        is_linear = False
        for idx in (i, j):
            atom = mol.GetAtomWithIdx(idx)
            nbs = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            if len(nbs) == 2:
                v, w = (_unit(xyz[k] - xyz[idx]) for k in nbs)
                is_linear |= float(v @ w) < -0.92
        for perpendicular in _perpendicular_axes(u, order, linear_double=is_linear):
            left, right = p(i, perpendicular), p(j, perpendicular)
            if left is None or right is None:
                continue
            axis_name = 'xyz'[int(np.argmax(abs(perpendicular)))]
            trials.append((root + f' π({axis_name})', add(left, right)))
    for i, atom in enumerate(mol.GetAtoms()):
        for key in ('s', 'px', 'py', 'pz'):
            col = atom_cols.get(valence(i) + key)
            if col is not None:
                trials.append((f'{atom.GetSymbol()}{i+1} lone-pair / {key} candidate', col))
    return [(name, vec) for name, vec in trials if vec is not None]


def assign_bond_labels(result, seed, rotated_valence, min_score=0.26):
    """One-to-one trial assignment; report reliability rather than overclaiming."""
    C = np.asarray(rotated_valence)
    S = result.mol.intor('int1e_ovlp')
    trials = bond_trials(result, seed)
    n = C.shape[1]
    if n == 0:
        return []
    valid = [(name, v, float(v @ S @ v)) for name, v in trials]
    valid = [(name, v, norm) for name, v, norm in valid if norm > 1e-9]
    if not valid:
        return [(f'Localized orbital {j+1}', 0.0) for j in range(n)]
    scores = np.array([[float((C[:, i] @ S @ v)**2 / norm) for name, v, norm in valid]
                       for i in range(n)])
    # Slight bond preference only breaks close score ties; the displayed
    # confidence is the unmodified squared overlap.
    priority = np.array([0.015 if ' σ' in name or ' π(' in name else 0 for name, _, _ in valid])
    rows, cols = linear_sum_assignment(-(scores + priority))
    output = [(f'Localized orbital {i+1} (mixed)', 0.0) for i in range(n)]
    for i, k in zip(rows, cols):
        score = float(scores[i, k])
        if score >= min_score:
            output[i] = (valid[k][0], score)
    return output


def validate_density_preservation(result, rotated_valence):
    """Return norm of the occupied-density difference from localization."""
    ids = result.valence_spatial_indices
    original = np.asarray(result.coefficients_ao[:, ids])
    rotated = np.asarray(rotated_valence)
    S = result.mol.intor('int1e_ovlp')
    if rotated.shape != original.shape:
        raise ValueError('Localization changed number of occupied orbitals')
    if not np.allclose(rotated.T @ S @ rotated, np.eye(len(ids)), atol=1e-5):
        raise ValueError('Localization broke orthonormality')
    return float(np.linalg.norm(2 * original @ original.T - 2 * rotated @ rotated.T))




def localized_spin_archive(result, localized_valence, orbital_names,
                           localization, *, fock_matrix=None):
    """Produce portable, explicit alpha/beta records of *localized* RHF valence MOs.

    The two spin records for each doubly occupied spatial orbital share one
    spatial wavefunction. Optional Fock matrix entries are expectations in
    that rotated occupied basis, NOT bond dissociation energies.
    """
    from io import BytesIO
    C = np.asarray(localized_valence, dtype=float)
    if C.ndim != 2:
        raise ValueError('Expected a 2D array of spatial orbital coefficients')
    n = C.shape[1]
    if len(orbital_names) != n:
        raise ValueError('Mismatch between localized orbitals and names')
    labels = [entry[0] for entry in orbital_names]
    scores = [entry[1] for entry in orbital_names]
    spins = np.tile(np.asarray(['alpha', 'beta']), n)
    spatial_ids = np.repeat(np.arange(n), 2)
    packed = {
        'mode': str(result.mode),
        'localization': str(localization),
        'xyz_angstrom': np.asarray(result.xyz_angstrom),
        'total_hf_energy_hartree': float(result.energy_hartree),
        'valence_spatial_coefficients_ao': C,
        'valence_spatial_labels': np.asarray(labels),
        'orbital_match_scores': np.asarray(scores, dtype=float),
        'valence_spin_coefficients_ao': np.repeat(C,2,axis=1),
        'valence_spin_labels': np.repeat(labels,2),
        'valence_spin': spins,
        'valence_spin_spatial_indices': spatial_ids,
        'frozen_core': bool(result.frozen_core),
    }
    if fock_matrix is not None:
        F = np.asarray(fock_matrix,dtype=float)
        if F.shape != (n,n):
            raise ValueError('Fock matrix does not match valence orbital count')
        packed['occupied_fock_matrix_hartree'] = F
        packed['localized_spatial_fock_expectation_hartree'] = F.diagonal()
        packed['localized_spin_fock_expectation_hartree'] = np.repeat(F.diagonal(),2)
    buffer = BytesIO()
    np.savez_compressed(buffer,**packed)
    return buffer.getvalue()
