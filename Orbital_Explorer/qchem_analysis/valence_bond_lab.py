"""Bond-centered interpretation of RHF and an optional selected-pair CAS(2,2).

The CAS calculation is a *single selected-pair* correlated singlet with all
other occupied orbitals inactive. It is NOT full simultaneous GVB-PP.
Orbital Fock expectations and pair correlation lowering are NOT bond energies.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

HARTREE_TO_EV = 27.211386245988


def norm_s(c, S):
    return float(np.asarray(c) @ S @ np.asarray(c))


def normalize_s(c, S):
    c = np.asarray(c, dtype=float)
    n = norm_s(c, S)
    if n <= 1e-12:
        raise ValueError('Cannot normalize zero-norm or linearly dependent orbital')
    return c / np.sqrt(n)


def atomic_hybrid_character(projection, *, min_atom_fraction=0.025):
    """Derive local s/p character and p-direction for each atom in an MO.

    Fractions are defined within each atom's projected *valence* contribution.
    This is not a unique population analysis; use the reported projection norm.
    """
    labels = projection['labels']
    a = np.asarray(projection['coefficients'], dtype=float)
    G = np.asarray(projection['gram'], dtype=float)
    atoms = list(dict.fromkeys(label.split(':', 1)[0] for label in labels))
    rows = []
    strengths = {}
    for atom in atoms:
        ids = [k for k, lab in enumerate(labels) if lab.split(':', 1)[0] == atom]
        strength = float(a[ids] @ G[np.ix_(ids, ids)] @ a[ids])
        strengths[atom] = max(strength, 0.)
    total = sum(strengths.values())
    for atom in atoms:
        ids = [k for k, lab in enumerate(labels) if lab.split(':', 1)[0] == atom]
        local = strengths[atom]
        if not ids or local < min_atom_fraction * max(total, 1e-12):
            continue
        s_ids = [k for k in ids if labels[k].split(':', 1)[1].endswith('s')]
        p_ids = [k for k in ids if labels[k].split(':', 1)[1].split('p')[-1] in ('x','y','z') and 'p' in labels[k].split(':', 1)[1]]
        # Explicitly use the outermost valence shell; inner 1s does not count.
        shell_numbers = sorted({int(labels[k].split(':',1)[1][0]) for k in ids if labels[k].split(':',1)[1][0].isdigit()})
        if not shell_numbers:
            continue
        n = shell_numbers[-1]
        s_ids = [k for k in s_ids if labels[k].split(':',1)[1] == f'{n}s']
        p_ids = [k for k in p_ids if labels[k].split(':',1)[1] in (f'{n}px',f'{n}py',f'{n}pz')]
        # On an isolated atom, valence s/p angular channels are orthogonal.
        s_norm = float(a[s_ids] @ G[np.ix_(s_ids,s_ids)] @ a[s_ids]) if s_ids else 0.
        p_norm = float(a[p_ids] @ G[np.ix_(p_ids,p_ids)] @ a[p_ids]) if p_ids else 0.
        sp_norm = max(s_norm + p_norm, 0.)
        if sp_norm < 1e-10:
            continue
        f_s = float(np.clip(s_norm/sp_norm, 0, 1))
        p_vec = np.zeros(3)
        for k in p_ids:
            axis = 'xyz'.index(labels[k][-1])
            p_vec[axis] = a[k] * np.sqrt(max(G[k,k],0.))
        direction = p_vec / np.linalg.norm(p_vec) if np.linalg.norm(p_vec) > 1e-10 else np.zeros(3)
        rows.append({
            'Atom':atom,
            's character':f_s,
            'p character':1.-f_s,
            'Effective sp^n':(1.-f_s)/f_s if f_s > 1e-6 else float('inf'),
            'p direction x':float(direction[0]),
            'p direction y':float(direction[1]),
            'p direction z':float(direction[2]),
            'Local norm':local,
            'Projection fraction':float(projection.get('probability_represented',np.nan)),
        })
    return rows


def bond_sites(seed):
    """Selectable bond list keyed by the actual connectivity."""
    output = []
    for bond in seed.rdkit_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        name = f'{seed.symbols[i]}{i+1}–{seed.symbols[j]}{j+1}'
        output.append({'index':len(output), 'a':i, 'b':j,
                       'name':name, 'order':float(bond.GetBondTypeAsDouble())})
    return output


def bond_map_figure(seed, xyz, selection=None):
    """3D molecular stick drawing with selectable marker at each bond center."""
    import plotly.graph_objects as go
    xyz=np.asarray(xyz,dtype=float)
    fig=go.Figure()
    bonds=bond_sites(seed)
    for b in bonds:
        i,j=b['a'],b['b']
        fig.add_trace(go.Scatter3d(x=[xyz[i,0],xyz[j,0]],y=[xyz[i,1],xyz[j,1]],
            z=[xyz[i,2],xyz[j,2]],mode='lines',line={'width':8,'color':'#84929b'},
            showlegend=False,hoverinfo='skip'))
    for k,name in enumerate(seed.symbols):
        r=xyz[k]
        fig.add_trace(go.Scatter3d(x=[r[0]],y=[r[1]],z=[r[2]],
            mode='markers+text',marker={'size':9 if name!='H' else 6,'color':'#4e96ba'},
            text=[f'{name}{k+1}'],textposition='top center',showlegend=False,
            hovertemplate=f'{name}{k+1}<extra></extra>'))
    centers=np.array([(xyz[b['a']]+xyz[b['b']])/2 for b in bonds]) if bonds else np.zeros((0,3))
    fig.add_trace(go.Scatter3d(
        x=centers[:,0],y=centers[:,1],z=centers[:,2],mode='markers',
        marker={'size':11,'symbol':'diamond','color':['#ec8b38' if b['index']==selection else '#5bceae' for b in bonds],
                'line':{'color':'#10212a','width':1}},
        customdata=[b['index'] for b in bonds],
        hovertemplate='Bond center · %{customdata}<extra>Click to select bond if supported</extra>',
        name='Select bond centers',showlegend=True))
    fig.update_layout(height=370,margin={'l':0,'r':0,'b':0,'t':15},
        scene={'xaxis_title':'x (Å)','yaxis_title':'y (Å)','zaxis_title':'z (Å)', 'aspectmode':'data'})
    return fig


def bond_orbital_candidates(names, bond):
    """Local orbital indices associated with selected bond, ordered σ then π."""
    prefix=bond['name']
    matched=[i for i,(name,score) in enumerate(names)
             if name.startswith(prefix+' ') and (' σ' in name or ' π(' in name)]
    return sorted(matched,key=lambda i:(0 if ' σ' in names[i][0] else 1, names[i][0]))


def two_site_bond_functions(result, projected, atom_a, atom_b):
    """Return local contributions on two atoms, allowing coherent σ/π assembly."""
    labels=projected['labels']
    components=np.asarray(projected['components_ao'])
    left=[i for i,s in enumerate(labels) if s.split(':',1)[0]==atom_a]
    right=[i for i,s in enumerate(labels) if s.split(':',1)[0]==atom_b]
    if not left or not right:
        raise ValueError('No saved atomic contributions for this bond')
    return components[:,left].sum(axis=1), components[:,right].sum(axis=1)


def mixed_bond_orbital(left,right, S, angle_degrees=45.,phase=1):
    """Coherent two-site mixing of normalized atom-local functions.

    This is an interactive orbital *model*, not a self-consistent HF solution.
    """
    left=normalize_s(left,S)
    right=normalize_s(right,S)
    theta=np.deg2rad(angle_degrees)
    c=np.cos(theta)*left+int(phase)*np.sin(theta)*right
    return normalize_s(c,S)


def orbital_fock_expectation(mol, occupied_ao, trial):
    from pyscf import scf
    S=mol.intor('int1e_ovlp')
    trial=normalize_s(trial,S)
    D=2*np.asarray(occupied_ao)@np.asarray(occupied_ao).T
    F=scf.hf.get_hcore(mol)+scf.hf.get_veff(mol,D)
    return float(trial@F@trial)


def selected_pair_active_space(result, localized_valence, chosen, projection, atom_a, atom_b):
    """Build complete orthonormal CAS orbital frame: inactive, bond, antibond, rest.

    The occupied bonding orbital is rotated out of the occupied subspace, while
    the antibonding partner is built from the *difference* of atom-local
    contributions projected into the virtual subspace.
    """
    if result.mode != 'full':
        raise ValueError('Pair CAS currently requires the full Gaussian RHF model')
    S=result.mol.intor('int1e_ovlp')
    nocc=result.nocc
    full=np.asarray(result.coefficients_ao,dtype=float)
    loc=np.asarray(localized_valence,dtype=float)
    occupied=np.column_stack([full[:,:result.core_spatial_count],loc])
    if not np.allclose(occupied.T@S@occupied,np.eye(nocc),atol=3e-5):
        raise ValueError('Localized occupied orbitals are not orthonormal')
    target=result.core_spatial_count+int(chosen)
    selected=occupied[:,target]
    inactive=np.delete(occupied,target,axis=1)
    left,right=two_site_bond_functions(result,projection,atom_a,atom_b)
    # Align the sign on each atom with the selected localized MO.
    if left@S@selected<0: left=-left
    if right@S@selected<0: right=-right
    opposite=left-right
    # Project explicitly out of ALL occupied orbitals.
    opposite=opposite-occupied@(occupied.T@S@opposite)
    if norm_s(opposite,S)<1e-7:
        raise ValueError('Atomic antibonding trial lies almost entirely inside occupied space')
    partner=normalize_s(opposite,S)
    # Build an orthonormal complement of the newly selected antibonding vector
    # using the canonical virtual orbitals and an S-metric Gram-Schmidt step.
    columns=[*inactive.T,selected,partner]
    for v in full[:,nocc:].T:
        w=v.copy()
        for b in columns:
            w-=b*float(b@S@w)
        if norm_s(w,S)>1e-9:
            columns.append(normalize_s(w,S))
        if len(columns)==full.shape[1]:break
    C=np.column_stack(columns)
    if C.shape!=full.shape or not np.allclose(C.T@S@C,np.eye(C.shape[1]),atol=2e-5):
        raise RuntimeError('Failed to build a complete CAS orbital frame')
    return C, partner


def selected_pair_casci(result, localized_valence, chosen, projection, atom_a, atom_b):
    """CASCI(2e,2o) singlet for one chosen bond with inactive electrons present.

    This is a variational *total molecular* energy at fixed RHF orbitals,
    NOT a bond dissociation energy, and NOT a full multi-pair GVB calculation.
    """
    from pyscf import mcscf, fci
    if result.mode != 'full' or result.frozen_core:
        raise ValueError('Selected-pair CASCI currently requires all-electron full Gaussian RHF')
    C,partner=selected_pair_active_space(result,localized_valence,chosen,projection,atom_a,atom_b)
    # Use the original converged RHF mean-field object for the full AO basis.
    mc=mcscf.CASCI(result.mf,2,2)
    mc.ncore=result.nocc-1
    mc.fcisolver=fci.solver(result.mol,singlet=True)
    mc.canonicalization=False
    e_total=mc.kernel(C)[0]
    occupation=np.linalg.eigvalsh(mc.fcisolver.make_rdm1(mc.ci,2,(1,1)))[::-1]
    return {
        'total_energy_hartree':float(e_total),
        'rhf_total_energy_hartree':float(result.energy_hartree),
        'pair_correlation_lowering_hartree':float(e_total-result.energy_hartree),
        'pair_correlation_lowering_ev':float((e_total-result.energy_hartree)*HARTREE_TO_EV),
        'active_natural_occupations':occupation,
        'active_orbitals_ao':C[:,result.nocc-1:result.nocc+1],
        'model':'Selected bond CASCI(2,2), fixed orbital shapes',
    }
