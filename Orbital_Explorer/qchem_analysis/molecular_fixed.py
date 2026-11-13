"""Fast fixed-geometry molecular reference and chemically interpretable IAO projections.

The MO and GVB views deliberately share the same molecular Hamiltonian, nuclear
coordinates, AO basis, IBOs, and bond mapping. Geometry is never optimized here.

Only this module (not the atomic explorer) changes the underlying data workflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .molecular_geometry import MolecularSeed
from .molecular_hf import (
    LocalBasis, RHFResult, _hybrid_transform, _match_hybrid_directions,
    _templates, core_count_from_rdkit, molecule_from_xyz,
)
from .bonding_analysis import bond_trials, validate_density_preservation
from .bond_selection import bond_sites

BASIS = '6-31g'  # Fixed molecular electronic basis.
IAO_REFERENCE = 'sto-3g'  # Includes essential Be 2p valence hybrids omitted from MINAO.
from .fixed_geometry_data import fixed_molecule_presets, fixed_seed, parse_xyz, bond_angles


def _shell_label(label) -> str:
    """PySCF ao_labels(fmt=False) tuple: (atom index, symbol, '2px', ...)."""
    parts=label if isinstance(label,(tuple,list)) else str(label).split()
    if len(parts)>=3:
        return (str(parts[2])+ (str(parts[3]) if len(parts)>3 else '')).replace(' ','')
    raise ValueError(f'Unexpected minimal AO label: {label!r}')


def _valence_shell(atom_z:int) -> int:
    return 1 if atom_z<=2 else 2 if atom_z<=10 else 3


def make_iao_local_basis(mol, occupied: np.ndarray, seed: MolecularSeed) -> LocalBasis:
    """Use IAOs with an STO-3G minimal reference, without isolated-atom HF runs.

    Lowdin-orthogonal IAOs furnish local s/p orbital *projections*. Rotating
    each atom's valence s/p block gives idealized VSEPR hybrids. Occupied MOs
    are represented well, while a GVB virtual partner can have a nonzero
    residual outside this small atomic space, which is explicitly reported.
    """
    from pyscf import lo, gto
    overlap=np.asarray(mol.intor('int1e_ovlp'),float)
    iaos=lo.iao.iao(mol,occupied,minao=IAO_REFERENCE)
    iaos=lo.vec_lowdin(iaos,overlap)
    ref=gto.M(atom=[(s,p.tolist()) for s,p in zip(seed.symbols,seed.xyz_angstrom)],
              unit='Angstrom',basis=IAO_REFERENCE,spin=0,charge=int(mol.charge),verbose=0,cart=False)
    if iaos.shape[1]!=ref.nao_nr():
        raise RuntimeError('IAO/reference-minimal dimension mismatch; unable to label local atomic orbitals.')
    atom_slices=ref.aoslice_by_atom()
    raw_labels=ref.ao_labels(fmt=False)
    labels=[f'{seed.symbols[int(row[0])]}{int(row[0])+1}:{_shell_label(row)}' for row in raw_labels]
    col_order=[];hybrid_labels=[];atom_ids=[]
    xyz=np.asarray(seed.xyz_angstrom,float)
    for i,atom in enumerate(seed.rdkit_mol.GetAtoms()):
        start,end=map(int,atom_slices[i,2:4]); indices=list(range(start,end))
        domain=seed.vsepr_domains[i]
        count=domain if domain in (2,3,4) else 0
        principal=_valence_shell(atom.GetAtomicNum())
        names=[labels[k].split(':',1)[1] for k in indices]
        needed=[f'{principal}s']+[f'{principal}p{d}' for d in 'xyz']
        sub=np.eye(len(indices))
        new_names=names.copy()
        if count and all(x in names for x in needed):
            nbrs=[nb.GetIdx() for nb in atom.GetNeighbors()]
            directions=xyz[nbrs]-xyz[i]
            template=_match_hybrid_directions(_templates(count),directions)
            transform,aliases=_hybrid_transform(template)
            ix=[names.index(x) for x in needed]
            sub[np.ix_(ix,ix)]=transform
            for idx,alias in zip(ix,aliases):
                new_names[idx]=alias
        for j,index in enumerate(indices):
            col=iaos[:,indices] @ sub[:,j]
            col_order.append(col)
            hybrid_labels.append(f'{seed.symbols[i]}{i+1}:{new_names[j]}')
            atom_ids.append(i)
    B=np.column_stack(col_order)
    if not np.allclose(B.T@overlap@B,np.eye(B.shape[1]),atol=1e-5):
        raise RuntimeError('Atomic projection basis lost S-orthonormality.')
    return LocalBasis(B,hybrid_labels,atom_ids,iaos,labels)


def localize_bond_ibos(result:RHFResult)->np.ndarray:
    """IBO localization with the same valence-complete STO-3G reference.

    The default MINAO omits Be 2p orbitals, obscuring the sp Be-H bond
    decomposition. Procrustes returns an exact occupied-space RHF rotation.
    """
    from pyscf import lo
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    val=np.asarray(result.coefficients_ao[:,result.valence_spatial_indices],float)
    if val.shape[1]<2:return val.copy()
    occ=np.asarray(result.coefficients_ao[:,:result.nocc],float)
    iaos=lo.iao.iao(result.mol,occ,minao=IAO_REFERENCE)
    iaos=lo.vec_lowdin(iaos,S)
    raw=lo.ibo.ibo(result.mol,val,iaos=iaos)
    u,_,vt=np.linalg.svd(val.T@S@raw,full_matrices=False)
    localized=val@(u@vt)
    if not np.allclose(localized.T@S@localized,np.eye(val.shape[1]),atol=2e-6):
        raise RuntimeError('IBO rotation failed to preserve occupied orthogonality.')
    return localized


def single_point(seed: MolecularSeed, charge:int=0, *, basis:str=BASIS) -> RHFResult:
    """One full-Gaussian RHF run, at fixed supplied nuclear coordinates."""
    from pyscf import scf
    ne=sum(atom.GetAtomicNum() for atom in seed.rdkit_mol.GetAtoms())-int(charge)
    if ne<2 or ne%2:
        raise ValueError('A closed-shell singlet requires a positive even electron count.')
    mol=molecule_from_xyz(seed.symbols,seed.xyz_angstrom,basis,int(charge))
    mf=scf.RHF(mol)
    mf.conv_tol=1e-9;mf.max_cycle=110;mf.verbose=0
    energy=mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF failed to converge at this fixed geometry.')
    # Optimize only orbital coefficients; no nuclear geometry optimization.
    local=make_iao_local_basis(mol,mf.mo_coeff[:,:mol.nelectron//2],seed)
    ncore=core_count_from_rdkit(seed.rdkit_mol)
    return RHFResult('full',float(energy),np.asarray(seed.xyz_angstrom,float),mol,mf,
                     local,np.asarray(mf.mo_coeff,float),np.asarray(mf.mo_energy,float),
                     True,True,'fixed input nuclear geometry',ncore,[float(energy)],
                     frozen_core=False,occupations=mf.mo_occ.copy())


@dataclass(frozen=True)
class BondOrbital:
    orbital_index: int
    bond_index: int
    label: str
    confidence: float
    kind: str


def atomic_iao_populations(result, coefficient: np.ndarray) -> dict[str, float]:
    """Orthogonal-IAO atomic projection weights for one normalized spatial orbital.

    The values are fractions of the FULL AO orbital norm, not renormalized
    percentages of the represented minimal projection. An unrepresented
    remainder is reported explicitly rather than silently assigned to atoms.
    """
    S = np.asarray(result.mol.intor('int1e_ovlp'), float)
    c = np.asarray(coefficient, float)
    B = np.asarray(result.local_basis.B_atomic, float)
    labels = result.local_basis.atomic_labels
    if B.shape[1] != len(labels):
        raise ValueError('IAO labels and basis dimensions differ')
    amps = B.T @ S @ c
    weights: dict[str, float] = {}
    for label,amp in zip(labels,amps):
        atom = label.split(':',1)[0]
        weights[atom] = weights.get(atom,0.0) + float(amp*amp)
    weights['unrepresented'] = max(0.,float(c@S@c)-sum(weights.values()))
    return weights


def bond_participation(left: float, right: float) -> float:
    """Symmetric 0..1 measure of shared atomic support.

    A lone pair has zero participation on the other atom. A strongly polar
    bond is allowed: 95/5 atomic weights still give participation ~0.44.
    This is used *only* to select the correct occupied IBO, not to change
    its shape or impose a covalent/ionic character on the optimized wavefunction.
    """
    total = float(left + right)
    if total <= 1e-14:
        return 0.
    return float(np.sqrt(max(0.,4.*left*right))/total)


def bond_orbital_map(result: RHFResult, seed: MolecularSeed,
                     localized: np.ndarray) -> list[BondOrbital]:
    """Match occupied IBOs to bond trials, penalizing lone-pair impostors.

    Idealized bond-trial overlap ALONE can incorrectly select a lone pair
    in strongly polar molecules (especially HF). Actual IAO populations
    must show participation from both bonded atoms. The confidence returned
    is the raw trial overlap; the participation criterion is internal.
    """
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    trials=bond_trials(result,seed)
    bonds=bond_sites(seed)
    items=[]
    for label,vec in trials:
        for bond in bonds:
            if not label.startswith(bond['name']+' '):continue
            if ' σ' not in label and ' π(' not in label:continue
            norm=float(vec@S@vec)
            if norm>1e-10:
                items.append((bond['index'],label,np.asarray(vec,float),norm,bond.get('a',0),bond.get('b',0)))
            break
    C=np.asarray(localized,float)
    # A two-electron, one-bond diatomic (notably H2) has exactly one
    # occupied valence spatial orbital. There is no competing lone pair to
    # assign. Treat that occupied orbital as the sigma bond even if a bond
    # trial is missing or its IAO projection accidentally vanishes. This
    # special case must not weaken the polar multi-IBO HF safeguard below.
    if len(bonds)==1 and C.ndim==2 and C.shape[1]==1 and len(seed.symbols)==2:
        bond=bonds[0]
        candidates=[(label,vec,norm) for bi,label,vec,norm,_,_ in items
                    if bi==bond['index'] and ' σ' in label]
        confidence=max((float((C[:,0]@S@vec)**2/norm)
                        for _,vec,norm in candidates),default=0.)
        return [BondOrbital(0,bond['index'],bond['name']+' σ',confidence,'σ')]
    if not items:return []
    raw=np.array([[(C[:,i]@S@trial)**2/norm
                   for _,_,trial,norm,_,_ in items]
                  for i in range(C.shape[1])],float)
    # Synthetic tests using a minimal result without IAOs retain the old
    # overlap-only behavior. Real molecular calculations always supply IAOs.
    if hasattr(result,'local_basis') and hasattr(result.local_basis,'B_atomic'):
        atomic=[atomic_iao_populations(result,C[:,i]) for i in range(C.shape[1])]
        score=np.zeros_like(raw)
        for i,pop in enumerate(atomic):
            for j,(*_,a,b) in enumerate(items):
                left=f'{seed.symbols[a]}{a+1}'
                right=f'{seed.symbols[b]}{b+1}'
                score[i,j]=raw[i,j]*bond_participation(pop.get(left,0.),pop.get(right,0.))
    else:
        score=raw.copy()
    rows,cols=linear_sum_assignment(-score)
    matches=[]
    for idx,j in zip(rows,cols):
        bond,label,*_=items[j]
        # Never present an unshared pure lone pair as a confirmed bond orbital.
        if score[idx,j]<1e-8:continue
        matches.append(BondOrbital(int(idx),int(bond),label,float(raw[idx,j]),
                                   'π' if ' π(' in label else 'σ'))
    return sorted(matches,key=lambda b:(b.bond_index,0 if b.kind=='σ' else 1,b.label))

def iao_site_s_p(result: RHFResult, site:int, coeff:np.ndarray) -> tuple[float,np.ndarray,float]:
    """IAO-projected s amplitude, p-vector and total site projection weight."""
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    B=result.local_basis.B_atomic
    labels=result.local_basis.atomic_labels
    names={label:i for i,label in enumerate(labels)}
    atom=result.mol.atom_symbol(site)
    prefix=f'{atom}{site+1}:'
    z=result.mol.atom_charge(site)
    n=_valence_shell(z)
    amplitudes=B.T@S@np.asarray(coeff,float)
    s=float(amplitudes[names[prefix+str(n)+'s']]) if prefix+str(n)+'s' in names else 0.
    p=np.array([amplitudes[names[prefix+str(n)+'p'+d]] if prefix+str(n)+'p'+d in names else 0. for d in 'xyz'])
    site_ids=[i for i,label in enumerate(labels) if label.startswith(prefix)]
    fraction=float(np.sum(amplitudes[site_ids]**2))
    return s,p,fraction


def ideal_site_orbital(result: RHFResult, seed:MolecularSeed,
                       atom_idx:int,neighbor_idx:int,kind:str='σ',
                       reference:np.ndarray | None=None) -> tuple[np.ndarray,dict]:
    """An idealized VSEPR sigma hybrid or perpendicular pi p orbital.

    The ideal function is a comparison *basis function*, not a variational
    orbital, an eigenfunction, or an independently calculated energy state.
    """
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    xyz=np.asarray(seed.xyz_angstrom,float)
    direction=xyz[neighbor_idx]-xyz[atom_idx]
    direction/=np.linalg.norm(direction)
    indices={label:i for i,label in enumerate(result.local_basis.atomic_labels)}
    prefix=f'{seed.symbols[atom_idx]}{atom_idx+1}:'
    n=_valence_shell(seed.rdkit_mol.GetAtomWithIdx(atom_idx).GetAtomicNum())
    sidx=indices.get(prefix+str(n)+'s')
    pids=[indices.get(prefix+str(n)+'p'+axis) for axis in 'xyz']
    B=result.local_basis.B_atomic
    if kind=='π':
        if any(idx is None for idx in pids):
            raise ValueError('No atomic p orbitals available for a π bond on this atom.')
        if reference is None:
            v=np.eye(3)[int(np.argmin(np.abs(direction)))]
        else:
            _,v,_=iao_site_s_p(result,atom_idx,reference)
        v=v-np.dot(v,direction)*direction
        if np.linalg.norm(v)<1e-8:
            v=np.eye(3)[int(np.argmin(np.abs(direction)))]
            v=v-np.dot(v,direction)*direction
        v/=np.linalg.norm(v)
        ideal=B[:,pids]@v
        meta={'hybrid':'perpendicular p (π)', 'ideal_s_fraction':0.,'ideal_p_fraction':1.,'p_direction':v}
    elif sidx is None:
        raise ValueError('No atomic s orbital for this site.')
    elif any(idx is None for idx in pids):
        ideal=B[:,sidx].copy()
        meta={'hybrid':'1s','ideal_s_fraction':1.,'ideal_p_fraction':0.,'p_direction':None}
    else:
        count=seed.vsepr_domains[atom_idx]
        count=count if count in (2,3,4) else 2
        # For a particular sigma bond, the VSEPR hybrid points toward its bonded atom.
        # Other hybrids on the atom may not align perfectly with measured geometry.
        ideal=np.sqrt(1/count)*B[:,sidx]+np.sqrt((count-1)/count)*(B[:,pids]@direction)
        meta={'hybrid':{2:'sp',3:'sp²',4:'sp³'}[count],
              'ideal_s_fraction':1/count,'ideal_p_fraction':(count-1)/count,
              'p_direction':direction}
    norm=np.sqrt(ideal@S@ideal)
    return ideal/norm,meta


def hybrid_comparison(result,seed,bond:dict,orbital:np.ndarray,kind:str) -> list[dict]:
    rows=[]
    for i,j in ((bond['a'],bond['b']),(bond['b'],bond['a'])):
        ideal,meta=ideal_site_orbital(result,seed,i,j,kind,orbital)
        s,p,f=iao_site_s_p(result,i,orbital)
        valence=s*s+float(p@p)
        pfrac=(p@p)/valence if valence>1e-10 else float('nan')
        q=meta['p_direction']
        cosangle=float('nan')
        if q is not None and np.linalg.norm(p)>1e-8:
            q=np.asarray(q,float)
            if abs(s)>1e-6 and kind=='σ':
                cosangle=float(np.clip(np.dot(np.sign(s)*p/np.linalg.norm(p),q),-1,1))
            else:
                cosangle=float(np.clip(abs(np.dot(p/np.linalg.norm(p),q)),-1,1))
        # For sigma bonds, the angle compares the p vector to the bond-directed
        # ideal hybrid. For pi bonds, the reference p axis is chosen *from* the
        # calculated p vector, so comparing to it would give a tautological 0°.
        # Instead the pi angle measures departure from perpendicularity to the
        # internuclear axis, an independent geometric diagnostic.
        if kind=='π' and np.linalg.norm(p)>1e-8:
            axis=np.asarray(seed.xyz_angstrom[j]-seed.xyz_angstrom[i],float)
            axis/=np.linalg.norm(axis)
            cosangle=float(np.clip(np.sqrt(1-np.clip(abs(np.dot(p/np.linalg.norm(p),axis)),0,1)**2),0,1))
        # Quantify agreement of the *normalized site-projected shape* with the
        # ideal hybrid, not an entire molecular-orbital population.
        S=np.asarray(result.mol.intor('int1e_ovlp'),float)
        atom_prefix=f'{seed.symbols[i]}{i+1}:'
        atom_cols=[k for k,label in enumerate(result.local_basis.atomic_labels)
                   if label.startswith(atom_prefix)]
        B=result.local_basis.B_atomic[:,atom_cols]
        local_component=B @ (B.T@S@np.asarray(orbital,float))
        local_norm=float(local_component@S@local_component)
        local_match=float((local_component@S@ideal)**2/local_norm) if local_norm>1e-10 else float('nan')
        rows.append({'Atom':f'{seed.symbols[i]}{i+1}',
                     'Ideal VSEPR function':meta['hybrid'],
                     'Ideal s %':round(100*meta['ideal_s_fraction'],1),
                     'Calculated s %':round(100*(1-pfrac),1) if np.isfinite(pfrac) else None,
                     'Calculated p %':round(100*pfrac,1) if np.isfinite(pfrac) else None,
                     'Atom projection %':round(100*f,1),
                     'Ideal hybrid match %':round(100*local_match,1) if np.isfinite(local_match) else None,
                     'p direction error °':round(float(np.degrees(np.arccos(cosangle))),1) if np.isfinite(cosangle) else None,
                     'Ideal AO':ideal})
    return rows


@dataclass(frozen=True)
class LocalizedPair:
    """One occupied IBO classified as a bond, lone pair, or uncertain pair.

    The catalog partitions the existing *occupied valence IBOs*: it does not
    alter the RHF density, introduce new orbital localization, or identify
    lone pairs from VSEPR electron-count rules alone.
    """
    orbital_index: int
    label: str
    kind: str
    bond_index: int | None = None
    atom_index: int | None = None
    confidence: float = 0.0
    site_weight: float = 0.0


def localized_pair_catalog(result: RHFResult, seed: MolecularSeed,
                           localized: np.ndarray,
                           bonds: list[BondOrbital] | None = None) -> list[LocalizedPair]:
    """Enumerate every localized occupied valence pair, including lone pairs.

    Strongly atom-centered leftover IBOs are labeled lone pairs. Ambiguous
    leftovers remain explicitly 'unassigned' rather than inventing a lone
    pair or silently dropping occupied electrons. Core pairs are excluded.
    """
    bond_matches = bond_orbital_map(result, seed, localized) if bonds is None else bonds
    catalog = [LocalizedPair(m.orbital_index, m.label, m.kind,
                             bond_index=m.bond_index, confidence=m.confidence)
               for m in bond_matches]
    assigned = {m.orbital_index for m in bond_matches}
    lone_indices = {}
    unassigned = 0
    for index in range(localized.shape[1]):
        if index in assigned:
            continue
        populations = atomic_iao_populations(result, localized[:,index])
        atoms = [(i, float(populations.get(f'{sym}{i+1}', 0.)))
                 for i,sym in enumerate(seed.symbols)]
        atoms.sort(key=lambda pair:pair[1], reverse=True)
        host, weight = atoms[0]
        runner = atoms[1][1] if len(atoms)>1 else 0.
        # Absolute IAO weights, not normalized by the captured IAO subspace.
        # The cutoff also rejects substantial missing analysis-space coverage.
        confident = weight >= 0.60 and weight >= 2.0 * max(runner, 1e-8)
        if confident:
            s, p, _ = iao_site_s_p(result, host, localized[:,index])
            frac_s = float(s*s / (s*s + p@p)) if s*s+p@p > 1e-12 else 0.
            character = 's-rich' if frac_s >= .65 else 'p-rich' if frac_s <= .30 else 's/p mixed'
            lone_indices[host] = lone_indices.get(host,0)+1
            label = f'{seed.symbols[host]}{host+1} lone pair {lone_indices[host]} ({character})'
            catalog.append(LocalizedPair(index,label,'lone pair',atom_index=host,
                                          confidence=weight,site_weight=weight))
        else:
            unassigned += 1
            catalog.append(LocalizedPair(index,f'Unassigned localized pair {unassigned}',
                                          'unassigned',confidence=weight,
                                          atom_index=host,site_weight=weight))
    return catalog


def lone_pair_virtual_seed(result: RHFResult, localized_orbital: np.ndarray,
                           atom_index: int) -> np.ndarray:
    """Atom-centered virtual seed matched to a lone pair's s/p direction.

    Constructs radially distinct atomic-Gaussian trial hybrids on the host
    atom, projects them into the RHF *virtual* space, and picks the strongest
    localized residual. Subsequent pair-space construction orthogonalizes
    these seeds across selected pairs. Unlike bond trials this never requires
    a fictitious second atom for a lone pair.
    """
    import re
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    occupied=np.asarray(result.coefficients_ao[:,:result.nocc],float)
    virtuals=np.asarray(result.coefficients_ao[:,result.nocc:],float)
    if virtuals.shape[1]==0:
        raise ValueError('No available virtual orbital to correlate this lone pair.')
    seed=np.asarray(localized_orbital,float)
    amps=result.local_basis.B_atomic.T@S@seed
    sym=result.mol.atom_symbol(atom_index)
    ref_labels=result.local_basis.atomic_labels
    reference={key.split(':')[1]:float(value)
               for key,value in zip(ref_labels,amps)
               if key.startswith(f'{sym}{atom_index+1}:')}
    valence=_valence_shell(int(result.mol.atom_charge(atom_index)))
    channel={part:reference.get(f'{valence}{part}',0.) for part in ('s','px','py','pz')}
    # PySCF label tuples are (atom_index, symbol, '2p', 'x') or
    # (atom_index, symbol, '2s', ''). Group all radial shells.
    shells={}
    for j,row in enumerate(result.mol.ao_labels(fmt=False)):
        if int(row[0]) != atom_index:
            continue
        label=_shell_label(row)
        match=re.fullmatch(r'(\d+)(s|px|py|pz)',label)
        if match:
            n=int(match.group(1)); kind=match.group(2)
            shells.setdefault(n,{})[kind]=j
    candidates=[]
    for n,entries in shells.items():
        trial=np.zeros(S.shape[0])
        for kind,j in entries.items():
            trial[j]=channel[kind]
        if np.linalg.norm(trial)<1e-12:
            continue
        trial -= occupied@(occupied.T@S@trial)
        norm=float(trial@S@trial)
        if norm>1e-10:
            trial/=np.sqrt(norm)
            # Prefer the local residual whose angular shape matches the IBO;
            # among similar candidates favor small AO radial excitation.
            candidates.append((norm/(1.+.12*abs(n-valence)),trial))
    if not candidates:
        raise ValueError(f'No orbital-character-matched atom-local virtual seed for '
                         f'{sym}{atom_index+1}. The lone pair can still be inspected in MO mode.')
    _,best=max(candidates,key=lambda item:item[0])
    return best


def lone_pair_ideal_reference(result: RHFResult, seed: MolecularSeed,
                              atom_index: int, occupied_orbital: np.ndarray
                              ) -> tuple[np.ndarray,dict]:
    """Closest unused VSEPR hybrid on the lone-pair atom, for comparison only.

    Ideal lone-pair axes are nonunique (especially for equivalent lone pairs).
    The displayed squared overlap is a *shape convention*, not a physical
    assignment of an electron pair to a unique VSEPR direction.
    """
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    B=result.local_basis.B
    labels=result.local_basis.labels
    prefix=f'{seed.symbols[atom_index]}{atom_index+1}:'
    count=int(seed.vsepr_domains[atom_index])
    degree=seed.rdkit_mol.GetAtomWithIdx(atom_index).GetDegree()
    if count not in (2,3,4):
        raise ValueError('VSEPR hybrid is not defined for this lone-pair site.')
    # Hybrid columns are ordered bond-oriented first, then remaining directions
    # using the same _match_hybrid_directions rule as the IAO analysis basis.
    candidates=[(i,B[:,i]) for i,label in enumerate(labels)
                if label.startswith(prefix) and
                f'-hybrid-' in label and
                int(label.rsplit('-',1)[1])>degree]
    if not candidates:
        raise ValueError('No unused ideal hybrid direction for this atom.')
    index,ideal=max(candidates,key=lambda pair:abs(float(pair[1]@S@occupied_orbital)))
    return np.asarray(ideal,float),{
        'hybrid':f'{ {2:"sp",3:"sp²",4:"sp³"}[count]} lone-pair reference',
        'note':'Direction is nonunique; this is one basis convention.',
        'ideal_s_fraction':1./count,
        'ideal_p_fraction':(count-1.)/count,
        'name':labels[index],
    }


@dataclass(frozen=True)
class LocalizedPair:
    """One occupied IBO classified as a bond, lone pair, or uncertain pair.

    The catalog partitions the existing *occupied valence IBOs*: it does not
    alter the RHF density, introduce new orbital localization, or identify
    lone pairs from VSEPR electron-count rules alone.
    """
    orbital_index: int
    label: str
    kind: str
    bond_index: int | None = None
    atom_index: int | None = None
    confidence: float = 0.0
    site_weight: float = 0.0


def localized_pair_catalog(result: RHFResult, seed: MolecularSeed,
                           localized: np.ndarray,
                           bonds: list[BondOrbital] | None = None) -> list[LocalizedPair]:
    """Enumerate every localized occupied valence pair, including lone pairs.

    Strongly atom-centered leftover IBOs are labeled lone pairs. Ambiguous
    leftovers remain explicitly 'unassigned' rather than inventing a lone
    pair or silently dropping occupied electrons. Core pairs are excluded.
    """
    bond_matches = bond_orbital_map(result, seed, localized) if bonds is None else bonds
    catalog = [LocalizedPair(m.orbital_index, m.label, m.kind,
                             bond_index=m.bond_index, confidence=m.confidence)
               for m in bond_matches]
    assigned = {m.orbital_index for m in bond_matches}
    lone_indices = {}
    unassigned = 0
    for index in range(localized.shape[1]):
        if index in assigned:
            continue
        populations = atomic_iao_populations(result, localized[:,index])
        atoms = [(i, float(populations.get(f'{sym}{i+1}', 0.)))
                 for i,sym in enumerate(seed.symbols)]
        atoms.sort(key=lambda pair:pair[1], reverse=True)
        host, weight = atoms[0]
        runner = atoms[1][1] if len(atoms)>1 else 0.
        # Absolute IAO weights, not normalized by the captured IAO subspace.
        # The cutoff also rejects substantial missing analysis-space coverage.
        confident = weight >= 0.60 and weight >= 2.0 * max(runner, 1e-8)
        if confident:
            s, p, _ = iao_site_s_p(result, host, localized[:,index])
            frac_s = float(s*s / (s*s + p@p)) if s*s+p@p > 1e-12 else 0.
            character = 's-rich' if frac_s >= .65 else 'p-rich' if frac_s <= .30 else 's/p mixed'
            lone_indices[host] = lone_indices.get(host,0)+1
            label = f'{seed.symbols[host]}{host+1} lone pair {lone_indices[host]} ({character})'
            catalog.append(LocalizedPair(index,label,'lone pair',atom_index=host,
                                          confidence=weight,site_weight=weight))
        else:
            unassigned += 1
            catalog.append(LocalizedPair(index,f'Unassigned localized pair {unassigned}',
                                          'unassigned',confidence=weight,
                                          atom_index=host,site_weight=weight))
    return catalog


def lone_pair_virtual_seed(result: RHFResult, localized_orbital: np.ndarray,
                           atom_index: int) -> np.ndarray:
    """Atom-centered virtual seed matched to a lone pair's s/p direction.

    Constructs radially distinct atomic-Gaussian trial hybrids on the host
    atom, projects them into the RHF *virtual* space, and picks the strongest
    localized residual. Subsequent pair-space construction orthogonalizes
    these seeds across selected pairs. Unlike bond trials this never requires
    a fictitious second atom for a lone pair.
    """
    import re
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    occupied=np.asarray(result.coefficients_ao[:,:result.nocc],float)
    virtuals=np.asarray(result.coefficients_ao[:,result.nocc:],float)
    if virtuals.shape[1]==0:
        raise ValueError('No available virtual orbital to correlate this lone pair.')
    seed=np.asarray(localized_orbital,float)
    amps=result.local_basis.B_atomic.T@S@seed
    sym=result.mol.atom_symbol(atom_index)
    ref_labels=result.local_basis.atomic_labels
    reference={key.split(':')[1]:float(value)
               for key,value in zip(ref_labels,amps)
               if key.startswith(f'{sym}{atom_index+1}:')}
    valence=_valence_shell(int(result.mol.atom_charge(atom_index)))
    channel={part:reference.get(f'{valence}{part}',0.) for part in ('s','px','py','pz')}
    # PySCF label tuples are (atom_index, symbol, '2p', 'x') or
    # (atom_index, symbol, '2s', ''). Group all radial shells.
    shells={}
    for j,row in enumerate(result.mol.ao_labels(fmt=False)):
        if int(row[0]) != atom_index:
            continue
        label=_shell_label(row)
        match=re.fullmatch(r'(\d+)(s|px|py|pz)',label)
        if match:
            n=int(match.group(1)); kind=match.group(2)
            shells.setdefault(n,{})[kind]=j
    candidates=[]
    for n,entries in shells.items():
        trial=np.zeros(S.shape[0])
        for kind,j in entries.items():
            trial[j]=channel[kind]
        if np.linalg.norm(trial)<1e-12:
            continue
        trial -= occupied@(occupied.T@S@trial)
        norm=float(trial@S@trial)
        if norm>1e-10:
            trial/=np.sqrt(norm)
            # Prefer the local residual whose angular shape matches the IBO;
            # among similar candidates favor small AO radial excitation.
            candidates.append((norm/(1.+.12*abs(n-valence)),trial))
    if not candidates:
        raise ValueError(f'No orbital-character-matched atom-local virtual seed for '
                         f'{sym}{atom_index+1}. The lone pair can still be inspected in MO mode.')
    _,best=max(candidates,key=lambda item:item[0])
    return best


def lone_pair_ideal_reference(result: RHFResult, seed: MolecularSeed,
                              atom_index: int, occupied_orbital: np.ndarray
                              ) -> tuple[np.ndarray,dict]:
    """Closest unused VSEPR hybrid on the lone-pair atom, for comparison only.

    Ideal lone-pair axes are nonunique (especially for equivalent lone pairs).
    The displayed squared overlap is a *shape convention*, not a physical
    assignment of an electron pair to a unique VSEPR direction.
    """
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    B=result.local_basis.B
    labels=result.local_basis.labels
    prefix=f'{seed.symbols[atom_index]}{atom_index+1}:'
    count=int(seed.vsepr_domains[atom_index])
    degree=seed.rdkit_mol.GetAtomWithIdx(atom_index).GetDegree()
    if count not in (2,3,4):
        raise ValueError('VSEPR hybrid is not defined for this lone-pair site.')
    # Hybrid columns are ordered bond-oriented first, then remaining directions
    # using the same _match_hybrid_directions rule as the IAO analysis basis.
    candidates=[(i,B[:,i]) for i,label in enumerate(labels)
                if label.startswith(prefix) and
                f'-hybrid-' in label and
                int(label.rsplit('-',1)[1])>degree]
    if not candidates:
        raise ValueError('No unused ideal hybrid direction for this atom.')
    index,ideal=max(candidates,key=lambda pair:abs(float(pair[1]@S@occupied_orbital)))
    return np.asarray(ideal,float),{
        'hybrid':f'{ {2:"sp",3:"sp²",4:"sp³"}[count]} lone-pair reference',
        'note':'Direction is nonunique; this is one basis convention.',
        'ideal_s_fraction':1./count,
        'ideal_p_fraction':(count-1.)/count,
        'name':labels[index],
    }
