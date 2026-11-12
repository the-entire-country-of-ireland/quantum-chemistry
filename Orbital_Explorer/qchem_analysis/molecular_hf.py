"""Restricted Hartree-Fock molecular geometry optimization in two AO spaces.

Both modes use the same atom-centered Gaussian integral engine. 'full' uses
all basis functions; 'reduced' uses only saved isolated-atom HF core and
s/p-shell functions, transformed into a bond-aligned hybrid representation.
Only neutral/charged even-electron closed-shell molecules are supported.

Install alongside atomic_orbitals.py and hf_calculations.py in qchem_analysis.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation
from pyscf import gto, scf, ao2mo

from .atomic_orbitals import (calculate_atomic_hf_orbitals,
                              load_atomic_orbitals, save_atomic_orbitals)
from .molecular_geometry import (MolecularSeed, geometry_from_smiles,
                                 BOHR_PER_ANGSTROM, align_molecular_axes)


def _templates(count: int) -> np.ndarray:
    if count == 2:
        return np.array([[1,0,0],[-1,0,0]],float)
    if count == 3:
        a = np.arange(3)*2*np.pi/3
        return np.column_stack((np.cos(a), np.sin(a), np.zeros(3)))
    if count == 4:
        return np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)/np.sqrt(3)
    raise ValueError("Hybrid count must be 2, 3, or 4")


def _match_hybrid_directions(template: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Properly rotate and reorder ideal hybrid directions toward bond vectors."""
    targets = np.asarray(targets, dtype=float)
    if not len(targets):
        return template.copy()
    targets = targets / np.linalg.norm(targets,axis=1)[:,None]
    n = min(len(targets),len(template))
    best_score, best = -np.inf, None
    for selected in permutations(range(len(template)),n):
        src = template[list(selected)]
        if n == 1:
            rot,_ = Rotation.align_vectors(targets, src)
            Q = rot.as_matrix().T
        else:
            u,_,vt = np.linalg.svd(src.T @ targets)
            d = np.linalg.det(u @ vt)
            Q = u @ np.diag([1.,1.,np.sign(d)]) @ vt
        reordered = list(selected) + [i for i in range(len(template)) if i not in selected]
        aligned = template[reordered] @ Q
        score = float(np.sum(aligned[:n]*targets[:n]))
        if score > best_score:
            best_score, best = score, aligned
    return best


def _hybrid_transform(directions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Orthonormal 4x4 matrix: columns are sp/sp2/sp3 hybrids + unused p."""
    m = len(directions)
    columns = [np.r_[np.sqrt(1./m), np.sqrt((m-1.)/m)*d] for d in directions]
    # Complete the valence span by retaining every unused perpendicular p.
    p_occupied = directions.T
    u, s, _ = np.linalg.svd(p_occupied, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    for perp in u[:, rank:].T:
        columns.append(np.r_[0.,perp])
    mat = np.column_stack(columns)
    if mat.shape != (4,4) or not np.allclose(mat.T @ mat, np.eye(4),atol=1e-8):
        raise RuntimeError("Failed to build complete orthonormal hybrid basis")
    scheme = {2: "sp", 3: "sp2", 4: "sp3"}[m]
    names = [f"{scheme}-hybrid-{i+1}" for i in range(m)]
    names += [f"perpendicular-p-{i+1}" for i in range(4-m)]
    return mat, names


@dataclass
class AtomicLibrary:
    """Saved HF atomic functions for each element and one Gaussian basis."""
    basis: str
    directory: Path
    atoms: dict

    @classmethod
    def prepare(cls, atomic_numbers, basis='cc-pVDZ', directory='saved_atomic_hf',
                saved_paths: Optional[dict[int,str|Path]] = None):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        atoms = {}
        saved_paths = {} if saved_paths is None else saved_paths
        for z in sorted(set(map(int,atomic_numbers))):
            filename = Path(saved_paths.get(z, directory / f"Z{z}_{basis.replace('/','_')}.npz"))
            if filename.exists():
                result = load_atomic_orbitals(filename)
                if result.atomic_number != z or result.basis.lower()!=basis.lower():
                    raise ValueError(f"Atomic file {filename} has inconsistent atom/basis")
            else:
                filename.parent.mkdir(parents=True,exist_ok=True)
                print(f"Computing and saving isolated atom Z={z} in {basis}...")
                result = calculate_atomic_hf_orbitals(z,basis)
                save_atomic_orbitals(result,filename)
            atoms[z] = result
        return cls(basis,directory,atoms)


@dataclass
class LocalBasis:
    """Atom-local, nonorthogonal saved-orbital/hybrid basis in molecular AOs."""
    B: np.ndarray  # full Gaussian AO to reduced AO coefficients
    labels: list[str]
    atom_ids: list[int]
    B_atomic: np.ndarray
    atomic_labels: list[str]



def _geometry_hybrid_count(atom, xyz_angstrom: np.ndarray) -> int | None:
    """Choose an interpretable local s/p basis from actual bond directions.

    This is a *representation convention*, not a measured electronic
    hybridization fraction. The complete ns+np span is always retained.

    A terminal atom with a single sigma bond uses a bond-aligned sp pair and
    two perpendicular p orbitals. This intentionally avoids labeling HF's
    F contribution as four tetrahedral sp3 functions just because F has
    three lone pairs. A terminal double-bonded atom uses sp2 to retain a
    perpendicular p direction for its pi bond.
    """
    neighbors = list(atom.GetNeighbors())
    degree = len(neighbors)
    if degree == 0:
        return None
    if degree == 1:
        bond = atom.GetBonds()[0]
        order = float(bond.GetBondTypeAsDouble())
        return 3 if 1.4 <= order < 2.5 else 2
    center = int(atom.GetIdx())
    v = np.asarray(xyz_angstrom[[n.GetIdx() for n in neighbors]], dtype=float)
    v = v - np.asarray(xyz_angstrom[center], dtype=float)
    lengths = np.linalg.norm(v, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError('Coincident nuclei in geometry-based hybridization')
    v /= lengths[:, None]
    if degree == 2:
        angle = np.degrees(np.arccos(np.clip(v[0] @ v[1], -1., 1.)))
        if angle >= 155.:
            return 2          # linear: sp
        if angle >= 115.:
            return 3          # ~120 deg: sp2
        return 4              # strongly bent: sp3-style
    if degree == 3:
        # Scalar triple product is zero for three coplanar sigma bonds;
        # it increases as the center becomes pyramidal.
        pyramidal = abs(float(np.linalg.det(v[:3])))
        return 3 if pyramidal < 0.12 else 4
    if degree == 4:
        return 4
    return None  # beyond supported sp/sp2/sp3 geometries


def build_local_basis(mol, xyz_angstrom: np.ndarray, rdkit_mol,
                      library: AtomicLibrary) -> LocalBasis:
    nao = mol.nao_nr()
    aoslices = mol.aoslice_by_atom()
    columns, labels, atom_ids = [], [], []
    atomic_columns, atomic_labels = [], []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        z = atom.GetAtomicNum()
        saved = library.atoms[z]
        ao_start, ao_end = map(int,aoslices[i,2:4])
        if ao_end-ao_start != saved.molecule.nao_nr():
            raise RuntimeError("Atomic/molecular AO ordering differs. Use matching basis & spherical functions.")
        shell = saved.principal_n
        orbital_names = list(saved.shell_orbitals)
        for name in orbital_names:
            base_col = np.zeros(nao,dtype=float)
            base_col[ao_start:ao_end] = saved.shell_orbitals[name]
            atomic_columns.append(base_col)
            atomic_labels.append(f"{atom.GetSymbol()}{i+1}:{name}")
        local = []
        names = []
        for name in orbital_names:
            if not name.startswith(str(shell)):
                local.append(saved.shell_orbitals[name])
                names.append(name)
        valence = [f'{shell}s']+[f'{shell}p{ax}' for ax in 'xyz']
        if all(name in saved.shell_orbitals for name in valence):
            valence_matrix = np.column_stack([saved.shell_orbitals[name] for name in valence])
            egroup = _geometry_hybrid_count(atom, xyz_angstrom)
            if egroup in (2,3,4):
                nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
                vectors = xyz_angstrom[nbrs] - xyz_angstrom[i] if nbrs else np.zeros((0,3))
                dirs = _match_hybrid_directions(_templates(egroup), vectors)
                transform, hybrid_labels = _hybrid_transform(dirs)
                for j,name in enumerate(hybrid_labels):
                    local.append(valence_matrix @ transform[:,j]);names.append(name)
            else:
                for j,name in enumerate(valence):
                    local.append(valence_matrix[:,j]);names.append(name)
        else:
            for name in orbital_names:
                if name.startswith(str(shell)):
                    local.append(saved.shell_orbitals[name]);names.append(name)
        for c,name in zip(local,names):
            full = np.zeros(nao,dtype=float)
            full[ao_start:ao_end] = c
            columns.append(full)
            labels.append(f"{atom.GetSymbol()}{i+1}:{name}")
            atom_ids.append(i)
    B = np.column_stack(columns)
    overlap = B.T @ mol.intor('int1e_ovlp') @ B
    evals = np.linalg.eigvalsh(overlap)
    if evals[0]<1e-8:
        raise ValueError(f"Reduced basis became singular (min overlap eigenvalue {evals[0]:.2e})")
    return LocalBasis(B,labels,atom_ids,np.column_stack(atomic_columns), atomic_labels)


def molecule_from_xyz(symbols, xyz_angstrom, basis: str, charge: int = 0):
    return gto.M(atom=[(s,p.tolist()) for s,p in zip(symbols,xyz_angstrom)],
                 unit='Angstrom',basis=basis,charge=int(charge),spin=0,
                 verbose=0,cart=False)


@dataclass
class RHFResult:
    mode: str
    energy_hartree: float
    xyz_angstrom: np.ndarray
    mol: object
    mf: object
    local_basis: LocalBasis
    coefficients_ao: np.ndarray
    orbital_energies: np.ndarray
    converged: bool
    geometry_converged: bool
    geometry_message: str
    core_spatial_count: int
    history: list[float]
    frozen_core: bool = False
    occupations: Optional[np.ndarray] = None

    @property
    def nocc(self):
        return int(self.mol.nelectron//2)

    @property
    def valence_spatial_indices(self):
        return list(range(self.core_spatial_count,self.nocc))

    def valence_spin_orbitals(self):
        """RHF: both alpha and beta for each occupied valence spatial MO."""
        result=[]
        for i in self.valence_spatial_indices:
            for spin in ('alpha','beta'):
                result.append(dict(spin=spin, mo_index=i,
                                   energy_hartree=float(self.orbital_energies[i]),
                                   coefficients_ao=self.coefficients_ao[:,i].copy()))
        return result


def core_count_from_rdkit(rdkit_mol) -> int:
    # Conventional chemically inert core for valence reporting and optional freezing.
    return sum(1 if a.GetAtomicNum() in range(3,11) else
               5 if a.GetAtomicNum() in range(11,19) else 0
               for a in rdkit_mol.GetAtoms())


def _core_labels_for_z(z: int) -> tuple[str, ...]:
    """Closed-shell core partition for elements H--Ar, not an ECP."""
    if 3 <= z <= 10:
        return ('1s',)
    if 11 <= z <= 18:
        return ('1s', '2s', '2px', '2py', '2pz')
    return ()


def _frozen_core_at_geometry(seed, xyz, basis, charge, library, *, reduced):
    """Frozen occupied *spatial* core orbitals; variational valence RHF.

    Core orbitals are atom-centered isolated-HF functions, orthonormalized
    in the current molecular overlap metric. They follow nuclear positions
    during geometry optimization but are never varied by the valence SCF.
    All core--valence Coulomb/exchange terms are included exactly within
    the chosen finite AO space. Numerical geometry gradients include this
    geometry dependence.
    """
    mol = molecule_from_xyz(seed.symbols, xyz, basis, charge)
    local = build_local_basis(mol, xyz, seed.rdkit_mol, library)
    B = local.B if reduced else np.eye(mol.nao_nr())
    S_ao = mol.intor('int1e_ovlp')
    S = B.T @ S_ao @ B
    se, U = np.linalg.eigh(S)
    if se.min() <= 1e-9:
        raise ValueError('Ill-conditioned SCF overlap in frozen-core solver')
    X = (U * (se**-0.5)) @ U.T   # X.T S X = identity
    nao = mol.nao_nr()
    atom_slices = mol.aoslice_by_atom()
    core_cols=[]
    for i, atom in enumerate(seed.rdkit_mol.GetAtoms()):
        z=atom.GetAtomicNum()
        lo, hi=map(int, atom_slices[i,2:4])
        saved=library.atoms[z]
        for name in _core_labels_for_z(z):
            c=np.zeros(nao)
            c[lo:hi]=saved.shell_orbitals[name]
            core_cols.append(c)
    nc=len(core_cols)
    if 2*nc >= mol.nelectron:
        raise ValueError('No valence electrons remain in frozen-core RHF')
    if nc:
        # Orthogonalize core functions and obtain their orthogonal complement.
        C0=np.column_stack(core_cols)
        # B projects molecular AOs into the reduced AO space, if necessary.
        core_coords = X.T @ S @ np.linalg.solve(S, B.T @ S_ao @ C0)
        Qc,_ = np.linalg.qr(core_coords, mode='reduced')
        Qv=null_space(Qc.T)
        core=X @ Qc
    else:
        Qv=np.eye(B.shape[1]); core=np.empty((B.shape[1],0))
    valence=X @ Qv
    assert np.allclose(valence.T @ S @ valence, np.eye(valence.shape[1]),atol=1e-7)
    Dcore_ao=2*(B@core)@(B@core).T
    h_ao=scf.hf.get_hcore(mol)
    Vcore_ao=scf.hf.get_veff(mol,Dcore_ao)
    Ecore=float(np.einsum('ij,ji->',Dcore_ao,h_ao+0.5*Vcore_ao))
    V=B@valence
    h_eff=V.T @ (h_ao+Vcore_ao) @ V
    eri=ao2mo.kernel(mol,V)
    # Use an otherwise identical molecule with a lower electron count; the
    # valence RHF solver sees an orthonormal basis and the core mean field.
    vmol=mol.copy()
    vmol.charge = int(mol.charge + 2*nc)
    mf=scf.RHF(vmol)
    mf.verbose=0; mf.conv_tol=1e-9; mf.max_cycle=120
    mf.get_hcore=lambda *args: h_eff
    mf.get_ovlp=lambda *args: np.eye(V.shape[1])
    mf._eri=ao2mo.restore(8,eri,V.shape[1])
    mf.init_guess='1e'
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Frozen-core valence RHF did not converge')
    # The core electronic energy is omitted by the valence-only SCF energy.
    total_energy=float(mf.e_tot + Ecore)
    # Return occupied core first, then valence MOs and the virtual complement.
    Cfull=np.column_stack((B@core,V@mf.mo_coeff))
    # Core diagonal expectation values in the final full effective Fock.
    Dtotal=Dcore_ao+2*(V@mf.mo_coeff[:,:vmol.nelectron//2]) @ (V@mf.mo_coeff[:,:vmol.nelectron//2]).T
    F=h_ao+scf.hf.get_veff(mol,Dtotal)
    ecore=np.einsum('ki,kl,li->i',B@core,F,B@core)
    energies=np.r_[ecore,mf.mo_energy]
    occ=np.r_[np.full(nc,2.),mf.mo_occ]
    return mol,mf,local,total_energy,Cfull,energies,occ


def _reduced_at_geometry(seed, xyz, basis, charge, library, *, verbose=0):
    mol = molecule_from_xyz(seed.symbols,xyz,basis,charge)
    local = build_local_basis(mol,xyz,seed.rdkit_mol,library)
    B = local.B
    S_ao = mol.intor('int1e_ovlp')
    S = B.T @ S_ao @ B
    one = scf.hf.get_hcore(mol)
    h = B.T @ one @ B
    eri = ao2mo.kernel(mol, B)
    mf = scf.RHF(mol)
    mf.verbose = verbose
    mf.conv_tol = 1e-9
    mf.max_cycle=100
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: S
    mf._eri = ao2mo.restore(8,eri,B.shape[1])
    mf.init_guess = '1e'
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Reduced-basis RHF did not converge')
    return mol, mf, local


def optimize_reduced(seed: MolecularSeed, library: AtomicLibrary,
                     charge: int = 0, maxsteps=35, geom_tolerance=1e-3,
                     frozen_core=False) -> RHFResult:
    """Variational RHF at each geometry; finite-difference nuclear gradient.

    Derivatives are numerical to include *all* basis motion / Pulay effects;
    the standard full-basis analytic RHF gradients are not valid for a custom
    geometry-dependent, projected AO basis.
    """
    initial = seed.xyz_angstrom.copy()
    history=[]
    cache = {}
    def calc(flat):
        xyz = np.asarray(flat).reshape(-1,3) / BOHR_PER_ANGSTROM
        key = tuple(np.round(flat,12))
        if key not in cache:
            if frozen_core:
                mol,mf,local,energy,coeff,eps,occ = _frozen_core_at_geometry(
                    seed,xyz,library.basis,charge,library,reduced=True)
            else:
                mol,mf,local = _reduced_at_geometry(seed,xyz,library.basis,charge,library)
                energy=float(mf.e_tot)
                coeff=local.B@mf.mo_coeff
                eps=mf.mo_energy.copy()
                occ=mf.mo_occ.copy()
            cache.clear()  # avoid huge integral arrays from many geometries
            cache[key] = (energy,mol,mf,local,coeff,eps,occ)
            history.append(energy)
        return cache[key]
    res = minimize(lambda x: calc(x)[0], initial.ravel()*BOHR_PER_ANGSTROM,
                   method='BFGS', jac='3-point', options={'gtol':geom_tolerance,
                                                        'maxiter':maxsteps,
                                                        'finite_diff_rel_step':1e-3})
    final_xyz=align_molecular_axes(res.x.reshape(-1,3)/BOHR_PER_ANGSTROM,
                                   seed.rdkit_mol)
    if frozen_core:
        mol,mf,local,energy,coeff,eps,occ=_frozen_core_at_geometry(
            seed,final_xyz,library.basis,charge,library,reduced=True)
    else:
        mol,mf,local=_reduced_at_geometry(seed,final_xyz,library.basis,charge,library)
        energy=float(mf.e_tot)
        coeff=local.B@mf.mo_coeff
        eps=mf.mo_energy.copy()
        occ=mf.mo_occ.copy()
    return RHFResult('reduced',energy,final_xyz,
                     mol,mf,local,coeff,eps,
                     bool(mf.converged), bool(res.success),str(res.message),
                     core_count_from_rdkit(seed.rdkit_mol),history,
                     frozen_core=frozen_core,occupations=occ)


def optimize_full(seed: MolecularSeed, library: AtomicLibrary, charge=0,
                  maxsteps=70, geom_tolerance=1e-4, seed_from_reduced=True,
                  frozen_core=False) -> RHFResult:
    """Full Gaussian RHF: analytic gradients, or numerical when core is frozen."""
    if frozen_core:
        history=[]; cache={}
        def calc(x):
            key=tuple(np.round(x,12))
            if key not in cache:
                xyz=np.asarray(x).reshape(-1,3)/BOHR_PER_ANGSTROM
                value=_frozen_core_at_geometry(seed,xyz,library.basis,charge,library,reduced=False)
                cache.clear(); cache[key]=value
                history.append(value[3])
            return cache[key]
        res=minimize(lambda x:calc(x)[3],seed.xyz_angstrom.ravel()*BOHR_PER_ANGSTROM,
                     method='BFGS',jac='3-point',
                     options={'maxiter':maxsteps,'gtol':geom_tolerance,
                              'finite_diff_rel_step':1e-3})
        final_xyz=align_molecular_axes(res.x.reshape(-1,3)/BOHR_PER_ANGSTROM,
                                       seed.rdkit_mol)
        mol,mf,local,e,coeff,eps,occ=_frozen_core_at_geometry(
            seed,final_xyz,library.basis,charge,library,reduced=False)
        return RHFResult('full',e,final_xyz,
                         mol,mf,local,coeff,eps,bool(mf.converged),
                         bool(res.success),str(res.message),
                         core_count_from_rdkit(seed.rdkit_mol),history,
                         frozen_core=True,occupations=occ)
    xyz=seed.xyz_angstrom
    mol=molecule_from_xyz(seed.symbols,xyz,library.basis,charge)
    dm0=None
    if seed_from_reduced:
        _,reduced,local=_reduced_at_geometry(seed,xyz,library.basis,charge,library)
        C = local.B @ reduced.mo_coeff[:,:mol.nelectron//2]
        dm0 = 2. * C @ C.T
    mf=scf.RHF(mol)
    mf.conv_tol=1e-9
    mf.max_cycle=100
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError('Initial full Gaussian-basis RHF did not converge')
    gradient_scanner=mf.nuc_grad_method().as_scanner()
    history=[float(mf.e_tot)]
    saved={}
    def eval_grad(x):
        key=tuple(np.round(x,12))
        if saved.get('key')!=key:
            new_mol=mol.set_geom_(x.reshape(-1,3),unit='Bohr',inplace=False)
            e,g=gradient_scanner(new_mol)
            saved.update(key=key, value=(float(e),np.asarray(g).ravel()))
            history.append(float(e))
        return saved['value']
    result=minimize(lambda x:eval_grad(x)[0], mol.atom_coords().ravel(),
                    method='BFGS',jac=lambda x:eval_grad(x)[1],
                    options={'maxiter':maxsteps,'gtol':geom_tolerance})
    final_xyz=align_molecular_axes(result.x.reshape(-1,3)/BOHR_PER_ANGSTROM,
                                  seed.rdkit_mol)
    final_mol=molecule_from_xyz(seed.symbols, final_xyz,library.basis,charge)
    final_mf=scf.RHF(final_mol)
    final_mf.conv_tol=1e-10
    final_mf.max_cycle=120
    final_mf.kernel()
    if not final_mf.converged:
        raise RuntimeError('Final full-basis RHF did not converge')
    local=build_local_basis(final_mol,final_xyz,seed.rdkit_mol,library)
    return RHFResult('full',float(final_mf.e_tot),final_xyz,final_mol,final_mf,
                     local,final_mf.mo_coeff.copy(),final_mf.mo_energy.copy(),
                     True,bool(result.success),str(result.message),
                     core_count_from_rdkit(seed.rdkit_mol),history,
                     frozen_core=False,occupations=final_mf.mo_occ.copy())


def run_both(smiles: str, basis='cc-pVDZ', charge=0,
             atom_directory='saved_atomic_hf', random_seed=11,
             reduced_steps=35, full_steps=70, saved_paths=None,
             frozen_core=False):
    """Start from identical VSEPR seed; independently optimize both HF models."""
    seed=geometry_from_smiles(smiles,random_seed)
    if charge is None:
        charge=sum(a.GetFormalCharge() for a in seed.rdkit_mol.GetAtoms())
    nelec = sum(a.GetAtomicNum() for a in seed.rdkit_mol.GetAtoms()) - int(charge)
    if nelec % 2:
        raise ValueError('Odd electron count: RHF cannot represent this state; use ROHF/UHF')
    if nelec<2:
        raise ValueError('Must have >=2 electrons')
    lib=AtomicLibrary.prepare((a.GetAtomicNum() for a in seed.rdkit_mol.GetAtoms()),
                              basis,atom_directory,saved_paths=saved_paths)
    reduced=optimize_reduced(seed,lib,charge,maxsteps=reduced_steps,
                             frozen_core=frozen_core)
    full=optimize_full(seed,lib,charge,maxsteps=full_steps,
                       frozen_core=frozen_core)
    return seed,lib,reduced,full


def decomposition(result: RHFResult, mo_index:int, representation='atomic',
                  coefficients_ao=None):
    """S-metric projection of an MO into saved atomic or hybrid functions.

    In the full basis, the unrepresented residual is reported; use the signed
    coefficients ONLY as amplitudes, not orthogonal orbital probabilities.
    """
    if representation == 'hybrid':
        B=result.local_basis.B
        labels=result.local_basis.labels
    elif representation == 'atomic':
        B=result.local_basis.B_atomic
        labels=result.local_basis.atomic_labels
    else:
        raise ValueError("Use 'atomic' or 'hybrid'")
    S=result.mol.intor('int1e_ovlp')
    c=(result.coefficients_ao[:,mo_index] if coefficients_ao is None
       else np.asarray(coefficients_ao,dtype=float))
    G=B.T@S@B
    rhs=B.T@S@c
    a=np.linalg.solve(G,rhs)
    projected=B@a
    projection_norm=float(a@G@a)
    es,U=np.linalg.eigh(G)
    if es.min()<=0:
        raise RuntimeError('Non-positive Gram matrix')
    lowdin_sqrt=(U*np.sqrt(es))@U.T
    w=lowdin_sqrt@a
    weights=w*w
    if projection_norm>0:
        weights/=projection_norm
    return dict(labels=list(labels), coefficients=a,
                weights=weights, projected_ao=projected,
                probability_represented=float(np.clip(projection_norm,0,1)),
                residual_norm=float(np.sqrt(max(0,1-projection_norm))),
                components_ao=B*a[None,:], gram=G)



def localize_occupied_standard(result: RHFResult, method='boys'):
    """Standard energy-invariant occupied valence localization: Boys or PM.

    Orthonormal valence orbitals stay in the same occupied subspace,
    preserving the HF density, total energy and frozen-core separation.
    """
    from pyscf import lo
    occ = result.coefficients_ao[:, result.valence_spatial_indices]
    if method == 'boys':
        return lo.Boys(result.mol, occ).kernel()
    if method in ('pipek', 'pm'):
        return lo.PM(result.mol, occ).kernel()
    raise ValueError("method must be 'boys' or 'pipek'")


def save_result(result:RHFResult, path:str|Path):
    """Portable numerical output; save metadata + occupied valence spin MOs."""
    valence_spins=result.valence_spin_orbitals()
    np.savez_compressed(path, mode=result.mode,
                        symbols=np.asarray([result.mol.atom_symbol(i) for i in range(result.mol.natm)]),
                        xyz_angstrom=result.xyz_angstrom, basis=str(result.mol.basis),
                        energy_hartree=result.energy_hartree,
                        orbital_energies=result.orbital_energies,
                        coefficients_ao=result.coefficients_ao,
                        occupations=result.occupations if result.occupations is not None
                            else result.mf.mo_occ,
                        frozen_core=result.frozen_core,
                        local_basis=result.local_basis.B,
                        local_labels=np.asarray(result.local_basis.labels),
                        valence_indices=np.asarray(result.valence_spatial_indices),
                        valence_spin_coefficients_ao=np.column_stack(
                            [item['coefficients_ao'] for item in valence_spins]),
                        valence_spin_labels=np.asarray([item['spin'] for item in valence_spins]),
                        valence_spin_spatial_indices=np.asarray([item['mo_index'] for item in valence_spins]),
                        valence_spin_energies=np.asarray([item['energy_hartree'] for item in valence_spins]))



def vsepr_single_point(seed: MolecularSeed, library: AtomicLibrary,
                       charge: int = 0, *, reduced: bool = False,
                       frozen_core: bool = False) -> RHFResult:
    """An electronic RHF energy at the *fixed*, VSEPR-seeded coordinates.

    This is NOT the VSEPR steric force-field score. It is a true RHF
    single-point calculation at the VSEPR initial geometry, suitable for
    direct energy comparison with the same basis after geometry optimization.
    """
    xyz = align_molecular_axes(seed.xyz_angstrom, seed.rdkit_mol)
    if frozen_core:
        mol,mf,local,energy,coeff,eps,occ = _frozen_core_at_geometry(
            seed, xyz, library.basis, charge, library, reduced=reduced)
    elif reduced:
        mol,mf,local = _reduced_at_geometry(seed, xyz, library.basis, charge, library)
        energy=float(mf.e_tot)
        coeff=local.B@mf.mo_coeff
        eps=mf.mo_energy.copy()
        occ=mf.mo_occ.copy()
    else:
        mol=molecule_from_xyz(seed.symbols, xyz, library.basis, charge)
        mf=scf.RHF(mol)
        mf.conv_tol=1e-9; mf.max_cycle=120; mf.verbose=0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('VSEPR full-basis single-point RHF did not converge')
        local=build_local_basis(mol, xyz, seed.rdkit_mol, library)
        energy=float(mf.e_tot)
        coeff=mf.mo_coeff.copy()
        eps=mf.mo_energy.copy()
        occ=mf.mo_occ.copy()
    return RHFResult('reduced' if reduced else 'full', energy,xyz,mol,mf,
                     local,coeff,eps,bool(mf.converged),True,
                     'VSEPR-seeded geometry fixed (no geometry optimization)',
                     core_count_from_rdkit(seed.rdkit_mol), [energy],
                     frozen_core=frozen_core, occupations=occ)


def localize_intrinsic_bond_orbitals(result: RHFResult) -> np.ndarray:
    """IBO localization, projected back to an EXACT valence occupied rotation.

    PySCF IBOs are used as a localization *target*. An orthogonal Procrustes
    alignment makes the reported functions strictly occupy the original
    RHF occupied valence subspace, so the density/total energy stay unchanged.
    """
    from pyscf import lo
    C=result.coefficients_ao[:,result.valence_spatial_indices]
    if C.shape[1] < 2:
        return C.copy()
    S=result.mol.intor('int1e_ovlp')
    all_occ=result.coefficients_ao[:,:result.nocc]
    iaos=lo.iao.iao(result.mol, all_occ)
    iaos=lo.vec_lowdin(iaos,S)
    approx=lo.ibo.ibo(result.mol,C,iaos=iaos)
    W=C.T@S@approx
    left,_,right=np.linalg.svd(W,full_matrices=False)
    Q=left@right
    result_coeff=C@Q
    if not np.allclose(result_coeff.T@S@result_coeff,np.eye(C.shape[1]),atol=1e-6):
        raise RuntimeError('IBO rotation failed to preserve occupied orthogonality')
    return result_coeff
