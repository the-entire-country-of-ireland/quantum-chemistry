"""Spin-adapted Heitler--London and variational GVB perfect pairing.

A *genuine simultaneous* perfect-pairing wavefunction is used:

    A[ core(1)^2 ... core(n)^2  prod_i (u_i v_i + v_i u_i) chi_s(i) ].

The optimized orbital pairs are mutually strongly orthogonal.  For every pair
its geminal is expressed in its natural orbital basis as
    cos(theta_i) |p_i^2> - sin(theta_i) |q_i^2>, 0 <= theta <= pi/4.
The sign restriction is important: it is the real spin-coupled GVB ansatz,
not unrestricted CASCI/DOCI with arbitrary pair CI coefficients.

All molecular integrals (including e-e repulsion and nuclear repulsion) are
computed from the same PySCF Hamiltonian as the parent RHF calculation.
The inactive occupied orbitals and the selected active space are fixed; optional
active-space orbital rotations vary the subspaces between pairs. This is NOT a
fully self-consistent all-space GVB-SCF optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass
class PairReference:
    """One localized occupied RHF orbital and two atom-specific trial pieces."""
    index: int
    label: str
    occupied_ao: np.ndarray
    atom_left_ao: np.ndarray
    atom_right_ao: np.ndarray
    atoms: tuple[str, str]


@dataclass
class PairSpace:
    """Independent 2D pair subspaces embedded in one orthonormal AO frame."""
    orbitals_ao: np.ndarray      # columns g1,h1,g2,h2,..., all S-orthonormal
    inactive_ao: np.ndarray
    pair_names: tuple[str, ...]
    atom_names: tuple[tuple[str, str], ...]
    hl_matrices: np.ndarray      # normalized real symmetric geminal matrices (m,2,2)
    hl_site_ao: np.ndarray       # atom-piece projections (m,2,nao)
    site_fidelities: np.ndarray  # (m,2), within-pair projection fractions
    partner_fidelities: np.ndarray


@dataclass
class PairHamiltonian:
    """Fixed-active-space many-electron Hamiltonian with inactive closed shells."""
    h: np.ndarray               # one-body h + exact Coulomb/exchange from inactive core
    eri: np.ndarray             # chemist's (ij|kl), not antisymmetrized integrals
    constant: float             # inactive-electron energy + E_nuclear
    rhf_energy: float


@dataclass
class PairWavefunction:
    label: str
    energy: float
    frame: np.ndarray           # natural orbital columns in active pair space
    theta: np.ndarray           # signed geminal coefficients c=cos, d=-sin
    occupations: np.ndarray     # (m,2), 2c^2, 2d^2
    converged: bool
    message: str
    iterations: int


def _metric_norm(vector: np.ndarray, overlap: np.ndarray) -> float:
    return float(np.asarray(vector, float) @ overlap @ np.asarray(vector, float))


def _normalized(vector: np.ndarray, overlap: np.ndarray, *, what: str) -> np.ndarray:
    n = _metric_norm(vector, overlap)
    if n < 1e-12:
        raise ValueError(f'{what} has negligible norm after orthogonalization')
    return np.asarray(vector, float) / np.sqrt(n)


def _subtract(vector: np.ndarray, columns: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    if columns.size == 0:
        return np.asarray(vector, float).copy()
    return vector - columns @ (columns.T @ overlap @ vector)


def _site_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized symmetric covalent pair from two REAL normalized site vectors."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if np.dot(a, b) < 0:
        b = -b
    if abs(float(np.dot(a,b))) > 0.9995:
        raise ValueError('Atomic site projections are linearly dependent in this pair space')
    mat = np.outer(a, b) + np.outer(b, a)
    mat /= np.linalg.norm(mat)
    return mat


def construct_pair_space(result, localized_valence: np.ndarray,
                         references: Sequence[PairReference]) -> PairSpace:
    """Build distinct occupied/antibonding subspaces for ALL selected pairs.

    Selected occupied orbitals are exact localized RHF occupied orbitals.  Their
    antibonding partners are drawn exclusively from the original virtual space,
    using the difference between atom-local orbital ingredients as a seed.
    Gram--Schmidt avoids double-counting the same virtual orbital in two pairs.
    HL atomic pieces are then projected into EACH strongly orthogonal pair
    subspace, making the approximation and projection losses explicit.
    """
    if result.mode != 'full' or result.frozen_core:
        raise ValueError('HL/GVB requires an all-electron full-Gaussian RHF result (no frozen core).')
    if not references:
        raise ValueError('Select at least one localized bond.')
    S = np.asarray(result.mol.intor('int1e_ovlp'), float)
    C = np.asarray(result.coefficients_ao, float)
    occupied = np.column_stack([C[:, :result.core_spatial_count],
                                np.asarray(localized_valence, float)])
    nocc = int(result.nocc)
    if occupied.shape[1] != nocc or not np.allclose(occupied.T @ S @ occupied, np.eye(nocc), atol=4e-5):
        raise ValueError('Localized RHF occupied frame is not S-orthonormal.')
    selected = [int(ref.index) for ref in references]
    if len(set(selected)) != len(selected) or any(k < 0 or k >= localized_valence.shape[1] for k in selected):
        raise ValueError('Selected localized orbital indices are invalid or duplicated.')
    if C.shape[1] - nocc < len(references):
        raise ValueError('Not enough virtual orbitals for one antibonding partner per bond; increase basis size.')

    inactive_indices = [i for i in range(nocc) if i not in
                        {result.core_spatial_count + x for x in selected}]
    inactive = occupied[:, inactive_indices]
    partners = []
    for ref in references:
        left = np.asarray(ref.atom_left_ao, float)
        right = np.asarray(ref.atom_right_ao, float)
        # Left and right contributions are signed pieces of the same occupied MO.
        trial = left-right
        virtual = _subtract(trial, occupied, S)
        if partners:
            virtual = _subtract(virtual, np.column_stack(partners), S)
        trial_norm = _metric_norm(virtual, S)
        if trial_norm < 1e-8:
            # A poor atomic projection is NOT silently replaced by a random virtual.
            raise ValueError(f'No independent antibonding virtual for {ref.label}. '
                             'Try a larger basis or a different localized orbital.')
        virtual = _normalized(virtual, S, what=f'{ref.label} virtual partner')
        partners.append(virtual)

    active = np.column_stack([v for ref,partner in zip(references,partners)
                              for v in (np.asarray(ref.occupied_ao,float),partner)])
    if not np.allclose(active.T @ S @ active, np.eye(active.shape[1]), atol=4e-5):
        raise ValueError('Active pair orbitals are not orthonormal.')
    if inactive.size and not np.allclose(inactive.T@S@active, 0, atol=4e-5):
        raise ValueError('Inactive and active orbitals are not orthogonal.')

    matrices, site_ao, fidelities, partner_fidelity = [], [], [], []
    for i, ref in enumerate(references):
        pair_ao = active[:, 2*i:2*i+2]
        a = np.asarray(ref.atom_left_ao, float)
        b = np.asarray(ref.atom_right_ao, float)
        site_vecs = []
        fracs = []
        for site in (a,b):
            norm = _metric_norm(site,S)
            if norm < 1e-10:
                raise ValueError(f'Missing atom-local component for {ref.label}.')
            v = pair_ao.T @ S @ site
            frac = float(np.dot(v,v) / norm)
            fracs.append(float(np.clip(frac,0.,1.)))
            site_vecs.append(v/np.linalg.norm(v))
        a_pair, b_pair = site_vecs
        if np.dot(a_pair,b_pair)<0: b_pair = -b_pair
        matrices.append(_site_matrix(a_pair,b_pair))
        site_ao.append(np.vstack([pair_ao @ a_pair,pair_ao @ b_pair]))
        fidelities.append(fracs)
        initial_virtual = _subtract(np.asarray(ref.atom_left_ao)-np.asarray(ref.atom_right_ao), occupied,S)
        denom = _metric_norm(initial_virtual,S)
        partner_fidelity.append(float(np.clip(_metric_norm(pair_ao[:,1] * float(pair_ao[:,1]@S@initial_virtual),S)/denom,0,1)) if denom>1e-12 else 0.)
    return PairSpace(active, inactive, tuple(r.label for r in references),
                     tuple(r.atoms for r in references),
                     np.asarray(matrices), np.asarray(site_ao),
                     np.asarray(fidelities), np.asarray(partner_fidelity))


def integral_hamiltonian(result, space: PairSpace) -> PairHamiltonian:
    """Project actual interacting molecular Hamiltonian into the PP active space."""
    from pyscf import ao2mo
    mol = result.mol
    mf = result.mf
    active, inactive = space.orbitals_ao, space.inactive_ao
    hcore = np.asarray(mf.get_hcore(), float)
    dm = 2. * (inactive @ inactive.T)
    vcore = np.asarray(mf.get_veff(mol, dm), float) if inactive.shape[1] else np.zeros_like(hcore)
    constant = float(mol.energy_nuc()+np.einsum('ij,ji->',dm,hcore)
                     +0.5*np.einsum('ij,ji->',dm,vcore))
    h = active.T @ (hcore + vcore) @ active
    n = active.shape[1]
    eri = ao2mo.kernel(mol, active, compact=False).reshape(n,n,n,n)
    ham = PairHamiltonian(h, eri, constant, float(result.energy_hartree))
    check = pp_energy(ham, np.eye(n), np.zeros(n//2))
    if not np.isclose(check, result.energy_hartree, rtol=0, atol=2e-4):
        raise ValueError(f'Active-space RHF consistency failed: reconstructed {check:.9f} Ha '
                         f'versus original {result.energy_hartree:.9f} Ha.')
    return ham


def _rotate_eri(eri: np.ndarray, frame: np.ndarray) -> np.ndarray:
    # Chemist's integral transformation (ij|kl): all four orbital indices rotate.
    return np.einsum('abcd,ai,bj,ck,dl->ijkl',eri,frame,frame,frame,frame,optimize=True)


def pp_energy(ham: PairHamiltonian, frame: np.ndarray, theta: np.ndarray) -> float:
    """Exact <H> for strongly orthogonal *real* perfect-pairing geminals.

    Assumes an inactive RHF closed shell and one singlet geminal per pair.
    This expression includes BOTH Coulomb and exchange between pairs, as well
    as the off-diagonal pair-excitation matrix element (pq|pq).
    """
    n = ham.h.shape[0]
    theta=np.asarray(theta,float)
    if n%2 or len(theta)!=n//2:
        raise ValueError('Expected two natural orbitals for each singlet pair.')
    frame=np.asarray(frame,float)
    if frame.shape!=(n,n) or not np.allclose(frame.T@frame,np.eye(n),atol=3e-6):
        raise ValueError('Pair orbital frame must be orthonormal.')
    h = frame.T@ham.h@frame
    eri = _rotate_eri(ham.eri,frame)
    c=np.cos(theta)
    d=-np.sin(theta)
    weights=np.column_stack([c*c,d*d]).ravel()
    energy = float(ham.constant)
    for i in range(n):
        energy += weights[i]*(2*h[i,i]+eri[i,i,i,i])
    for k in range(n//2):
        p,q=2*k,2*k+1
        energy += 2*c[k]*d[k]*eri[p,q,p,q]
    for i in range(n//2):
        for j in range(i+1,n//2):
            for p in (2*i,2*i+1):
                for q in (2*j,2*j+1):
                    energy += weights[p]*weights[q]*(4*eri[p,p,q,q]-2*eri[p,q,p,q])
    return float(energy)


def _hl_natural_frame(space: PairSpace) -> tuple[np.ndarray,np.ndarray]:
    """Diagonalize each covalent site geminal; eigenvalues have opposite signs."""
    n = space.orbitals_ao.shape[1]
    mat = np.eye(n)
    angles=[]
    for k,M in enumerate(space.hl_matrices):
        eig,U=np.linalg.eigh(M)
        ix=np.argsort(eig)[::-1]
        eig=eig[ix]
        U=U[:,ix]
        if eig[0]<=0 or eig[1]>=0:
            raise ValueError('The HL pair geminal is not a real singlet-coupled two-orbital product.')
        # Our phase convention ensures the positive eigenvalue has >= magnitude.
        if eig[0] + eig[1] < -1e-9:
            raise ValueError('HL site orbital overlap phase could not be aligned.')
        mat[2*k:2*k+2,2*k:2*k+2]=U
        angles.append(float(np.arctan2(-eig[1],eig[0])))
    return mat,np.asarray(angles)


def make_rhf_wavefunction(ham: PairHamiltonian) -> PairWavefunction:
    n=ham.h.shape[0]
    return _wave('RHF',ham,np.eye(n),np.zeros(n//2),True,'RHF reference',0)


def make_hl_wavefunction(ham: PairHamiltonian, space: PairSpace) -> PairWavefunction:
    frame,angles=_hl_natural_frame(space)
    return _wave('Projected HL-VB',ham,frame,angles,True,
                 'Fixed atom-local trial orbitals projected into strongly orthogonal pair subspaces',0)


def _wave(name,ham,frame,theta,converged,message,iters):
    return PairWavefunction(name,pp_energy(ham,frame,theta),np.array(frame),np.array(theta),
                            np.column_stack([2*np.cos(theta)**2,2*np.sin(theta)**2]),
                            bool(converged),str(message),int(iters))


def _compose_frame(theta_phi, n, rotate_subspaces):
    m=n//2
    phi=theta_phi[m:2*m]
    mat=np.eye(n)
    if rotate_subspaces:
        skew=np.zeros((n,n))
        ix=2*m
        for p,q in combinations(range(n),2):
            if p//2 != q//2:
                skew[p,q]=theta_phi[ix];skew[q,p]=-theta_phi[ix]
                ix+=1
        mat=expm(skew)
    for i in range(m):
        angle=phi[i]
        rotation=np.array([[np.cos(angle),-np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        mat[:,2*i:2*i+2] = mat[:,2*i:2*i+2] @ rotation
    return mat


def optimize_gvb_pp(ham: PairHamiltonian, space: PairSpace,
                    *, rotate_subspaces: bool = False, maxiter: int = 100) -> PairWavefunction:
    """Optimize one spin-coupled geminal per pair, simultaneously.

    Every active orbital is free to rotate within its initial local pair block.
    Optional `rotate_subspaces` also optimizes interpair active orbital mixing.
    Inactive and external virtual orbital subspaces remain fixed. Both are
    explicitly advertised limitations relative to full AO-space GVB-SCF.
    """
    n=ham.h.shape[0]
    m=n//2
    hl_frame,hl_theta=_hl_natural_frame(space)
    hl_phi=np.array([np.arctan2(hl_frame[2*i+1,2*i],hl_frame[2*i,2*i]) for i in range(m)])
    nrot=sum(i//2!=j//2 for i,j in combinations(range(n),2)) if rotate_subspaces else 0
    bounds=[(0.,np.pi/4.)]*m+[(-np.pi,np.pi)]*m+[(-np.pi/2,np.pi/2)]*nrot
    starts=[np.zeros(2*m+nrot),np.r_[hl_theta,hl_phi,np.zeros(nrot)]]
    # An independent near-RHF start prevents exactly-zero theta/phi from
    # suppressing the finite-difference direction at a boundary.
    starts.append(np.r_[np.full(m,0.08),np.zeros(m+nrot)])
    best=_wave('GVB-PP (fixed active space)',ham,np.eye(n),np.zeros(m),True,'RHF baseline',0)
    best_fun=best.energy
    success=False
    status=[]
    iterations=0
    def objective(x):
        frame=_compose_frame(x,n,rotate_subspaces)
        return pp_energy(ham,frame,x[:m])
    for initial in starts:
        # Preliminary minimization from both RHF and HL is important where
        # distinct local minima exist for stretched/polarized bonds.
        opt=minimize(objective, initial, method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':int(maxiter),'ftol':2e-10,'gtol':2e-6,'maxls':22})
        iterations+=int(opt.nit)
        status.append(f'{opt.message} (nit={opt.nit})')
        if float(opt.fun)<best_fun+1e-10:
            if float(opt.fun)<best_fun-1e-9 or not success:
                best_fun=float(opt.fun)
                frame=_compose_frame(opt.x,n,rotate_subspaces)
                best=_wave('GVB-PP (active-space variational)',ham,frame,opt.x[:m],
                           bool(opt.success),str(opt.message),int(opt.nit))
        success=success or bool(opt.success)
    # If every optimizer failed, the RHF baseline is still a valid state. Do
    # not claim convergence or an optimum that was not established.
    best.converged=bool(best.converged and success)
    best.message='; '.join(status)
    best.iterations=iterations
    return best


def bond_pair_orbitals(space: PairSpace, wave: PairWavefunction, bond: int) -> tuple[np.ndarray,np.ndarray,float]:
    """Real localized v,w whose symmetrized product is the optimized geminal."""
    if bond<0 or bond>=len(wave.theta):
        raise IndexError('Unknown bond index')
    theta=float(wave.theta[bond])
    p=space.orbitals_ao @ wave.frame[:,2*bond]
    q=space.orbitals_ao @ wave.frame[:,2*bond+1]
    beta=float(np.arctan(np.sqrt(np.tan(theta)))) if theta>1e-12 else 0.
    v=np.cos(beta)*p+np.sin(beta)*q
    w=np.cos(beta)*p-np.sin(beta)*q
    return v,w,float(np.cos(2*beta))


def pair_slice(mol, space: PairSpace, wave: PairWavefunction, bond: int,
               center_a: np.ndarray, center_b: np.ndarray, points: int=87, pad_bohr: float=1.8):
    """Actual 3D AO geminal evaluated on a one-dimensional bond-axis slice.

    Returns a 2D slice of |Psi(r1,r2)|^2. This is NOT a fully integrated 3D
    conditional probability; its integral over the displayed axis is arbitrary.
    """
    a=np.asarray(center_a,float)
    b=np.asarray(center_b,float)
    vec=b-a
    R=float(np.linalg.norm(vec))
    if R<1e-8:raise ValueError('Coincident atom centers')
    unit=vec/R
    x=np.linspace(-pad_bohr,R+pad_bohr,int(points))
    xyz=a[None,:]+x[:,None]*unit[None,:]
    ao=mol.eval_gto('GTOval_sph',xyz)
    frame=space.orbitals_ao@wave.frame[:,2*bond:2*bond+2]
    orbital=ao@frame
    c=np.cos(wave.theta[bond]);d=-np.sin(wave.theta[bond])
    psi=c*np.outer(orbital[:,0],orbital[:,0])+d*np.outer(orbital[:,1],orbital[:,1])
    density=np.abs(psi)**2
    return x,density,orbital
