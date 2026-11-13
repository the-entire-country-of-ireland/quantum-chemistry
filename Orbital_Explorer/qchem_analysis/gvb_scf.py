"""All-space orbital-optimized, spin-adapted GVB perfect-pairing SCF.

The orbital variational space is the *complete finite molecular AO basis*, not
merely the initially chosen two-dimensional pair subspaces.  Every occupied and
virtual orbital may relax: active/active, active/inactive, active/virtual and
inactive/virtual rotations are included. Rotations within solely inactive or
solely unoccupied orbitals are redundant and removed.

The wavefunction nevertheless retains the PERFECT-PAIRING approximation:
selected geminals occupy two *strongly orthogonal* natural orbitals with
amplitudes cos(theta), -sin(theta); unselected electron pairs remain doubly
occupied (though their orbitals are fully optimized). This is not general VBSCF,
spin recoupling, UHF, or unrestricted spin symmetry breaking. All calculations
are variational for the physical Coulomb Hamiltonian in the chosen Gaussian
basis. Nuclear geometry is held fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from .perfect_pairing import PairSpace, PairWavefunction


@dataclass
class GVBSCFProblem:
    """Full orthonormal molecular orbital frame and unprojected ERIs.

    Input frame order is [unpaired RHF occupied/inactive | pair 1 (g,v) |
    ... | pair m (g,v) | remaining virtual]. The reference frame C0 is
    S-orthonormal in the original AO basis.
    """
    orbitals_ao: np.ndarray
    h: np.ndarray                         # spatial orbital hcore, full n*n
    eri: np.ndarray                       # chemist's (pq|rs), full n^4
    nuclear_energy: float
    rhf_energy: float
    inactive_count: int
    pair_names: tuple[str, ...]

    @property
    def pair_offset(self) -> int:
        return self.inactive_count

    @property
    def pair_count(self) -> int:
        return len(self.pair_names)

    @property
    def orbital_count(self) -> int:
        return len(self.h)

    @property
    def inactive_ao(self) -> np.ndarray:
        return self.orbitals_ao[:, :self.inactive_count]


def construct_gvb_scf_problem(result, space: PairSpace, *, max_orbitals: int = 44) -> GVBSCFProblem:
    """Construct a full-space objective from the physical molecular integrals.

    No effective frozen-core Hamiltonian is used; the original one- and
    two-electron integrals include interactions and relaxation of all electrons.
    """
    from pyscf import ao2mo

    if getattr(result,'mode',None) != 'full' or getattr(result,'frozen_core',False):
        raise ValueError('All-space GVB-SCF requires a full all-electron RHF calculation.')
    C=np.asarray(result.coefficients_ao,float)
    n=C.shape[1]
    if n>max_orbitals:
        raise ValueError(f'{n} orbitals would require a large dense four-index integral tensor. '
                         f'The interactive all-space solver currently supports at most {max_orbitals}.')
    S=np.asarray(result.mol.intor('int1e_ovlp'),float)
    inactive=np.asarray(space.inactive_ao,float)
    active=np.asarray(space.orbitals_ao,float)
    first=np.column_stack([inactive,active])
    k=first.shape[1]
    if k>n or not np.allclose(first.T@S@first,np.eye(k),atol=5e-6):
        raise ValueError('Starting GVB and inactive orbitals are not mutually S-orthonormal.')
    # Columns of C are complete in the same finite AO space. The nullspace
    # adds every external orbital left out of the initial GVB-pair construction.
    comp=null_space(first.T@S@C, rcond=1e-11)
    if comp.shape[1] != n-k:
        raise ValueError('Failed to complete GVB orbitals into the full molecular orbital space.')
    full=np.column_stack([first,C@comp])
    if not np.allclose(full.T@S@full,np.eye(n),atol=1e-6):
        raise ValueError('Full orbital frame lost S orthogonality.')
    hcore=np.asarray(result.mf.get_hcore(),float)
    h=full.T@hcore@full
    eri=np.asarray(ao2mo.kernel(result.mol,full,compact=False),float).reshape(n,n,n,n)
    prob=GVBSCFProblem(full,h,eri,float(result.mol.energy_nuc()),
                       float(result.energy_hartree),inactive.shape[1],tuple(space.pair_names))
    calculated=gvb_scf_energy_gradient(prob,np.eye(n),np.zeros(prob.pair_count),gradient=False)[0]
    if abs(calculated-prob.rhf_energy)>2e-4:
        raise ValueError(f'RHF reference energy does not reconstruct in full GVB space: '
                         f'{calculated:.9f} versus {prob.rhf_energy:.9f} Eh')
    return prob


def _population_weights(problem: GVBSCFProblem, theta: np.ndarray):
    n=problem.orbital_count
    m=problem.pair_count
    offset=problem.pair_offset
    if theta.shape!=(m,):
        raise ValueError('Incorrect perfect-pairing angle dimension.')
    w=np.zeros(n)
    w[:offset]=1.0
    for i,t in enumerate(theta):
        w[offset+2*i]=np.cos(t)**2
        w[offset+2*i+1]=np.sin(t)**2
    return w


def gvb_scf_energy_gradient(problem: GVBSCFProblem, frame: np.ndarray,
                            theta: np.ndarray, *, gradient: bool=True):
    """Return E and the exact Cartesian orbital/angle gradient.

    dE/d(frame[:,i]) is unconstrained; orthonormality is imposed by a Cayley
    parametrization in the outer optimizer. Contracted J/K matrices are built
    directly from full Coulomb integrals, avoiding repeated four-index rotations
    and providing analytic derivatives for efficient SCF iteration.
    """
    Q=np.asarray(frame,float)
    theta=np.asarray(theta,float)
    n=problem.orbital_count
    if Q.shape!=(n,n) or not np.allclose(Q.T@Q,np.eye(n),atol=1e-5):
        raise ValueError('Full GVB orbital rotation must be orthogonal.')
    weights=_population_weights(problem,theta)
    offset=problem.pair_offset
    used=list(range(offset+2*problem.pair_count))
    # Inactive doubly occupied orbitals are distinct singlet electron pairs;
    # two natural orbitals in one correlated geminal may never be simultaneous.
    def group(i):
        return i if i<offset else offset+(i-offset)//2
    Js={};Ks={}
    for i in used:
        v=Q[:,i]
        Js[i]=np.einsum('abcd,c,d->ab',problem.eri,v,v,optimize=True)
        Ks[i]=np.einsum('acbd,c,d->ab',problem.eri,v,v,optimize=True)
    energy=problem.nuclear_energy
    G=np.zeros((n,n)) if gradient else None
    partial=np.zeros(n)
    for i in used:
        wi=weights[i]
        v=Q[:,i]
        hi=float(v@problem.h@v)
        selfJ=float(v@Js[i]@v)
        energy+=wi*(2*hi+selfJ)
        partial[i]=2*hi+selfJ
        if gradient and wi:
            G[:,i]+=4*wi*(problem.h@v+Js[i]@v)
    for idx,i in enumerate(used):
        vi=Q[:,i]
        for j in used[idx+1:]:
            if group(i)==group(j):
                continue
            vj=Q[:,j]
            jij=float(vi@Js[j]@vi)
            kij=float(vi@Ks[j]@vi)
            fac=4*jij-2*kij
            energy+=weights[i]*weights[j]*fac
            partial[i]+=weights[j]*fac
            partial[j]+=weights[i]*fac
            if gradient:
                G[:,i]+=weights[i]*weights[j]*(8*Js[j]@vi-4*Ks[j]@vi)
                G[:,j]+=weights[i]*weights[j]*(8*Js[i]@vj-4*Ks[i]@vj)
    angle_gradient=np.zeros(problem.pair_count)
    for pair,t in enumerate(theta):
        p,q=offset+2*pair,offset+2*pair+1
        cp=np.cos(t);dp=-np.sin(t)
        integral=float(Q[:,p]@Ks[q]@Q[:,p])
        energy+=2*cp*dp*integral
        angle_gradient[pair]=np.sin(2*t)*(partial[q]-partial[p])-2*np.cos(2*t)*integral
        if gradient:
            G[:,p]+=4*cp*dp*(Ks[q]@Q[:,p])
            G[:,q]+=4*cp*dp*(Ks[p]@Q[:,q])
    return float(energy),G,angle_gradient


def _allowed_rotations(problem: GVBSCFProblem):
    """All nonredundant rotations for a chosen perfect-pair partition."""
    n=problem.orbital_count
    inactive=problem.inactive_count
    end=inactive+2*problem.pair_count
    # Omit only rotations within the fully occupied inactive subspace and
    # within the entirely empty external virtual subspace.
    return tuple((i,j) for i,j in combinations(range(n),2)
                 if not ((i<inactive and j<inactive) or (i>=end and j>=end)))


def _cayley_frame_and_grad(Q0, params, pairs, G=None):
    """Cayley rotation with exact derivative of energy versus skew entries.

    For K^T=-K, Q=(I-K/2)^(-1)(I+K/2) and
    dQ=1/2 (I-K/2)^(-1) dK (I+Q).
    """
    n=Q0.shape[0]
    K=np.zeros((n,n))
    for val,(i,j) in zip(params,pairs):
        K[i,j]=val
        K[j,i]=-val
    A=np.eye(n)-.5*K
    rotated=np.linalg.solve(A,np.eye(n)+.5*K)
    Q=Q0@rotated
    if G is None:
        return Q,None
    G0=Q0.T@G
    M=.5*np.linalg.solve(A.T,G0)@(np.eye(n)+rotated).T
    return Q,np.array([M[i,j]-M[j,i] for i,j in pairs])


def optimize_full_gvb_scf(problem: GVBSCFProblem, *,
                          initial_pair_wave: PairWavefunction|None=None,
                          maxiter:int=160, gtol:float=1.0e-5) -> PairWavefunction:
    """Unrestricted MO-space orbital relaxation of spin-pure GVB-PP geminals.

    Local variational minimization from RHF, a weakly correlated seed, and an
    optional optimized fixed-active-space starting point. The solution's
    `.frame` is expressed in the full `problem.orbitals_ao` basis. `converged`
    describes the winning optimizer, not global-optimum certification.
    """
    m=problem.pair_count
    n=problem.orbital_count
    off=problem.pair_offset
    if m<1:
        raise ValueError('Select at least one correlated electron pair.')
    pairs=_allowed_rotations(problem)
    nparam=len(pairs)
    def eval_fg(x):
        angles=x[:m]
        Q,_=_cayley_frame_and_grad(np.eye(n),x[m:],pairs)
        e,g,theta_grad=gvb_scf_energy_gradient(problem,Q,angles)
        _,kgrad=_cayley_frame_and_grad(np.eye(n),x[m:],pairs,g)
        return e,np.r_[theta_grad,kgrad]
    bounds=[(0.,np.pi/4)]*m+[(None,None)]*nparam
    starts=[np.zeros(m+nparam), np.r_[np.full(m,.09),np.zeros(nparam)]]
    if initial_pair_wave is not None and len(initial_pair_wave.theta)==m:
        x=np.zeros(m+nparam)
        x[:m]=initial_pair_wave.theta
        indices={p:k for k,p in enumerate(pairs)}
        for i in range(m):
            angle=np.arctan2(initial_pair_wave.frame[2*i+1,2*i],
                             initial_pair_wave.frame[2*i,2*i])
            # A Cayley K with K[p,q]=-2 tan(phi/2) generates a +phi rotation.
            key=(off+2*i,off+2*i+1)
            if key in indices:
                x[m+indices[key]]=-2*np.tan(.5*angle)
        starts.append(x)
    baseline=gvb_scf_energy_gradient(problem,np.eye(n),np.zeros(m),gradient=False)[0]
    best_energy=baseline
    best_x=np.zeros(m+nparam)
    best_opt=None
    history=[]
    totaliter=0
    for seed in starts:
        opt=minimize(eval_fg,seed,method='L-BFGS-B',jac=True,bounds=bounds,
                     options={'maxiter':int(maxiter),'ftol':2e-11,'gtol':gtol,
                              'maxls':35,'maxcor':14})
        totaliter+=int(opt.nit)
        history.append(f'{str(opt.message)} ({opt.nit} steps)')
        if np.isfinite(opt.fun) and opt.fun<best_energy-1.e-9:
            best_energy=float(opt.fun)
            best_x=opt.x.copy()
            best_opt=opt
    Q,_=_cayley_frame_and_grad(np.eye(n),best_x[m:],pairs)
    final_e=gvb_scf_energy_gradient(problem,Q,best_x[:m],gradient=False)[0]
    if final_e>baseline+1e-8:
        raise RuntimeError('GVB-SCF optimizer violated the variational RHF baseline.')
    # If no meaningful improvement was obtained, RHF is still valid, but not
    # evidence that all-space optimization converged to a stationary solution.
    converged=bool(best_opt is not None and best_opt.success)
    return PairWavefunction('Full-orbital GVB-PP SCF',float(final_e),Q,
                            np.asarray(best_x[:m]),
                            np.column_stack([2*np.cos(best_x[:m])**2,
                                             2*np.sin(best_x[:m])**2]),
                            converged,'; '.join(history),totaliter)


def full_pair_orbitals(problem: GVBSCFProblem, wave: PairWavefunction,
                       pair_index:int):
    """Two overlapping spatial orbitals of one optimized singlet geminal."""
    if pair_index<0 or pair_index>=problem.pair_count:
        raise IndexError('Unknown singlet pair index.')
    p=problem.pair_offset+2*pair_index
    theta=float(wave.theta[pair_index])
    v=problem.orbitals_ao @ wave.frame[:,p]
    w=problem.orbitals_ao @ wave.frame[:,p+1]
    # Geminal amplitudes cos(theta) >0, -sin(theta) <=0, converted into
    # symmetrized product of two nonorthogonal normalized spatial orbitals.
    beta=float(np.arctan(np.sqrt(np.tan(theta)))) if theta>1e-12 else 0.
    a=np.cos(beta)*v+np.sin(beta)*w
    b=np.cos(beta)*v-np.sin(beta)*w
    return a,b,float(np.cos(2*beta))
