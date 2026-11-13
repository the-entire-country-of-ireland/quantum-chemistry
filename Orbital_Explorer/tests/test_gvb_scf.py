"""Analytical-gradient and orbital-relaxation tests without PySCF."""
import numpy as np
from scipy.linalg import expm

from qchem_analysis.gvb_scf import (
    GVBSCFProblem, _allowed_rotations, _cayley_frame_and_grad,
    gvb_scf_energy_gradient, optimize_full_gvb_scf, full_pair_orbitals,
)


def physical_test_problem(n=5, inactive=1, pairs=1):
    rng=np.random.default_rng(42)
    h=rng.normal(size=(n,n));h=(h+h.T)*.5
    factors=rng.normal(size=(n,n,7))
    # Positive Coulomb (ab|cd), symmetric in ab and cd, appropriate for gradient tests.
    factors=.5*(factors+factors.swapaxes(0,1))
    eri=np.einsum('abk,cdk->abcd',factors,factors)
    base=GVBSCFProblem(np.eye(n),h,eri,1.234,0.,inactive,tuple('pair'+str(i) for i in range(pairs)))
    base.rhf_energy=gvb_scf_energy_gradient(base,np.eye(n),np.zeros(pairs),gradient=False)[0]
    return base


def test_full_orbital_and_pair_gradients_numerically():
    p=physical_test_problem(6,2,1)
    rng=np.random.default_rng(123)
    k=rng.normal(size=(6,6));k=.08*(k-k.T)
    Q=expm(k)
    theta=np.array([.21])
    E,G,dtheta=gvb_scf_energy_gradient(p,Q,theta)
    eps=1e-6
    v1=gvb_scf_energy_gradient(p,Q,np.array([.21+eps]),gradient=False)[0]
    v0=gvb_scf_energy_gradient(p,Q,np.array([.21-eps]),gradient=False)[0]
    assert abs(dtheta[0]-(v1-v0)/(2*eps))<1e-6
    # Cartesian derivative: perturb the unconstrained Q without demanding orthonormality.
    # Instead, analytically project to the tangent of a right rotation at Q.
    for i,j in ((0,2),(1,5),(2,3),(3,4)):
        a=np.zeros((6,6));a[i,j]=1;a[j,i]=-1
        slope=float(np.sum(G*(Q@a)))
        vp=gvb_scf_energy_gradient(p,Q@expm(eps*a),theta,gradient=False)[0]
        vm=gvb_scf_energy_gradient(p,Q@expm(-eps*a),theta,gradient=False)[0]
        assert abs(slope-(vp-vm)/(2*eps))<2e-6,(i,j,slope,(vp-vm)/(2*eps))


def test_cayley_rotation_gradient_vs_finite_difference():
    p=physical_test_problem(6,2,1)
    pairs=_allowed_rotations(p)
    x=np.linspace(-.12,.11,len(pairs))
    Q,_=_cayley_frame_and_grad(np.eye(6),x,pairs)
    energy,G,_=gvb_scf_energy_gradient(p,Q,np.array([.17]))
    _,dx=_cayley_frame_and_grad(np.eye(6),x,pairs,G)
    eps=1e-6
    for idx in [0,len(pairs)//2,len(pairs)-1]:
        y=x.copy();y[idx]+=eps
        plus,_=_cayley_frame_and_grad(np.eye(6),y,pairs)
        y[idx]-=2*eps
        minus,_=_cayley_frame_and_grad(np.eye(6),y,pairs)
        ep=gvb_scf_energy_gradient(p,plus,np.array([.17]),gradient=False)[0]
        em=gvb_scf_energy_gradient(p,minus,np.array([.17]),gradient=False)[0]
        assert abs(dx[idx]-(ep-em)/(2*eps))<2e-6,(idx,dx[idx],(ep-em)/(2*eps))


def test_hubbard_dimer_full_gvb_recovers_exact_ground_energy():
    t=1.;u=4.
    A=np.array([[1,1],[1,-1]])/np.sqrt(2)
    eri=np.zeros((2,2,2,2))
    for i,j,k,l in np.ndindex(2,2,2,2):
        eri[i,j,k,l]=u*sum(A[s,i]*A[s,j]*A[s,k]*A[s,l] for s in range(2))
    p=GVBSCFProblem(np.eye(2),np.diag([-t,t]),eri,0.,-2+t*0+u/2,0,('H-H',))
    result=optimize_full_gvb_scf(p,maxiter=100)
    assert np.isclose(result.energy,(u-np.sqrt(u*u+16))/2,atol=2e-6)
    assert result.energy<p.rhf_energy
    v,w,overlap=full_pair_orbitals(p,result,0)
    assert np.isclose(np.linalg.norm(v),1)
    assert np.isclose(np.linalg.norm(w),1)
    assert np.isclose(v@w,overlap)


def test_inactive_and_external_orbitals_really_relax():
    # One occupied inactive/core, a selected pair, and one external virtual;
    # strong offdiagonal one-electron integrals encourage mixing beyond pair.
    h=np.diag([-5.,-1.,2.,.7])
    h[0,3]=h[3,0]=.35
    h[1,3]=h[3,1]=.45
    eri=np.zeros((4,4,4,4))
    p=GVBSCFProblem(np.eye(4),h,eri,0.,-12.,1,('A-B',))
    p.rhf_energy=gvb_scf_energy_gradient(p,np.eye(4),np.zeros(1),gradient=False)[0]
    opt=optimize_full_gvb_scf(p,maxiter=120)
    assert opt.energy < p.rhf_energy-.05,(opt.energy,p.rhf_energy)
    # Full four-orbital rotation includes inactive-to-virtual mixing.
    Q=opt.frame
    assert abs(Q[3,0])>.02
    assert np.allclose(Q.T@Q,np.eye(4),atol=1e-8)


def test_nonredundant_rotations_include_all_orbital_sectors():
    p=physical_test_problem(7,2,1)
    allowed=set(_allowed_rotations(p))
    assert (0,1) not in allowed  # inactive-inactive gauge
    assert (4,5) not in allowed  # virtual-virtual gauge
    assert (0,2) in allowed      # inactive-pair
    assert (0,5) in allowed      # inactive-virtual
    assert (2,5) in allowed      # pair-virtual
    assert (2,3) in allowed      # natural orbital pair rotation


def test_full_coulomb_energy_matches_exact_frozen_core_pair_formula():
    from qchem_analysis.perfect_pairing import PairHamiltonian,pp_energy
    p=physical_test_problem(n=5,inactive=1,pairs=2)
    core=0
    active=np.arange(1,5)
    h=p.h[np.ix_(active,active)].copy()
    for a,i in enumerate(active):
        for b,j in enumerate(active):
            h[a,b]+=2*p.eri[i,j,core,core]-p.eri[i,core,j,core]
    eri=p.eri[np.ix_(active,active,active,active)]
    constant=p.nuclear_energy+2*p.h[core,core]+p.eri[core,core,core,core]
    old=PairHamiltonian(h,eri,constant,0.)
    theta=np.array([.23,.31])
    old_e=pp_energy(old,np.eye(4),theta)
    new_e=gvb_scf_energy_gradient(p,np.eye(5),theta,gradient=False)[0]
    assert np.isclose(new_e,old_e,atol=1e-8), (old_e,new_e)


def test_two_pair_interpair_orbital_and_angle_derivatives():
    p=physical_test_problem(n=7,inactive=1,pairs=2)
    rng=np.random.default_rng(143)
    skew=rng.normal(size=(7,7));skew=.05*(skew-skew.T)
    Q=expm(skew)
    angles=np.array([.19,.27])
    e,G,grad=gvb_scf_energy_gradient(p,Q,angles)
    eps=1.e-6
    for k in (0,1):
        ap=angles.copy();am=angles.copy()
        ap[k]+=eps;am[k]-=eps
        ep=gvb_scf_energy_gradient(p,Q,ap,gradient=False)[0]
        em=gvb_scf_energy_gradient(p,Q,am,gradient=False)[0]
        assert np.isclose((ep-em)/(2*eps),grad[k],atol=2e-6)
    for i,j in [(1,3),(2,6),(0,4),(3,5)]:
        a=np.zeros((7,7));a[i,j]=1;a[j,i]=-1
        ep=gvb_scf_energy_gradient(p,Q@expm(eps*a),angles,gradient=False)[0]
        em=gvb_scf_energy_gradient(p,Q@expm(-eps*a),angles,gradient=False)[0]
        assert np.isclose((ep-em)/(2*eps),np.sum(G*(Q@a)),atol=2e-6)
