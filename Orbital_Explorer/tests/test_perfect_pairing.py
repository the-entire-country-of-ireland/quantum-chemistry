"""Algorithm tests independent of PySCF and Streamlit installation."""
import numpy as np
from itertools import combinations

from qchem_analysis.perfect_pairing import (
    PairHamiltonian, PairSpace, _site_matrix, _hl_natural_frame,
    _compose_frame, pp_energy, optimize_gvb_pp, make_hl_wavefunction,
    make_rhf_wavefunction,
)


def hubbard_dimer(t, u):
    # Orthonormal bonding/antibonding orbitals g=(L+R)/sqrt2,
    # u=(L-R)/sqrt2; on-site repulsion only.
    R=np.array([[1,1],[1,-1]])/np.sqrt(2)
    eri=np.zeros((2,2,2,2))
    for p,q,r,s in np.ndindex(eri.shape):
        eri[p,q,r,s]=u*sum(R[i,p]*R[i,q]*R[i,r]*R[i,s] for i in range(2))
    return PairHamiltonian(np.diag([-t,t]), eri, 0., -2*t+u/2)


def dummy_space(ham, mats=None):
    m=len(ham.h)//2
    if mats is None:
        mats=np.array([np.diag([1,-1])/np.sqrt(2) for _ in range(m)])
    return PairSpace(np.eye(2*m), np.zeros((2*m,0)), tuple(f'b{i}' for i in range(m)),
                     tuple(('A','B') for _ in range(m)), mats,
                     np.zeros((m,2,2*m)),np.ones((m,2)),np.ones(m))


def test_hl_site_normalization_and_symmetry():
    a=np.array([1.,1.])/np.sqrt(2)
    b=np.array([1.,-1.])/np.sqrt(2)
    M=_site_matrix(a,b)
    assert np.allclose(M,np.diag([1.,-1.])/np.sqrt(2))
    assert np.isclose(np.sum(M*M),1)
    mat,theta=_hl_natural_frame(dummy_space(hubbard_dimer(1,4),M[None]))
    assert np.allclose(theta,[np.pi/4])
    assert np.allclose(mat.T@mat,np.eye(2))


def test_hubbard_dimer_known_limits_and_exact_energy():
    for U in (0.,.5,2.,4.,12.):
        ham=hubbard_dimer(1.,U)
        space=dummy_space(ham)
        rhf=make_rhf_wavefunction(ham)
        hl=make_hl_wavefunction(ham,space)
        gvb=optimize_gvb_pp(ham,space,maxiter=95)
        exact=(U-np.sqrt(U*U+16))/2
        assert np.isclose(rhf.energy,-2+U/2,atol=1e-10)
        assert np.isclose(hl.energy,0.,atol=1e-10)
        assert np.isclose(gvb.energy,exact,atol=2e-6), (U,gvb.energy,exact)
        assert np.isclose(gvb.occupations.sum(),2.)
        assert gvb.energy<=rhf.energy+1e-9
        assert gvb.energy<=hl.energy+1e-9


def test_two_independent_dimers_energy_adds():
    one=hubbard_dimer(1.,5.)
    h=np.zeros((4,4));h[:2,:2]=one.h;h[2:,2:]=one.h
    eri=np.zeros((4,4,4,4));eri[:2,:2,:2,:2]=one.eri;eri[2:,2:,2:,2:]=one.eri
    h2=PairHamiltonian(h,eri,0.,2*one.rhf_energy)
    pp=optimize_gvb_pp(h2,dummy_space(h2),maxiter=95)
    expected=2*(5-np.sqrt(25+16))/2
    assert np.isclose(pp.energy,expected,atol=2e-6)
    assert pp.energy<=make_rhf_wavefunction(h2).energy+1e-8


def test_orbital_rotations_orthonormal():
    k=6;m=3
    x=np.r_[np.array([.1,.2,.3]),np.array([.5,-.25,.2]),
             np.linspace(-.4,.4,sum(i//2!=j//2 for i,j in combinations(range(k),2)))]
    mat=_compose_frame(x,k,True)
    assert np.allclose(mat.T@mat,np.eye(k),atol=1e-10)


def test_two_pair_energy_matches_explicit_second_quantization():
    """Independent reference calculation using creation/annihilation signs."""
    rng=np.random.default_rng(561)
    n=4
    h=rng.normal(size=(n,n));h=(h+h.T)/2
    B=rng.normal(size=(3,n,n))
    B=(B+np.transpose(B,(0,2,1)))/2
    eri=np.einsum('xpq,xrs->pqrs',B,B)
    ham=PairHamiltonian(h,eri,1.37,0.)
    phi=[.17,-.27]
    frame=np.eye(n)
    for k,a in enumerate(phi):
        frame[2*k:2*k+2,2*k:2*k+2]=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    theta=np.array([.21,.36])
    e_test=pp_energy(ham,frame,theta)
    # Rotate one- and two-electron integrals into frame of each geminal.
    h=frame.T@h@frame
    from qchem_analysis.perfect_pairing import _rotate_eri
    eri=_rotate_eri(eri,frame)
    # Occupied determinants: g1^2g2^2, g1^2u2^2, u1^2g2^2, u1^2u2^2.
    # A spin-orbital determinant is a bitfield: alpha orbitals 0..3, beta 4..7.
    coeffs={}
    for i,j in np.ndindex(2,2):
        p,q=i,2+j
        bits=(1<<p)|(1<<(n+p))|(1<<q)|(1<<(n+q))
        coeffs[bits]=(np.cos(theta[0]) if i==0 else -np.sin(theta[0])) * \
                     (np.cos(theta[1]) if j==0 else -np.sin(theta[1]))
    def annihilate(bit,idx):
        if not (bit>>idx)&1:return None
        sign=(-1)**((bit & ((1<<idx)-1)).bit_count())
        return bit ^ (1<<idx),sign
    def create(bit,idx):
        if (bit>>idx)&1:return None
        sign=(-1)**((bit & ((1<<idx)-1)).bit_count())
        return bit | (1<<idx),sign
    def apply(bit,steps):
        sign=1
        for maker,idx in steps:
            out=create(bit,idx) if maker else annihilate(bit,idx)
            if out is None:return None
            bit,s=out;sign*=s
        return bit,sign
    expected=ham.constant
    for bra,c_bra in coeffs.items():
        for ket,c_ket in coeffs.items():
            elem=0.
            for spin in range(2):
                for p,q in np.ndindex(n,n):
                    out=apply(ket,[(False,q+spin*n),(True,p+spin*n)])
                    if out and out[0]==bra: elem+=h[p,q]*out[1]
            for s1,s2 in np.ndindex(2,2):
                for p,q,r,s in np.ndindex(n,n,n,n):
                    out=apply(ket,[(False,q+s1*n),(False,s+s2*n),
                                   (True,r+s2*n),(True,p+s1*n)])
                    if out and out[0]==bra:elem+=.5*eri[p,q,r,s]*out[1]
            expected+=c_bra*c_ket*elem
    assert np.isclose(e_test,expected,atol=1e-9), (e_test,expected)


def test_finite_overlap_hl_pair_normalizes_correctly():
    from qchem_analysis.perfect_pairing import _site_matrix
    s=0.4
    a=np.array([1.,0.]); b=np.array([s,np.sqrt(1-s*s)])
    mat=_site_matrix(a,b)
    eig=np.linalg.eigvalsh(mat)
    assert np.isclose(eig[0],-(1-s)/np.sqrt(2*(1+s*s)))
    assert np.isclose(eig[1],(1+s)/np.sqrt(2*(1+s*s)))
    assert np.isclose(np.linalg.norm(mat),1.)


def test_two_pair_active_space_orthogonalization_and_projection():
    from types import SimpleNamespace
    from qchem_analysis.perfect_pairing import PairReference, construct_pair_space
    # Four orthonormal AOs: occupied g1,g2 and virtual u1,u2.
    # True atom-site ingredients are (g1+u1),(g1-u1), etc.
    n=4
    S=np.eye(n)
    occupied=np.eye(n)[:,:2]
    virtual=np.eye(n)[:,2:]
    full=np.column_stack([occupied,virtual])
    g1,g2,u1,u2=full.T
    res=SimpleNamespace(mode='full',frozen_core=False,
        mol=SimpleNamespace(intor=lambda _:S),
        coefficients_ao=full,nocc=2,core_spatial_count=0)
    refs=[PairReference(0,'A-B',g1,(g1+u1)/2,(g1-u1)/2,('A','B')),
          PairReference(1,'B-C',g2,(g2+u2)/2,(g2-u2)/2,('B','C'))]
    space=construct_pair_space(res,occupied,refs)
    assert np.allclose(space.orbitals_ao.T@S@space.orbitals_ao,np.eye(4))
    assert np.allclose(space.site_fidelities,np.ones((2,2)))
    assert np.allclose(space.hl_matrices, [np.diag([1,-1])/np.sqrt(2)]*2)
    assert space.inactive_ao.shape==(4,0)


def test_optional_interpair_rotations_do_not_lose_variational_rhf_baseline():
    one=hubbard_dimer(1.,3.)
    h=np.zeros((4,4)); h[:2,:2]=one.h; h[2:,2:]=one.h
    eri=np.zeros((4,4,4,4)); eri[:2,:2,:2,:2]=one.eri; eri[2:,2:,2:,2:]=one.eri
    ham=PairHamiltonian(h,eri,0.,2*one.rhf_energy)
    fixed=optimize_gvb_pp(ham,dummy_space(ham),maxiter=30)
    relaxed=optimize_gvb_pp(ham,dummy_space(ham),rotate_subspaces=True,maxiter=30)
    assert np.isfinite(relaxed.energy)
    assert relaxed.energy<=make_rhf_wavefunction(ham).energy+1e-8
    assert relaxed.energy<=fixed.energy+2e-5
