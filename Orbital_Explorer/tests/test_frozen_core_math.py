"""Small integral-tensor sanity tests for the constrained closed-shell HF energy."""
import numpy as np


def hf_energy(h, eri, density, nuclear=0):
    j=np.einsum('ijkl,kl->ij',eri,density)
    k=np.einsum('ikjl,kl->ij',eri,density)
    return float(np.einsum('ij,ij->',density,h + .5*(j-.5*k))+nuclear)


def test_frozen_core_energy_partition_includes_core_valence_interaction():
    rng=np.random.default_rng(183)
    # Real, 8-fold symmetric ERI built from a positive pair-density Gram matrix.
    n=5
    pair=rng.normal(size=(n,n,7))
    pair=.5*(pair+pair.transpose(1,0,2))
    eri=np.einsum('ijq,klq->ijkl',pair,pair)
    h=rng.normal(size=(n,n)); h=.5*(h+h.T)
    q,_=np.linalg.qr(rng.normal(size=(n,n)))
    core=2*np.outer(q[:,0],q[:,0])
    valence=2*np.outer(q[:,1],q[:,1])
    jcore=np.einsum('ijkl,kl->ij',eri,core)
    kcore=np.einsum('ikjl,kl->ij',eri,core)
    vcore=jcore-.5*kcore
    core_energy=np.einsum('ij,ij->',core,h+.5*vcore)
    jval=np.einsum('ijkl,kl->ij',eri,valence)
    kval=np.einsum('ikjl,kl->ij',eri,valence)
    # Valence SCF uses h_eff = h + J_core - 0.5 K_core.
    valence_energy=np.einsum('ij,ij->',valence,h+vcore+.5*(jval-.5*kval))
    assert np.allclose(core_energy+valence_energy,
                       hf_energy(h,eri,core+valence),atol=1e-9)


def test_overlap_orthogonality_of_fixed_core_valence_complement():
    from scipy.linalg import null_space
    rng=np.random.default_rng(31)
    a=rng.normal(size=(10,10))
    overlap=a.T@a+np.eye(10)
    w,u=np.linalg.eigh(overlap)
    x=(u*w**-.5)@u.T
    core_ao=rng.normal(size=(10,2))
    q,_=np.linalg.qr(x.T@overlap@core_ao)
    v=null_space(q.T)
    core=x@q
    valence=x@v
    c=np.column_stack((core,valence))
    assert np.allclose(c.T@overlap@c,np.eye(10),atol=1e-10)
