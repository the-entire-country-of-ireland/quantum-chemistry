"""Opt-in real-PySCF tests.

Run on a machine with requirements installed:
    ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 pytest -q tests/test_perfect_pairing_pyscf_integration.py

These are skipped in minimal CI environments lacking PySCF.
"""
import os
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(
    os.environ.get('ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION')!='1',
    reason='Set ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 for full molecular tests')


def _result(mol,mf,ncore):
    return SimpleNamespace(mol=mol,mf=mf,mode='full',frozen_core=False,
        coefficients_ao=mf.mo_coeff,energy_hartree=mf.e_tot,
        core_spatial_count=ncore,nocc=mol.nelectron//2)


def test_h2_sto3g_gvb_recovers_singlet_fci():
    pyscf=pytest.importorskip('pyscf')
    from pyscf import gto,scf,fci
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        make_rhf_wavefunction,make_hl_wavefunction,optimize_gvb_pp)
    mol=gto.M(atom='H 0 0 -0.37; H 0 0 0.37',unit='Angstrom',basis='sto-3g',verbose=0)
    mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()
    res=_result(mol,mf,0)
    occ=mf.mo_coeff[:,0]
    left=np.array([occ[0],0.])
    right=np.array([0.,occ[1]])
    # Full single-pair subspace and true H1/H2 atom components.
    refs=[PairReference(0,'H--H',occ,left,right,('H1','H2'))]
    space=construct_pair_space(res,occ[:,None],refs)
    ham=integral_hamiltonian(res,space)
    rhf=make_rhf_wavefunction(ham)
    hl=make_hl_wavefunction(ham,space)
    gvb=optimize_gvb_pp(ham,space,maxiter=90)
    exact=fci.FCI(mol,mf.mo_coeff).kernel()[0]
    assert np.isclose(rhf.energy,mf.e_tot,atol=1e-7)
    assert gvb.energy<=rhf.energy+1e-8
    assert gvb.energy<=hl.energy+1e-8
    # In the two-orbital homonuclear singlet with a real GVB pair, the
    # variational spin-coupled ansatz spans the FCI ground-state eigenvector.
    assert np.isclose(gvb.energy,exact,atol=2e-6),(gvb.energy,exact)


def test_methane_four_spin_singlet_pairs_are_simultaneous():
    pytest.importorskip('pyscf')
    from pyscf import gto,scf,lo
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        make_rhf_wavefunction,optimize_gvb_pp)
    xyz='''C 0.0 0.0 0.0;
    H 0.629 0.629 0.629;
    H 0.629 -0.629 -0.629;
    H -0.629 0.629 -0.629;
    H -0.629 -0.629 0.629'''
    mol=gto.M(atom=xyz,unit='Angstrom',basis='sto-3g',verbose=0)
    mf=scf.RHF(mol);mf.kernel()
    res=_result(mol,mf,1)
    localized=lo.Boys(mol,mf.mo_coeff[:,1:5]).kernel()
    C=mol.nao_nr()
    # Use atom-specific AO pieces of each Boys localized C-H bond.
    Hs=[]
    for h in range(1,5):
        ao0,ao1=mol.aoslice_by_atom()[h,2:]
        Hs.append((h,int(ao0),int(ao1)))
    remaining=set(range(1,5))
    refs=[]
    c0,c1=mol.aoslice_by_atom()[0,2:]
    for i in range(4):
        o=localized[:,i]
        h=max(remaining,key=lambda j:np.linalg.norm(o[Hs[j-1][1]:Hs[j-1][2]]))
        remaining.remove(h)
        left=np.zeros(C); left[c0:c1]=o[c0:c1]
        right=np.zeros(C)
        right[Hs[h-1][1]:Hs[h-1][2]]=o[Hs[h-1][1]:Hs[h-1][2]]
        refs.append(PairReference(i,f'C-H{h}',o,left,right,('C',f'H{h}')))
    space=construct_pair_space(res,localized,refs)
    ham=integral_hamiltonian(res,space)
    rhf=make_rhf_wavefunction(ham)
    gvb=optimize_gvb_pp(ham,space,maxiter=35)
    assert np.isclose(rhf.energy,mf.e_tot,atol=5e-6)
    assert gvb.energy<=rhf.energy+1e-8
    assert len(gvb.theta)==4
    assert np.isfinite(gvb.energy)


def test_h2_full_631g_gvb_scf_varies_external_orbitals():
    """Real Coulomb integrals: full orbital optimization cannot worsen active GVB."""
    pytest.importorskip('pyscf')
    from pyscf import gto, scf, fci
    from qchem_analysis.perfect_pairing import (
        PairReference, construct_pair_space, integral_hamiltonian, optimize_gvb_pp)
    from qchem_analysis.gvb_scf import construct_gvb_scf_problem,optimize_full_gvb_scf
    mol=gto.M(atom='H 0 0 -.37; H 0 0 .37', unit='Angstrom',basis='6-31g',verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-11;mf.kernel()
    res=_result(mol,mf,0)
    occ=mf.mo_coeff[:,0]
    slices=mol.aoslice_by_atom()
    left=np.zeros_like(occ);right=np.zeros_like(occ)
    left[slices[0,2]:slices[0,3]]=occ[slices[0,2]:slices[0,3]]
    right[slices[1,2]:slices[1,3]]=occ[slices[1,2]:slices[1,3]]
    space=construct_pair_space(res,occ[:,None],
        [PairReference(0,'H-H',occ,left,right,('H1','H2'))])
    fixedham=integral_hamiltonian(res,space)
    initial=optimize_gvb_pp(fixedham,space,maxiter=90)
    full=construct_gvb_scf_problem(res,space)
    wave=optimize_full_gvb_scf(full,initial_pair_wave=initial,maxiter=180)
    assert np.isclose(full.rhf_energy,mf.e_tot,atol=3e-6)
    assert wave.energy<=initial.energy+1e-5
    assert wave.energy<=mf.e_tot+1e-8
    assert wave.energy>=fci.FCI(mol,mf.mo_coeff).kernel()[0]-1e-7
    assert wave.occupations.shape==(1,2)
    assert np.allclose(wave.frame.T@wave.frame,np.eye(len(wave.frame)),atol=2e-6)


def test_hf_full_gvb_scf_bond_remains_fluorine_polarized():
    """Regression for observed wrong-HF interpretation (run locally with PySCF)."""
    pytest.importorskip('pyscf')
    from pyscf import gto,scf,lo
    from qchem_analysis.perfect_pairing import PairReference,construct_pair_space,integral_hamiltonian,optimize_gvb_pp
    from qchem_analysis.gvb_scf import construct_gvb_scf_problem,optimize_full_gvb_scf,full_pair_orbitals
    mol=gto.M(atom='F 0 0 0; H 0 0 .917',basis='6-31g',unit='Angstrom',verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-10;mf.kernel()
    res=_result(mol,mf,1)
    occupied=lo.Boys(mol,mf.mo_coeff[:,1:5]).kernel()
    S=mol.intor('int1e_ovlp')
    # H-bearing occupied orbital (H contribution in Mulliken AO partition).
    h0,h1=mol.aoslice_by_atom()[1,2:]
    idx=max(range(4),key=lambda j:np.linalg.norm(occupied[h0:h1,j]))
    occ=occupied[:,idx]
    f0,f1=mol.aoslice_by_atom()[0,2:]
    f=np.zeros_like(occ);f[f0:f1]=occ[f0:f1]
    h=np.zeros_like(occ);h[h0:h1]=occ[h0:h1]
    space=construct_pair_space(res,occupied,
        [PairReference(idx,'F-H',occ,f,h,('F','H'))])
    active=integral_hamiltonian(res,space)
    warm=optimize_gvb_pp(active,space,maxiter=75)
    full=construct_gvb_scf_problem(res,space)
    wave=optimize_full_gvb_scf(full,initial_pair_wave=warm,maxiter=180)
    assert wave.energy<=mf.e_tot+1e-7
    assert wave.energy<=warm.energy+1e-5
    u,v,suv=full_pair_orbitals(full,wave,0)
    assert abs(u@S@u-1)<1e-5 and abs(v@S@v-1)<1e-5
    # Calculate total F/H gross Mulliken population for the TWO-electron pair,
    # not a fictitious atom assignment for each nonorthogonal GVB orbital.
    natural=full.orbitals_ao@wave.frame[:,full.pair_offset:full.pair_offset+2]
    dm=(natural*wave.occupations[0])@natural.T
    ps=dm@S
    fpop=float(np.trace(ps[f0:f1,f0:f1]))
    hpop=float(np.trace(ps[h0:h1,h0:h1]))
    assert fpop>hpop,(fpop,hpop)


def test_beh2_full_gvb_scf_two_pairs_relax_against_core_and_virtuals():
    """Real BeH2 sigma bonds: full orbital SCF cannot raise active-space energy."""
    pytest.importorskip('pyscf')
    from pyscf import gto,scf,lo
    from qchem_analysis.perfect_pairing import PairReference,construct_pair_space,integral_hamiltonian,optimize_gvb_pp
    from qchem_analysis.gvb_scf import construct_gvb_scf_problem,optimize_full_gvb_scf
    mol=gto.M(atom='Be 0 0 0; H 0 0 -1.33; H 0 0 1.33',
              basis='sto-3g',unit='Angstrom',verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-10;mf.kernel()
    res=_result(mol,mf,1)
    val=lo.Boys(mol,mf.mo_coeff[:,1:3]).kernel()
    h_slices=[mol.aoslice_by_atom()[i,2:4] for i in (1,2)]
    available={0,1}
    refs=[]
    be0,be1=mol.aoslice_by_atom()[0,2:4]
    for idx in range(2):
        orb=val[:,idx]
        which=max(available,key=lambda j:float(np.linalg.norm(orb[h_slices[j][0]:h_slices[j][1]])))
        available.remove(which)
        a=np.zeros_like(orb);a[be0:be1]=orb[be0:be1]
        h0,h1=h_slices[which]
        b=np.zeros_like(orb);b[h0:h1]=orb[h0:h1]
        refs.append(PairReference(idx,f'Be-H{which+1}',orb,a,b,('Be',f'H{which+1}')))
    space=construct_pair_space(res,val,refs)
    fixed=integral_hamiltonian(res,space)
    warm=optimize_gvb_pp(fixed,space,maxiter=80)
    full=construct_gvb_scf_problem(res,space)
    state=optimize_full_gvb_scf(full,initial_pair_wave=warm,maxiter=160)
    assert state.energy<=full.rhf_energy+1e-7
    assert state.energy<=warm.energy+1e-5
    assert state.occupations.shape==(2,2)
    assert np.allclose(np.sum(state.occupations,axis=1),2,atol=1e-10)
