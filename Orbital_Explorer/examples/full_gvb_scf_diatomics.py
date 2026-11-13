"""Reproducible real-molecule GVB-SCF smoke check (requires PySCF).

    PYTHONPATH=. python examples/full_gvb_scf_diatomics.py

Runs physical 6-31G H2 and HF at their supplied geometries, prints RHF,
fixed-pair GVB, and all-orbital GVB-SCF energies and natural occupations.
"""
from __future__ import annotations

import numpy as np

from qchem_analysis.perfect_pairing import (
    PairReference,construct_pair_space,integral_hamiltonian,optimize_gvb_pp)
from qchem_analysis.gvb_scf import (
    construct_gvb_scf_problem,optimize_full_gvb_scf,full_pair_orbitals)


def run(name,xyz,center_left,center_right,core):
    from types import SimpleNamespace
    from pyscf import gto,scf,lo

    mol=gto.M(atom=xyz,basis='6-31g',unit='Angstrom',verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-10;mf.kernel()
    if not mf.converged:
        raise RuntimeError(f'{name}: RHF failed')
    occupied=mf.mo_coeff[:,core:mol.nelectron//2]
    if occupied.shape[1]>1:
        occupied=lo.Boys(mol,occupied).kernel()
    S=mol.intor('int1e_ovlp')
    il,ir=mol.aoslice_by_atom()[center_left,2:4],mol.aoslice_by_atom()[center_right,2:4]
    # Identify the selected bond occupied orbital by terminal-H atomic weight.
    idx=max(range(occupied.shape[1]),key=lambda k:np.linalg.norm(occupied[ir[0]:ir[1],k]))
    occ=occupied[:,idx]
    a=np.zeros_like(occ);b=np.zeros_like(occ)
    a[il[0]:il[1]]=occ[il[0]:il[1]]
    b[ir[0]:ir[1]]=occ[ir[0]:ir[1]]
    reference=PairReference(idx,name,occ,a,b,(mol.atom_symbol(center_left),mol.atom_symbol(center_right)))
    data=SimpleNamespace(mol=mol,mf=mf,mode='full',frozen_core=False,
        coefficients_ao=mf.mo_coeff,nocc=mol.nelectron//2,
        core_spatial_count=core,energy_hartree=float(mf.e_tot))
    sub=construct_pair_space(data,occupied,[reference])
    ham=integral_hamiltonian(data,sub)
    warm=optimize_gvb_pp(ham,sub,maxiter=95)
    prob=construct_gvb_scf_problem(data,sub)
    opt=optimize_full_gvb_scf(prob,initial_pair_wave=warm,maxiter=180)
    u,v,pair_overlap=full_pair_orbitals(prob,opt,0)
    assert np.isclose(u@S@u,1,atol=1e-7) and np.isclose(v@S@v,1,atol=1e-7)
    print(f'{name}: RHF={mf.e_tot:.9f}  fixed-pair={warm.energy:.9f}  full-GVB-SCF={opt.energy:.9f} Ha')
    print(f'  natural occupations={opt.occupations[0]}, pair overlap={pair_overlap:.6f},'
          f' converged={opt.converged}, total iterations={opt.iterations}')
    # AO Mulliken projection of this *selected pair's* two-electron 1-RDM.
    nat=prob.orbitals_ao@opt.frame[:,prob.pair_offset:prob.pair_offset+2]
    dm=(nat*opt.occupations[0])@nat.T
    pop=dm@S
    print('  pair electron populations:',{mol.atom_symbol(i)+str(i+1):
          float(np.trace(pop[s[2]:s[3],s[2]:s[3]]))
          for i,s in enumerate(mol.aoslice_by_atom())})
    return opt


if __name__=='__main__':
    run('H2','H 0 0 -.37; H 0 0 .37',0,1,0)
    run('HF','F 0 0 0; H 0 0 .917',0,1,1)
