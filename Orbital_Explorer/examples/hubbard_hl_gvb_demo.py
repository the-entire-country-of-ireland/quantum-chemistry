"""No-PySCF educational check of RHF vs HL vs GVB in one Hubbard bond.

Run from repository root: python examples/hubbard_hl_gvb_demo.py
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qchem_analysis.perfect_pairing import (
    PairHamiltonian,PairSpace,make_rhf_wavefunction,
    make_hl_wavefunction,optimize_gvb_pp,
)


def dimer(t,u):
    # Unitary site-to-bonding/antibonding orbitals.
    Q=np.array([[1,1],[1,-1]],float)/np.sqrt(2)
    eri=np.zeros((2,2,2,2))
    for p,q,r,s in np.ndindex(eri.shape):
        eri[p,q,r,s]=u*sum(Q[k,p]*Q[k,q]*Q[k,r]*Q[k,s] for k in range(2))
    return PairHamiltonian(np.diag([-t,t]),eri,0.,-2*t+u/2)


if __name__=='__main__':
    print('Two electrons, one symmetric two-site Hubbard bond; t=1')
    print('     U       RHF        HL        GVB-PP     exact singlet')
    for U in [0,1,2,4,8,16]:
        ham=dimer(1,U)
        space=PairSpace(np.eye(2),np.zeros((2,0)),('H--H',),(('H1','H2'),),
             np.array([np.diag([1,-1])/np.sqrt(2)]),np.zeros((1,2,2)),
             np.ones((1,2)),np.ones(1))
        rhf=make_rhf_wavefunction(ham).energy
        hl=make_hl_wavefunction(ham,space).energy
        pp=optimize_gvb_pp(ham,space).energy
        exact=(U-np.sqrt(U*U+16))/2
        print(f'{U:7.2f} {rhf:10.5f} {hl:10.5f} {pp:12.5f} {exact:15.5f}')
