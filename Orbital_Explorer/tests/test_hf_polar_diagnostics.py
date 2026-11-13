"""Regressions for the HF lone-pair/misbinding and ionic-pair display failure.

The pure tests do not claim that a 6-31G/PySCF HF calculation has run.
They test the orbital classification and physical pair-density bookkeeping.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parents[1]


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


def test_hf_occupied_bond_selection_excludes_pure_fluorine_lone_pair():
    """Old overlap-only matcher would choose orbital 0 (F lone pair)."""
    path=HERE/'qchem_analysis/molecular_fixed.py'
    namespace={'np':np}
    namespace['bond_participation']=extract(path,'bond_participation',namespace)
    namespace['atomic_iao_populations']=extract(path,'atomic_iao_populations',namespace)
    @dataclass(frozen=True)
    class BondOrbital:
        orbital_index:int;bond_index:int;label:str;confidence:float;kind:str
    namespace.update(BondOrbital=BondOrbital, RHFResult=object,
        linear_sum_assignment=linear_sum_assignment,
        bond_sites=lambda seed:[{'index':0,'name':'F1–H2','a':0,'b':1}],
        bond_trials=lambda result,seed:[('F1–H2 σ',np.array([0.98,0.07,0.,0.20]))])
    func=extract(path,'bond_orbital_map',namespace)
    # Three orthonormal occupied IBOs: F lone pair, F-H σ bond, F other lone pair.
    lone=np.array([1.,0,0,0])
    bond=np.array([0.,.8,0.,.6])
    other=np.array([0.,0,1.,0])
    occupied=np.column_stack([lone,bond,other])
    class FakeMol:
        def intor(self,tag):return np.eye(4)
    result=SimpleNamespace(mol=FakeMol(), local_basis=SimpleNamespace(
        B_atomic=np.eye(4), atomic_labels=['F1:2s','F1:2pz','F1:2px','H2:1s']))
    seed=SimpleNamespace(symbols=['F','H'])
    assert abs(lone@namespace['bond_trials'](result,seed)[0][1])**2 > abs(bond@namespace['bond_trials'](result,seed)[0][1])**2
    mapped=func(result,seed,occupied)
    assert len(mapped)==1
    assert mapped[0].orbital_index==1, 'Pure F lone pair must not be assigned as F-H bond'


def test_gvb_orbital_labels_are_not_forced_to_one_atom_each():
    code=(HERE/'pages/molecular.py').read_text()
    assert '_reorder_gvb_to_atoms' not in code
    assert "('GVB pair orbital 1',left),('GVB pair orbital 2',right)" in code
    assert 'pair_atomic_populations' in code


def test_two_electron_iao_populations_use_natural_occupation_not_vw_weights():
    path=HERE/'pages/molecular.py'
    func=extract(path,'pair_atomic_populations',{'np':np,
        'atomic_iao_populations':lambda result,orb: {'F1':float(orb[0]**2),
                                                     'H2':float(orb[1]**2),
                                                     'unrepresented':0.}})
    theta=np.arcsin(np.sqrt(.1))
    wave=SimpleNamespace(frame=np.eye(2),occupations=np.array([[1.8,.2]]))
    space=SimpleNamespace(orbitals_ao=np.eye(2))
    pop=func(None,space,wave,0)
    assert pop['F1']==pytest.approx(1.8)
    assert pop['H2']==pytest.approx(.2)
    assert sum(pop.values())==pytest.approx(2.)


def test_polar_two_site_interacting_pair_can_place_both_gvb_functions_on_f():
    from qchem_analysis.perfect_pairing import (
        PairSpace,PairHamiltonian,make_rhf_wavefunction,optimize_gvb_pp,bond_pair_orbitals)
    # Simplified energetics of electronegative F and electropositive H.
    h_site=np.array([[-3.,-.3],[-.3,0.]])
    eri_site=np.zeros((2,2,2,2))
    eri_site[0,0,0,0]=1.3
    eri_site[1,1,1,1]=1.3
    eri_site[0,0,1,1]=eri_site[1,1,0,0]=.2
    # Pick RHF occupied orbital close to the F-centered orbital.
    ang=.10
    Q=np.array([[np.cos(ang),-np.sin(ang)],
                [np.sin(ang), np.cos(ang)]])
    h=Q.T@h_site@Q
    v=np.einsum('ap,bq,cr,ds,abcd->pqrs',Q,Q,Q,Q,eri_site,optimize=True)
    ham=PairHamiltonian(h,v,0.,0.)
    gem=np.array([[0.,2**-.5],[2**-.5,0.]])
    space=PairSpace(Q,np.empty((2,0)),('F-H',),(('F1','H2'),),gem[None,:,:],
                    np.zeros((1,2,2)),np.ones((1,2)),np.ones(1))
    rhf=make_rhf_wavefunction(ham)
    gvb=optimize_gvb_pp(ham,space,maxiter=180)
    v,w,_=bond_pair_orbitals(space,gvb,0)
    assert gvb.energy<=rhf.energy+1e-8
    assert v[0]**2>v[1]**2
    assert w[0]**2>w[1]**2
