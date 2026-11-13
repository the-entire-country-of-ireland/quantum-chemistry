"""Pure orbital-space regression tests; real PySCF checks opt in separately."""
from types import SimpleNamespace
import numpy as np
import pytest
import ast
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from qchem_analysis.bond_selection import bond_sites


def _module_under_test():
    """Run chemistry-free function bodies without importing optional PySCF."""
    path=Path(__file__).resolve().parents[1]/'qchem_analysis/molecular_fixed.py'
    tree=ast.parse(path.read_text())
    names={'BondOrbital','LocalizedPair','_valence_shell','_shell_label',
           'bond_participation','atomic_iao_populations','bond_orbital_map',
           'iao_site_s_p','localized_pair_catalog','lone_pair_virtual_seed',
           'lone_pair_ideal_reference'}
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    nodes.extend(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
                 and node.name in names)
    ns={'np':np,'dataclass':dataclass,'linear_sum_assignment':linear_sum_assignment,
        'bond_sites':bond_sites}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 str(path),'exec'),ns)
    return ns


@pytest.fixture
def mf():
    return _module_under_test()
from qchem_analysis.fixed_geometry_data import fixed_seed


class FakeMol:
    def __init__(self, n, symbols=None):
        self.n = n
        self.symbols = symbols or []
    def intor(self, name):
        assert name == 'int1e_ovlp'
        return np.eye(self.n)
    def atom_symbol(self, i):
        return self.symbols[i]
    def atom_charge(self, i):
        return {'F':9,'O':8,'H':1}[self.symbols[i]]


def test_h2_sole_valence_ibo_is_bond_even_with_missing_ideal_trial(mf):
    seed,_=fixed_seed('H₂ · hydrogen')
    result=SimpleNamespace(mol=FakeMol(2,['H','H']),local_basis=SimpleNamespace(
        B_atomic=np.eye(2),atomic_labels=['H1:1s','H2:1s']))
    mo=np.array([[2**-.5],[2**-.5]])
    mf['bond_trials']=lambda result,seed: []
    bonds=mf['bond_orbital_map'](result,seed,mo)
    assert len(bonds)==1
    assert bonds[0].kind=='σ'
    assert bonds[0].orbital_index==0
    entries=mf['localized_pair_catalog'](result,seed,mo,bonds)
    assert len(entries)==1
    assert entries[0].kind=='σ'


def test_hf_bond_and_three_f_lone_pairs(mf):
    seed,_=fixed_seed('HF · hydrogen fluoride')
    basis=np.eye(5)
    atom_labels=['F1:2s','F1:2px','F1:2py','F1:2pz','H2:1s']
    result=SimpleNamespace(mol=FakeMol(5,['F','H']),local_basis=SimpleNamespace(
        B_atomic=basis,atomic_labels=atom_labels))
    sigma=np.zeros(5)
    sigma[3]=np.sqrt(.75);sigma[4]=np.sqrt(.25)
    occ=np.column_stack([basis[:,0],basis[:,1],basis[:,2],sigma])
    mf['bond_trials']=lambda result,seed:[('F1–H2 σ',sigma)]
    bond_matches=mf['bond_orbital_map'](result,seed,occ)
    assert len(bond_matches)==1
    assert bond_matches[0].orbital_index==3
    catalog=mf['localized_pair_catalog'](result,seed,occ,bond_matches)
    assert len(catalog)==4
    assert sum(item.kind=='σ' for item in catalog)==1
    lone=[item for item in catalog if item.kind=='lone pair']
    assert len(lone)==3
    assert all(x.atom_index==0 and x.site_weight>.9 for x in lone)
    assert len(set(x.orbital_index for x in catalog))==4


def test_water_two_bonds_and_two_oxygen_lone_pairs(mf):
    seed,_=fixed_seed('H₂O · water')
    basis=np.eye(6)
    labels=['O1:2s','O1:2px','O1:2py','O1:2pz','H2:1s','H3:1s']
    result=SimpleNamespace(mol=FakeMol(6,['O','H','H']),local_basis=SimpleNamespace(
        B_atomic=basis,atomic_labels=labels))
    bond1=np.zeros(6);bond1[1]=np.sqrt(.7);bond1[4]=np.sqrt(.3)
    bond2=np.zeros(6);bond2[3]=np.sqrt(.7);bond2[5]=np.sqrt(.3)
    occ=np.column_stack([basis[:,0],basis[:,2],bond1,bond2])
    bonds=[mf['BondOrbital'](2,0,'O1–H2 σ',1.,'σ'),
           mf['BondOrbital'](3,1,'O1–H3 σ',1.,'σ')]
    catalog=mf['localized_pair_catalog'](result,seed,occ,bonds)
    assert sum(x.kind=='lone pair' for x in catalog)==2
    assert sum(x.kind=='σ' for x in catalog)==2
    assert all(x.atom_index==0 for x in catalog if x.kind=='lone pair')


def test_unassigned_orbitals_are_not_falsely_called_lone_pairs(mf):
    seed,_=fixed_seed('HF · hydrogen fluoride')
    result=SimpleNamespace(mol=FakeMol(2,['F','H']),local_basis=SimpleNamespace(
        B_atomic=np.eye(2),atomic_labels=['F1:2s','H2:1s']))
    occupied=np.array([[2**-.5],[2**-.5]])
    catalog=mf['localized_pair_catalog'](result,seed,occupied,[])
    assert len(catalog)==1 and catalog[0].kind=='unassigned'


def test_lone_pair_virtual_partner_uses_same_atom_and_is_orthogonal(mf):
    # A split-valence basis: occupied 2s and vacant 3s radial functions on F.
    labels=[(0,'F','2s',''),(0,'F','3s',''),(0,'F','2p','x')]
    class SplitMol(FakeMol):
        def ao_labels(self,fmt=False):
            assert fmt is False
            return labels
    result=SimpleNamespace(mol=SplitMol(3,['F']),
        coefficients_ao=np.eye(3),nocc=1,local_basis=SimpleNamespace(
        B_atomic=np.eye(3)[:,:1],atomic_labels=['F1:2s']))
    partner=mf['lone_pair_virtual_seed'](result,np.array([1.,0.,0.]),0)
    assert np.isclose(partner@partner,1.)
    assert abs(partner[0])<1e-10
    assert np.isclose(abs(partner[1]),1.)


def test_ideal_lone_pair_reference_is_explicitly_a_convention(mf):
    seed,_=fixed_seed('HF · hydrogen fluoride')
    atom_labels=['F1:2s','F1:sp3-hybrid-1','F1:sp3-hybrid-2',
                 'F1:sp3-hybrid-3','F1:sp3-hybrid-4']
    b=np.eye(5)
    result=SimpleNamespace(mol=FakeMol(5,['F','H']),local_basis=SimpleNamespace(
        B=b,labels=atom_labels))
    ref,meta=mf['lone_pair_ideal_reference'](result,seed,0,b[:,2])
    assert np.isclose(abs(ref[2]),1.)
    assert meta['ideal_s_fraction']==pytest.approx(.25)
    assert meta['ideal_p_fraction']==pytest.approx(.75)
