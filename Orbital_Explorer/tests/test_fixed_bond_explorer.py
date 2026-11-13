"""Regressions for new fixed-geometry, bond-only MO/GVB workflows.

These tests intentionally run WITHOUT PySCF or Streamlit installed. Molecular
runtime integration is separately gated behind an opt-in PySCF test.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rdkit import Chem

from qchem_analysis.fixed_geometry_data import (
    fixed_molecule_presets, fixed_seed, parse_xyz, bond_angles,
)

ROOT=Path(__file__).resolve().parents[1]


def extract(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


def test_all_presets_have_consistent_fixed_nuclear_coordinates():
    items=fixed_molecule_presets()
    assert len(items)>=12
    for name,entry in items.items():
        seed,origin=fixed_seed(name)
        assert len(seed.symbols)==len(entry['symbols'])
        assert seed.xyz_angstrom.shape==(len(seed.symbols),3)
        assert np.all(np.isfinite(seed.xyz_angstrom))
        assert 'fixed' in origin.lower()
        xyz=seed.xyz_angstrom
        for bond in seed.rdkit_mol.GetBonds():
            d=np.linalg.norm(xyz[bond.GetBeginAtomIdx()]-xyz[bond.GetEndAtomIdx()])
            assert .55<d<2.35,(name,bond.GetIdx(),d)


def test_key_small_molecule_geometries():
    h2,_=fixed_seed('H₂ · hydrogen')
    r=np.linalg.norm(np.diff(h2.xyz_angstrom,axis=0))
    assert r==pytest.approx(.7414,abs=.0001)
    beh2,_=fixed_seed('BeH₂ · beryllium hydride')
    assert len(beh2.rdkit_mol.GetBonds())==2
    assert sorted(beh2.symbols)==['Be','H','H']
    for b in beh2.rdkit_mol.GetBonds():
        d=np.linalg.norm(beh2.xyz_angstrom[b.GetBeginAtomIdx()]-beh2.xyz_angstrom[b.GetEndAtomIdx()])
        assert d==pytest.approx(1.326,abs=.00001)
    angles=bond_angles(beh2,beh2.xyz_angstrom)
    assert angles[0]['Degrees (°)']==pytest.approx(180.,abs=.001)
    methane,_=fixed_seed('CH₄ · methane')
    for item in bond_angles(methane,methane.xyz_angstrom):
        assert item['Degrees (°)']==pytest.approx(109.471,abs=.002)
    water,_=fixed_seed('H₂O · water')
    assert bond_angles(water,water.xyz_angstrom)[0]['Degrees (°)']==pytest.approx(104.45,abs=.002)


def test_custom_geometry_is_explicit_and_never_optimized():
    xyz='3\ncustom water\nO 0 0 0\nH .7 0 .6\nH -.7 0 .6'
    seed,source=fixed_seed('Custom SMILES + XYZ',smiles='O',xyz_text=xyz)
    assert source=='user-supplied fixed XYZ geometry'
    assert sorted(np.linalg.norm(seed.xyz_angstrom-seed.xyz_angstrom[0],axis=1))[1]>0.8
    assert 'O'==seed.symbols[0]
    with pytest.raises(ValueError,match='require XYZ'):
        fixed_seed('Custom SMILES + XYZ',smiles='O',xyz_text='')
    with pytest.raises(ValueError,match='XYZ row'):
        fixed_seed('Custom SMILES + XYZ',smiles='O',xyz_text='O 0 0 0\nO 0 0 1\nH 0 1 0')


def test_invalid_xyz_rejected():
    for bad in ['2\nblank\nH 0 0 0\nH 1 0 0',
                'O 0 0 0\nH 0 0 nan\nH 1 0 0',
                'O 0 0 0\nH .01 0 0\nH 1 0 0']:
        with pytest.raises(ValueError):parse_xyz(bad,['O','H','H'])


def test_vsepr_hybrid_has_correct_sp_fraction_for_beh2():
    """Synthetic orthonormal IAO basis: Be 2s/p and H 1s at each end."""
    from qchem_analysis.fixed_geometry_data import fixed_seed
    p=ROOT/'qchem_analysis/molecular_fixed.py'
    namespace={'np':np}
    namespace['_valence_shell']=extract(p,'_valence_shell',namespace)
    namespace['iao_site_s_p']=extract(p,'iao_site_s_p',namespace)
    ideal=extract(p,'ideal_site_orbital',namespace)
    seed,_=fixed_seed('BeH₂ · beryllium hydride')
    class Mol:
        def intor(self,tag):return np.eye(6)
        def atom_charge(self,i):return [4,1,1][i]
        def atom_symbol(self,i):return ['Be','H','H'][i]
    labels=['Be1:1s','Be1:2s','Be1:2px','Be1:2py','Be1:2pz','H2:1s']
    result=SimpleNamespace(mol=Mol(),local_basis=SimpleNamespace(B_atomic=np.eye(6),atomic_labels=labels))
    phi,meta=ideal(result,seed,0,1,'σ')
    assert meta['hybrid']=='sp'
    assert meta['ideal_s_fraction']==pytest.approx(.5)
    assert np.dot(phi,phi)==pytest.approx(1.)
    assert phi[1]**2==pytest.approx(.5)
    assert np.dot(phi[2:5],phi[2:5])==pytest.approx(.5)


def test_one_to_one_bond_assignments_are_deterministic():
    p=ROOT/'qchem_analysis/molecular_fixed.py'
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class BondOrbital:
        orbital_index:int;bond_index:int;label:str;confidence:float;kind:str
    B=[{'index':0,'name':'A1–B2'},{'index':1,'name':'B2–C3'}]
    namespace={'np':np,'BondOrbital':BondOrbital,'RHFResult':object,
        'linear_sum_assignment':__import__('scipy.optimize',fromlist=['linear_sum_assignment']).linear_sum_assignment,
        'bond_sites':lambda seed:B,
        'bond_trials':lambda result,seed:[('A1–B2 σ',np.array([1.,0.])),('B2–C3 σ',np.array([0.,1.]))]}
    mapper=extract(p,'bond_orbital_map',namespace)
    result=SimpleNamespace(mol=SimpleNamespace(intor=lambda tag:np.eye(2)))
    out=mapper(result,None,np.eye(2))
    assert [(v.bond_index,v.orbital_index) for v in out]==[(0,0),(1,1)]
    assert [v.confidence for v in out]==[pytest.approx(1.),pytest.approx(1.)]


def test_molecular_page_has_only_two_pictures_and_no_geometry_optimizer():
    source=(ROOT/'pages/molecular.py').read_text()
    assert 'Bond Formation Lab' not in source
    assert 'Valence bond explorer' not in source
    assert "('MO (one doubly occupied IBO)','GVB-PP (all-orbital GVB-SCF)')" in source
    assert 'Optimize geometry' not in source and 'optimize_reduced' not in source
    assert "Boys localized" not in source
    assert "Pipek" not in source
    assert 'AtomicLibrary.prepare' not in source
    assert 'CASCI' not in source
    assert "6-31g" not in source # one backend constant, not an exposed choice


def test_pyscf_minao_ao_tuple_labels_keep_p_orientation():
    path=ROOT/'qchem_analysis/molecular_fixed.py'
    func=extract(path,'_shell_label',{})
    assert func((0,'C','2s',''))=='2s'
    assert func((0,'C','2p','x'))=='2px'
    assert func((0,'C','2p','y'))=='2py'
    assert func((0,'C','2p','z'))=='2pz'
