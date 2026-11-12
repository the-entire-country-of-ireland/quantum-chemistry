"""Check the geometry-aware basis labels without importing PySCF."""
import ast
from pathlib import Path
import numpy as np
from qchem_analysis.molecular_geometry import geometry_from_smiles

src = Path(__file__).resolve().parents[1] / 'qchem_analysis' / 'molecular_hf.py'
module = ast.parse(src.read_text())
func = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_geometry_hybrid_count')
scope = {'np': np}
exec(compile(ast.Module(body=[func], type_ignores=[]), str(src), 'exec'), scope)
choose = scope['_geometry_hybrid_count']


def chosen(smiles, atom_id):
    seed = geometry_from_smiles(smiles)
    return choose(seed.rdkit_mol.GetAtomWithIdx(atom_id), seed.xyz_angstrom)


def test_terminal_fluorine_hf_sp():
    assert chosen('F', 0) == 2


def test_methane_carbon_sp3():
    assert chosen('C', 0) == 4


def test_water_oxygen_sp3():
    assert chosen('O', 0) == 4


def test_ammonia_nitrogen_sp3():
    assert chosen('N', 0) == 4


def test_co2_center_sp():
    assert chosen('O=C=O', 1) == 2


def test_formaldehyde_carbon_sp2():
    assert chosen('C=O', 0) == 3


def test_ethylene_carbon_sp2():
    assert chosen('C=C', 0) == 3


def test_acetylene_carbon_sp():
    assert chosen('C#C', 0) == 2
