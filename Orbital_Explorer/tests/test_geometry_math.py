"""Pure numerical smoke tests (do not require PySCF)."""
from pathlib import Path
import ast
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from rdkit import Chem

from qchem_analysis.molecular_geometry import geometry_from_smiles


def _hybrid_helpers():
    # Avoid a PySCF import so the geometry/math checks can run separately.
    path=Path(__file__).resolve().parents[1] / 'qchem_analysis' / 'molecular_hf.py'
    tree=ast.parse(path.read_text())
    names={'_templates','_match_hybrid_directions','_hybrid_transform'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'np':np,'Rotation':Rotation,'permutations':itertools.permutations}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


def test_hybrid_transform_orthogonal_and_complete():
    h=_hybrid_helpers()
    for m in (2,3,4):
        dirs=h['_templates'](m)
        unitary, names=h['_hybrid_transform'](dirs)
        assert unitary.shape == (4,4)
        assert len(names) == 4
        assert np.allclose(unitary.T @ unitary, np.eye(4), atol=1e-12)


def test_hybrids_align_to_bond_axis():
    h=_hybrid_helpers()
    rotation=Rotation.from_euler('xyz',[.3,.2,-.4]).as_matrix()
    for m in (2,3,4):
        src=h['_templates'](m)
        for k in range(1,m+1):
            target=(src @ rotation.T)[:k]
            aligned=h['_match_hybrid_directions'](src,target)
            assert np.allclose(aligned[:k],target,atol=1e-8)


def test_vsepr_guess_reasonable():
    for smiles, center, targets in [('O',0,(102.,107.)),('N',0,(105.,109.)),
                                   ('C',0,(108.,111.)),('O=C=O',1,(176.,181.)),
                                   ('C=O',0,(117.,123.))]:
        seed=geometry_from_smiles(smiles)
        atom=seed.rdkit_mol.GetAtomWithIdx(center)
        n=[x.GetIdx() for x in atom.GetNeighbors()]
        u=seed.xyz_angstrom[n[0]]-seed.xyz_angstrom[center]
        v=seed.xyz_angstrom[n[1]]-seed.xyz_angstrom[center]
        angle=np.degrees(np.arccos(np.clip((u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
        assert targets[0]<angle<targets[1],(smiles,angle)
