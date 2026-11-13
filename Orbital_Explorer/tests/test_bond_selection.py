import numpy as np
from types import SimpleNamespace
from rdkit import Chem
from qchem_analysis.bond_selection import bond_sites,two_site_bond_functions


def test_two_site_ingredients_preserve_signed_terms():
    proj=dict(labels=['C1:2s','C1:2px','C2:2s','C2:2px'],
              components_ao=np.diag([.3,.4,.5,.6]))
    left,right=two_site_bond_functions(None,proj,'C1','C2')
    assert np.allclose(left,[.3,.4,0,0])
    assert np.allclose(right,[0,0,.5,.6])


def test_bond_list_uses_actual_rdkit_connectivity():
    mol=Chem.AddHs(Chem.MolFromSmiles('O'))
    seed=SimpleNamespace(rdkit_mol=mol,symbols=['O','H','H'])
    bonds=bond_sites(seed)
    assert len(bonds)==2
    assert {b['name'] for b in bonds}=={'O1–H2','O1–H3'}
