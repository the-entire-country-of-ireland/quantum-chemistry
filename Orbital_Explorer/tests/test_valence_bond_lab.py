import numpy as np
from types import SimpleNamespace
from qchem_analysis.valence_bond_lab import (
    atomic_hybrid_character, bond_sites, two_site_bond_functions,
    mixed_bond_orbital, selected_pair_active_space,
)


def test_hybrid_character_from_sp_projection():
    labels=['C1:2s','C1:2px','C1:2py','C1:2pz','H2:1s']
    coeffs=np.array([0.5, np.sqrt(.75),0,0,0.2])
    data=dict(labels=labels, coefficients=coeffs,gram=np.eye(5),probability_represented=.9)
    rows=atomic_hybrid_character(data)
    row=next(r for r in rows if r['Atom']=='C1')
    assert np.isclose(row['s character'],.25)
    assert np.isclose(row['Effective sp^n'],3.)
    assert np.allclose([row['p direction x'],row['p direction y'],row['p direction z']],[1,0,0])


def test_two_site_sum_and_mix():
    S=np.eye(4)
    projection=dict(labels=['C1:2s','C1:2px','C2:2s','C2:2px'],
                    components_ao=np.diag([.3,.4,.5,.6]))
    left,right=two_site_bond_functions(None,projection,'C1','C2')
    assert np.allclose(left,[.3,.4,0,0])
    assert np.allclose(right,[0,0,.5,.6])
    trial=mixed_bond_orbital(left,right,S,45,1)
    assert np.isclose(trial@S@trial,1.)
    other=mixed_bond_orbital(left,right,S,45,-1)
    assert np.isclose(other@S@other,1.)
    assert np.isclose(trial@other,0.)


def test_pair_active_space_orthonormal():
    # The bonding function is occupied; its opposite-sign atomic combination
    # is virtual, as it should be for H2 in the two-orbital limit.
    result=SimpleNamespace(mode='full', nocc=1,
       coefficients_ao=np.eye(4),
       core_spatial_count=0,
       mol=SimpleNamespace(intor=lambda _:np.eye(4)))
    projection={'labels':['H1:1s','H2:1s'],
                'components_ao':np.array([[.5,.5],[.5,-.5],[0.,0.],[0.,0.]])}
    C=np.eye(4)[:,:1]
    mo,partner=selected_pair_active_space(result,C,0,projection,'H1','H2')
    assert np.allclose(mo.T@mo,np.eye(4))
    assert np.isclose(abs(partner[1]),1.)
