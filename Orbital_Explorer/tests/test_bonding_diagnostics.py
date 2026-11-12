"""Pure numerical tests for bond-analysis diagnostics; no PySCF required."""
from types import SimpleNamespace
import numpy as np
import pytest
from qchem_analysis.bonding_analysis import (
    pair_overlap_matrix, overlap_analysis, atom_pair_overlap,
    validate_density_preservation, plot_overlap_heatmap,
)
from qchem_analysis.molecular_geometry import align_molecular_axes, geometry_from_smiles


def projection():
    # Non-orthogonal local functions with an explicitly normalized MO.
    raw = np.array([.65, -.35, .42, .1])
    S = np.array([[1.,.20,.14,.0],
                  [.20,1.,.0,.08],
                  [.14,.0,1.,.15],
                  [.0,.08,.15,1.]])
    a = raw / np.sqrt(raw @ S @ raw)
    return {'labels':['C1:2s','C1:2px','H2:1s','H2:1p'],
            'coefficients':a,'gram':S}


def test_signed_overlap_terms_reconstruct_norm():
    data=projection()
    matrix=pair_overlap_matrix(data['coefficients'],data['gram'])
    analysis=overlap_analysis(data)
    assert abs(matrix.sum()-1)<1e-13
    assert np.isclose(analysis['self_terms']+analysis['cross_terms'],1.)
    assert all(r['Interatomic'] for r in analysis['top_interatomic'])
    assert len(analysis['top_interatomic'])==4
    i,j=0,2
    row=next(r for r in analysis['top_interatomic']
             if {r['Component A'],r['Component B']}=={'C1:2s','H2:1s'})
    assert row['Contribution to norm']==pytest.approx(2*matrix[i,j])
    atoms, grouped=atom_pair_overlap(data)
    assert atoms==['C1','H2']
    assert grouped.shape==(2,2)
    assert np.isclose(grouped.sum(),1.)


def test_heatmap_is_meaningful():
    data=projection()
    for grouped in [True,False]:
        fig=plot_overlap_heatmap(data,grouped)
        assert fig.data[0].type=='heatmap'
        if not grouped:
            assert np.any(np.array(fig.data[0].z)<0)
        assert np.isclose(np.array(fig.data[0].z).sum(),1.)


def test_occupied_rotation_preserves_density_and_fock_block_spectrum():
    rng=np.random.default_rng(9)
    n=6; nval=3
    A=rng.normal(size=(n,n))
    S=A.T@A+np.eye(n)
    e,U=np.linalg.eigh(S)
    X=(U*e**-.5)@U.T
    Q,_=np.linalg.qr(rng.normal(size=(n,nval)))
    C=X@Q
    R,_=np.linalg.qr(rng.normal(size=(nval,nval)))
    rotated=C@R
    class Mol:
        def intor(self,name):
            assert name=='int1e_ovlp'
            return S
    obj=SimpleNamespace(mol=Mol(),coefficients_ao=C,valence_spatial_indices=list(range(nval)))
    assert validate_density_preservation(obj,rotated)<1e-12
    F0=np.diag([-1.7,-0.4,0.1])
    F1=R.T@F0@R
    assert np.allclose(np.linalg.eigvalsh(F1),np.diag(F0))
    assert not np.allclose(F1,np.diag(np.diag(F1)))


def test_molecular_frame_keeps_linear_acetylene_on_x():
    seed=geometry_from_smiles('C#C')
    aligned=align_molecular_axes(seed.xyz_angstrom,seed.rdkit_mol)
    # Geometry preservation allows the small departures from exact linearity
    # that remain after VSEPR/RDKit embedding; the frame itself is x-aligned.
    assert np.max(np.abs(aligned[:,1:]))<.025
    assert np.allclose(np.linalg.norm(aligned[:,None]-aligned[None,:],axis=2),
                       np.linalg.norm(seed.xyz_angstrom[:,None]-seed.xyz_angstrom[None,:],axis=2),atol=1e-8)


def test_localized_spin_export_contains_two_spins_per_spatial_orbital():
    from io import BytesIO
    from qchem_analysis.bonding_analysis import localized_spin_archive
    obj=SimpleNamespace(mode='full',xyz_angstrom=np.zeros((3,3)),
                        energy_hartree=-100., frozen_core=True)
    C=np.array([[1.,0.],[0.,1.],[0.,0.]])
    F=np.array([[-.7,.05],[.05,-.4]])
    data=localized_spin_archive(obj,C,[('bond 1',.9),('bond 2',.8)],'IBO',fock_matrix=F)
    with np.load(BytesIO(data),allow_pickle=False) as stored:
        assert stored['valence_spin_coefficients_ao'].shape==(3,4)
        assert stored['valence_spin'].tolist()==['alpha','beta','alpha','beta']
        assert stored['valence_spin_spatial_indices'].tolist()==[0,0,1,1]
        assert np.allclose(stored['occupied_fock_matrix_hartree'],F)
        assert np.array_equal(stored['valence_spin_coefficients_ao'][:,0],C[:,0])
