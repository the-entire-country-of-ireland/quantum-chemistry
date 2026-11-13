"""Opt-in real-molecule smoke tests for the rebuilt MO/GVB page.

Requires PySCF and may take several minutes. No claim of having run this test
should be made unless PySCF was available in the execution environment.
"""
import os
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(
    os.environ.get('ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION')!='1',
    reason='Opt in with ORBITAL_STUDIO_RUN_PYSCF_INTEGRATION=1 (requires PySCF).')


def setup_molecule(name):
    pytest.importorskip('pyscf')
    from qchem_analysis.fixed_geometry_data import fixed_seed
    from qchem_analysis.molecular_fixed import single_point,bond_orbital_map,localize_bond_ibos
    seed,_=fixed_seed(name)
    result=single_point(seed)
    localized=localize_bond_ibos(result)
    matches=bond_orbital_map(result,seed,localized)
    overlap=result.mol.intor('int1e_ovlp')
    assert np.allclose(localized.T@overlap@localized,np.eye(localized.shape[1]),atol=2e-5)
    assert np.isfinite(result.energy_hartree)
    assert matches
    return seed,result,localized,matches


def test_h2_two_pictures_and_energy_variational_order():
    from qchem_analysis.molecular_hf import decomposition
    from qchem_analysis.bond_selection import two_site_bond_functions,bond_sites
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        make_rhf_wavefunction,optimize_gvb_pp,bond_pair_orbitals)
    seed,result,localized,matches=setup_molecule('H₂ · hydrogen')
    assert len(matches)==1
    item=matches[0];bond=bond_sites(seed)[item.bond_index]
    projection=decomposition(result,result.valence_spatial_indices[item.orbital_index],
                             'atomic',coefficients_ao=localized[:,item.orbital_index])
    left,right=two_site_bond_functions(result,projection,
        f'{seed.symbols[bond["a"]]}{bond["a"]+1}',
        f'{seed.symbols[bond["b"]]}{bond["b"]+1}')
    ref=PairReference(item.orbital_index,item.label,localized[:,item.orbital_index],
                      left,right,('H1','H2'))
    space=construct_pair_space(result,localized,[ref]);ham=integral_hamiltonian(result,space)
    rhf=make_rhf_wavefunction(ham);gvb=optimize_gvb_pp(ham,space,maxiter=65)
    u,v,overlap=bond_pair_orbitals(space,gvb,0)
    S=result.mol.intor('int1e_ovlp')
    assert np.isclose(u@S@u,1.,atol=1e-5)
    assert np.isclose(v@S@v,1.,atol=1e-5)
    assert gvb.energy<=rhf.energy+1e-6
    assert -1.0001<=overlap<=1.0001
    assert np.isclose(gvb.occupations[0].sum(),2.,atol=1e-8)


def test_beh2_two_pairs_are_optimized_simultaneously():
    from qchem_analysis.molecular_hf import decomposition
    from qchem_analysis.bond_selection import bond_sites,two_site_bond_functions
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        make_rhf_wavefunction,optimize_gvb_pp)
    seed,result,localized,matches=setup_molecule('BeH₂ · beryllium hydride')
    refs=[];bonds=bond_sites(seed)
    for item in matches[:2]:
        bond=bonds[item.bond_index]
        p=decomposition(result,result.valence_spatial_indices[item.orbital_index],
                        'atomic',coefficients_ao=localized[:,item.orbital_index])
        a=f'{seed.symbols[bond["a"]]}{bond["a"]+1}'
        b=f'{seed.symbols[bond["b"]]}{bond["b"]+1}'
        left,right=two_site_bond_functions(result,p,a,b)
        refs.append(PairReference(item.orbital_index,item.label,
            localized[:,item.orbital_index],left,right,(a,b)))
    assert len(refs)==2
    space=construct_pair_space(result,localized,refs)
    ham=integral_hamiltonian(result,space)
    gvb=optimize_gvb_pp(ham,space,maxiter=65)
    assert gvb.energy<=make_rhf_wavefunction(ham).energy+1e-6
    assert gvb.occupations.shape==(2,2)


def test_methane_sp3_ideal_and_orbital_projection():
    from qchem_analysis.molecular_fixed import ideal_site_orbital,hybrid_comparison
    seed,result,localized,matches=setup_molecule('CH₄ · methane')
    assert len(matches)>=4
    bond=next(b for b in seed.rdkit_mol.GetBonds() if b.GetBeginAtomIdx()==0)
    neighbor=bond.GetEndAtomIdx()
    orbital,meta=ideal_site_orbital(result,seed,0,neighbor,'σ')
    assert meta['hybrid']=='sp³'
    assert np.isclose(meta['ideal_s_fraction'],.25)
    S=result.mol.intor('int1e_ovlp')
    assert np.isclose(orbital@S@orbital,1.,atol=1e-6)


def test_hf_bond_population_is_fluorine_polarized_after_gvb():
    """End-to-end HF check; requires the real PySCF 6-31G Hamiltonian.

    In a valid HF bond assignment the 2e pair is not concentrated on H.
    This test protects against both lone-pair-to-bond misassignment and
    an unphysical/incorrectly labeled GVB active-space result.
    """
    from qchem_analysis.molecular_fixed import atomic_iao_populations
    from qchem_analysis.molecular_hf import decomposition
    from qchem_analysis.bond_selection import two_site_bond_functions, bond_sites
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        optimize_gvb_pp,make_rhf_wavefunction,bond_pair_orbitals)
    seed,result,localized,matches=setup_molecule('HF · hydrogen fluoride')
    assert len(matches)==1
    item=matches[0]
    bond=bond_sites(seed)[item.bond_index]
    coeff=localized[:,item.orbital_index]
    atom_pop=atomic_iao_populations(result,coeff)
    assert atom_pop['F1']>atom_pop['H2'], (
        'HF MO was assigned to a hydrogen-dominated orbital',atom_pop)
    projection=decomposition(result,result.valence_spatial_indices[item.orbital_index],
                             'atomic',coefficients_ao=coeff)
    a,b=two_site_bond_functions(result,projection,'F1','H2')
    ref=PairReference(item.orbital_index,item.label,coeff,a,b,('F1','H2'))
    space=construct_pair_space(result,localized,[ref])
    ham=integral_hamiltonian(result,space)
    gvb=optimize_gvb_pp(ham,space,maxiter=130)
    assert gvb.energy <= make_rhf_wavefunction(ham).energy+1e-6
    v,w,_=bond_pair_orbitals(space,gvb,0)
    for orbital in (v,w):
        overlap=result.mol.intor('int1e_ovlp')
        assert np.isclose(orbital@overlap@orbital,1,atol=1e-5)
    nat=space.orbitals_ao@gvb.frame[:,:2]
    pair_pop={key:sum(float(occ)*atomic_iao_populations(result,nat[:,j]).get(key,0.)
                      for j,occ in enumerate(gvb.occupations[0]))
              for key in ('F1','H2')}
    assert pair_pop['F1']>pair_pop['H2'], (
        'GVB pair is implausibly more hydrogen-centered than fluorine',pair_pop)


def test_hf_and_water_lone_pair_catalog_real_iaos():
    """Opt-in real PySCF verification for nonbonding occupied orbital assignment."""
    from qchem_analysis.molecular_fixed import localized_pair_catalog,atomic_iao_populations
    for name,expected_bonds,expected_lone_pairs in (
        ('HF · hydrogen fluoride',1,3),
        ('H₂O · water',2,2),
    ):
        seed,result,localized,bond_matches=setup_molecule(name)
        catalog=localized_pair_catalog(result,seed,localized,bond_matches)
        pairs=[item for item in catalog if item.kind=='lone pair']
        bonds=[item for item in catalog if item.kind in ('σ','π')]
        assert len(pairs)==expected_lone_pairs, (
            name,[(x.label,x.kind,x.confidence) for x in catalog])
        assert len(bonds)==expected_bonds, (
            name,[(x.label,x.kind,x.confidence) for x in catalog])
        assert {p.orbital_index for p in pairs}.isdisjoint(
            {b.orbital_index for b in bonds})
        for p in pairs:
            assert p.atom_index==0, (name,p)
            weights=atomic_iao_populations(result,localized[:,p.orbital_index])
            assert weights[f'{seed.symbols[0]}1']>=.60, (name,p,weights)


def test_hf_lone_pair_gvb_partner_and_variational_energy():
    """Verify same-atom partner construction against the real F/H 6-31G basis."""
    from qchem_analysis.molecular_fixed import localized_pair_catalog,lone_pair_virtual_seed
    from qchem_analysis.perfect_pairing import (
        PairReference,construct_pair_space,integral_hamiltonian,
        make_rhf_wavefunction,optimize_gvb_pp)
    seed,result,localized,bonds=setup_molecule('HF · hydrogen fluoride')
    catalog=localized_pair_catalog(result,seed,localized,bonds)
    pair=next(item for item in catalog if item.kind=='lone pair')
    occupied=localized[:,pair.orbital_index]
    virtual=lone_pair_virtual_seed(result,occupied,pair.atom_index)
    overlap=result.mol.intor('int1e_ovlp')
    assert np.isclose(virtual@overlap@virtual,1.,atol=1e-6)
    assert abs(occupied@overlap@virtual)<1e-6
    reference=PairReference(pair.orbital_index,pair.label,occupied,occupied,
                            -virtual,('F1','F1'))
    space=construct_pair_space(result,localized,[reference])
    ham=integral_hamiltonian(result,space)
    pp=optimize_gvb_pp(ham,space,maxiter=90)
    assert pp.energy<=make_rhf_wavefunction(ham).energy+1e-6
    assert np.isclose(pp.occupations[0].sum(),2.)
