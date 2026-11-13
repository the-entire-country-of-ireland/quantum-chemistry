"""Orbital Studio molecular page: one geometry, one IBO map, two bonding views.

No geometry optimizer, basis selector, alternative localizers, or overlapping labs.
For molecular bonds, choose either a doubly occupied IBO (MO) or two optimized
spin-coupled spatial orbitals (GVB-PP). Atomic mode remains in pages/atomic.py.
"""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qchem_analysis.molecular_fixed import (
    BASIS, fixed_molecule_presets, fixed_seed, single_point, bond_angles, localize_bond_ibos,
    bond_orbital_map, localized_pair_catalog, lone_pair_virtual_seed,
    lone_pair_ideal_reference, ideal_site_orbital, hybrid_comparison, atomic_iao_populations,
)
from qchem_analysis.molecular_hf import decomposition
from qchem_analysis.bonding_analysis import validate_density_preservation
from qchem_analysis.bond_selection import (
    bond_sites, two_site_bond_functions, bond_map_figure,
)
from qchem_analysis.perfect_pairing import (
    PairReference, construct_pair_space, integral_hamiltonian,
    optimize_gvb_pp,
)
from qchem_analysis.gvb_scf import (
    construct_gvb_scf_problem, optimize_full_gvb_scf,
    full_pair_orbitals as bond_pair_orbitals,
)
from qchem_analysis.perfect_pairing import PairWavefunction
from qchem_analysis.molecular_viewer import plot_combined_with_bonds, plot_local_ingredients

st.markdown('''<style>
.block-container{max-width:1420px;padding-top:1.3rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.025em}
</style>''',unsafe_allow_html=True)


@st.cache_resource(max_entries=10,show_spinner=False)
def calculated_molecule(preset:str, custom_smiles:str, xyz_text:str, charge:int):
    """Heavy RHF/IAO/IBO work is cached independently of the plotting widgets."""
    seed,geometry_origin=fixed_seed(preset,smiles=custom_smiles,xyz_text=xyz_text)
    result=single_point(seed,int(charge))
    localized=localize_bond_ibos(result)
    delta=validate_density_preservation(result,localized)
    if delta>3e-5:
        raise RuntimeError(f'IBO localization altered RHF density (error={delta:.3g}).')
    bonds=bond_orbital_map(result,seed,localized)
    catalog=localized_pair_catalog(result,seed,localized,bonds)
    return seed,result,localized,catalog,geometry_origin


@st.cache_resource(max_entries=20,show_spinner=False)
def calculated_pairs(preset:str,custom_smiles:str,xyz_text:str,charge:int,
                     assignments:tuple[tuple[int,int],...]):
    """All-space many-pair orbital relaxation, with other pairs doubly occupied."""
    seed,result,localized,mapping,origin=calculated_molecule(
        preset,custom_smiles,xyz_text,charge)
    bonds=bond_sites(seed)
    label_map={(m.orbital_index,m.bond_index if m.bond_index is not None else -1):m.label
               for m in mapping}
    refs=[]
    valid={(entry.orbital_index,entry.bond_index if entry.bond_index is not None else -1):entry
           for entry in mapping if entry.kind!='unassigned'}
    for orb,bidx in assignments:
        if (orb,bidx) not in valid:
            raise ValueError('Invalid or unassigned selected occupied pair.')
        entry=valid[(orb,bidx)]
        if entry.kind=='lone pair':
            site=int(entry.atom_index)
            site_name=f'{seed.symbols[site]}{site+1}'
            # Unlike an internuclear bond, a lone pair is centered on ONE atom.
            # The second component seeds a virtual of matching local s/p shape;
            # it is not a fictitious second atomic site or an HL bond structure.
            left=localized[:,orb]
            right=-lone_pair_virtual_seed(result,left,site)
            refs.append(PairReference(orb,entry.label,left,left,right,
                                      (site_name,site_name)))
            continue
        bond=bonds[bidx]
        a_name=f'{seed.symbols[bond["a"]]}{bond["a"]+1}'
        b_name=f'{seed.symbols[bond["b"]]}{bond["b"]+1}'
        ingredients=decomposition(result,result.valence_spatial_indices[orb],
                                  'atomic',coefficients_ao=localized[:,orb])
        left,right=two_site_bond_functions(result,ingredients,a_name,b_name)
        refs.append(PairReference(orb,label_map.get((orb,bidx),f'{a_name}–{b_name}'),
                                  localized[:,orb],left,right,(a_name,b_name)))
    space=construct_pair_space(result,localized,refs)
    # The small pair space supplies chemically sensible initial orbitals ONLY.
    # The final SCF objective contains the COMPLETE molecular AO basis, all
    # occupied electrons and all virtual orbitals; there is no frozen core.
    full=construct_gvb_scf_problem(result,space)
    warm=None
    try:
        initial_ham=integral_hamiltonian(result,space)
        warm=optimize_gvb_pp(initial_ham,space,rotate_subspaces=False,maxiter=65)
    except (ValueError,RuntimeError,ArithmeticError):
        # Full-space RHF and weak-pairing starts remain available.
        pass
    optimized=optimize_full_gvb_scf(full,initial_pair_wave=warm,maxiter=155)
    return full,full,PairWavefunction('RHF',full.rhf_energy,
        np.eye(full.orbital_count),np.zeros(full.pair_count),
        np.column_stack([np.full(full.pair_count,2.),np.zeros(full.pair_count)]),
        True,'RHF orbital reference',0),optimized


def geometry_diagram(seed,selected_bond:int | None=None):
    fig=bond_map_figure(seed,seed.xyz_angstrom,selected_bond)
    fig.update_layout(height=300,showlegend=False,margin={'l':0,'r':0,'t':0,'b':0})
    return fig


def orbital_surface(result, seed, coef, title: str, render_mode: str = 'surface'):
    """Render one spatial orbital as a signed surface or a |psi|^2 point cloud.

    One molecular-level display choice applies identically to MO, both GVB
    orbitals, and ideal VSEPR references. Numerically meaningful plot defaults
    remain internal rather than becoming additional UI knobs.
    """
    if render_mode not in ('surface', 'points'):
        raise ValueError('render_mode must be surface or points')
    extent = float(np.max(np.abs(result.mol.atom_coords())) + 3.5)
    fig = plot_combined_with_bonds(
        result, 0, seed=seed, coefficients_ao=coef,
        mode=render_mode, grid_points=70,
        extent_bohr=max(5.0, extent), probability_coverage=.88,
        point_count=26000,
    )
    fig.update_layout(title=title, height=415,
                      margin={'l':0,'r':0,'t':48,'b':0}, showlegend=False,
                      # Preserve interactive camera position while changing
                      # the rendering of the same physical orbital.
                      scene={'uirevision':'molecular-orbital-view'})
    return fig


def component_table(result,coef,representation:str):
    proj=decomposition(result,0,representation,coefficients_ao=coef)
    ranked=np.argsort(proj['weights'])[::-1][:12]
    rows=[{'Atomic function':proj['labels'][i],
           'Signed coefficient':round(float(proj['coefficients'][i]),4),
           'Projection weight %':round(float(100*proj['weights'][i]),2)}
          for i in ranked if proj['weights'][i]>.002]
    return rows,proj['probability_represented']


def selected_orbital_summary(result,seed,localized,matches,space=None,wave=None):
    """Inexpensive all-bonds overview: no 3D surfaces or separate calculations."""
    bonds=bond_sites(seed)
    rows=[]
    for pair_index,item in enumerate(matches):
        ao=localized[:,item.orbital_index]
        if wave is None:
            orbitals=[('Bonding MO',ao)]
            occupations='2.000 / 0.000 (RHF)'
        else:
            u,v,_=bond_pair_orbitals(space,wave,pair_index)
            # Neither GVB pair orbital is forced to correspond to a different
            # atom: for an ionic bond BOTH can concentrate on the same site.
            orbitals=[('GVB pair orbital 1',u),('GVB pair orbital 2',v)]
            occ=wave.occupations[pair_index]
            occupations=f'{occ[0]:.3f} / {occ[1]:.3f}'
        for orbital_label,coef in orbitals:
            proj=decomposition(result,0,'hybrid',coefficients_ao=coef)
            inds=np.argsort(proj['weights'])[::-1]
            top=[f'{proj["labels"][k]} {proj["weights"][k]:.0%}'
                 for k in inds[:4] if proj['weights'][k]>.02]
            rows.append({'Occupied pair':item.label,'Displayed orbital':orbital_label,
                         'Leading atomic / hybrid ingredients':', '.join(top),
                         'IAO coverage':f'{proj["probability_represented"]:.0%}',
                         'Natural occupations (pair)':occupations})
    return rows


def orbital_breakdowns(result,seed,orbitals:list[tuple[str,np.ndarray]]):
    atom,hybrid=st.tabs(['Atomic s / p','VSEPR-aligned hybrids'])
    for container,style in ((atom,'atomic'),(hybrid,'hybrid')):
        with container:
            columns=st.columns(len(orbitals))
            for col,(name,coef) in zip(columns,orbitals):
                with col:
                    rows,coverage=component_table(result,coef,style)
                    st.markdown(f'**{name}**')
                    st.caption(f'IAO analysis-space coverage: {coverage:.1%} of orbital norm')
                    if coverage<.78:
                        st.warning('Large component outside the minimal atomic analysis space; coefficients are incomplete.')
                    if rows:
                        chart_rows=list(reversed(rows))
                        fig=go.Figure(go.Bar(
                            x=[row['Signed coefficient'] for row in chart_rows],
                            y=[row['Atomic function'] for row in chart_rows],
                            orientation='h',
                            marker_color=['#2878b5' if row['Signed coefficient']>=0 else '#d34a4a'
                                          for row in chart_rows],
                            customdata=[row['Projection weight %'] for row in chart_rows],
                            hovertemplate='%{y}<br>signed coefficient=%{x:.4f}<br>projection weight=%{customdata:.2f}%<extra></extra>'))
                        fig.update_layout(height=max(250,min(440,105+28*len(chart_rows))),
                            margin={'l':120,'r':18,'t':12,'b':40},
                            yaxis={'automargin':True},xaxis_title='Signed orbital amplitude')
                        st.plotly_chart(fig,use_container_width=True,key=f'decomp-{style}-{name}')
                        with st.expander('Numerical projection coefficients',expanded=False):
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('Phase-colored coefficients are wavefunction amplitudes, not spin assignments. '
               'IAO weights describe an orthogonalized atomic representation, not unique electron populations.')
    if st.toggle('Show individual signed orbital ingredients in 3D',value=False,
                 key='show_signed_ingredients'):
        mode=st.radio('Ingredient type',('Atomic s/p','VSEPR hybrids'),horizontal=True)
        representation='atomic' if mode=='Atomic s/p' else 'hybrid'
        columns=st.columns(len(orbitals))
        for col,(name,coef) in zip(columns,orbitals):
            with col:
                proj=decomposition(result,0,representation,coefficients_ao=coef)
                ranked=np.argsort(proj['weights'])[::-1][:4]
                ingredients=[proj['labels'][int(k)] for k in ranked
                             if proj['weights'][k]>.006]
                if not ingredients:
                    st.info('No significant atomic ingredients in this projection.')
                    continue
                try:
                    extent=max(5.,float(np.max(np.abs(result.mol.atom_coords()))+3.5))
                    fig=plot_local_ingredients(result,proj,ingredients,seed=seed,
                        extent_bohr=extent,grid_points=31,amplitude_cutoff=.10)
                    fig.update_layout(height=420,title=f'{name} · signed ingredients')
                    st.plotly_chart(fig,use_container_width=True,
                                    key=f'3d-ingredients-{representation}-{name}')
                except Exception as exc:
                    st.warning(f'Ingredient surfaces unavailable: {exc}')


def pair_atomic_populations(result,space,wave,pair_index):
    """Atomic populations of the TWO-ELECTRON pair density in orthogonal IAOs.

    Two spatial GVB orbitals are nonorthogonal and cannot be interpreted as
    distinguishable electrons assigned one to each atom. The natural orbital
    1-RDM of the geminal instead has occupations 2 cos²θ and 2 sin²θ. Its
    projection is the appropriate site population diagnostic (sums to 2 e
    modulo IAO-space incompleteness).
    """
    offset=int(getattr(space,'pair_offset',0))
    nat=space.orbitals_ao @ wave.frame[:,offset+2*pair_index:offset+2*pair_index+2]
    occupations=np.asarray(wave.occupations[pair_index],float)
    total={}
    for col,occ in zip(nat.T,occupations):
        for atom,weight in atomic_iao_populations(result,col).items():
            total[atom]=total.get(atom,0.)+float(occ*weight)
    return total


def pair_orbital_populations(result,orbital):
    """IAO-projected spatial orbital weights; NOT electron assignments."""
    return atomic_iao_populations(result,orbital)


def export_selected(seed,result,localized,mapping,orbitals,space=None,wave=None):
    stream=BytesIO()
    payload=dict(format_version='orbital-studio-mo-full-gvb-scf-2026-11',basis=BASIS,
        geometry_angstrom=result.xyz_angstrom,
        symbols=np.asarray(seed.symbols,dtype=str),
        ao_overlap=result.mol.intor('int1e_ovlp'),
        rhf_total_energy_hartree=result.energy_hartree,
        bond_names=np.asarray([m.label for m in mapping],dtype=str),
        bond_orbital_indices=np.asarray([m.orbital_index for m in mapping],int),
        ibo_valence_ao=localized,
        displayed_labels=np.asarray([x[0] for x in orbitals],dtype=str),
        displayed_orbitals_ao=np.column_stack([x[1] for x in orbitals]),
        iao_atomic_ao=result.local_basis.B_atomic,
        iao_atomic_labels=np.asarray(result.local_basis.atomic_labels,dtype=str))
    if space is not None and wave is not None:
        payload.update(gvb_pair_natural_occupations=wave.occupations,
                       gvb_total_energy_hartree=wave.energy,
                       gvb_converged=wave.converged,
                       gvb_reference_full_ao=space.orbitals_ao,
                       gvb_frame=wave.frame,
                       gvb_pair_offset=space.pair_offset,
                       gvb_theta_radians=wave.theta,
                       gvb_orbital_relaxation='all occupied, paired, virtual; spin-adapted PP',
                       inactive_ao=space.inactive_ao)
    np.savez_compressed(stream,**payload)
    return stream.getvalue()


st.title('Molecular electron-pair orbitals')
st.caption('Select bonds and lone pairs. MO shows occupied IBOs; GVB-SCF optimizes two singlet-coupled spatial orbitals per selected pair across the complete Gaussian orbital basis.')

preset_names=list(fixed_molecule_presets())+['Custom SMILES + XYZ']
with st.sidebar:
    st.header('Molecule')
    with st.form('molecular_input'):
        preset=st.selectbox('Molecule',preset_names,index=preset_names.index('CH₄ · methane'))
        if preset=='Custom SMILES + XYZ':
            smiles=st.text_input('SMILES connectivity',placeholder='O')
            xyz_text=st.text_area('XYZ coordinates (Å, in RDKit explicit-H atom order)',
                height=180,placeholder='O 0 0 0\nH 0.759 0 0.589\nH -0.759 0 0.589')
            charge=int(st.number_input('Molecular charge',value=0,min_value=-4,max_value=4,step=1))
        else:
            smiles='';xyz_text='';charge=0
        submitted=st.form_submit_button('Load molecular orbitals',type='primary',use_container_width=True)
    st.caption(f'One fixed-geometry RHF reference · {BASIS.upper()} · IBO localization')
    st.caption('Nuclear coordinates are never optimized. No separate atomic HF calculations are performed.')

if submitted:
    try:
        with st.spinner('Computing one RHF reference and IBO bond orbitals at the fixed geometry…'):
            calculated_molecule(preset,smiles,xyz_text,charge)
        st.session_state['loaded_molecule']=(preset,smiles,xyz_text,charge)
        st.session_state.pop('ready_pp',None)
        st.session_state.pop('selected_bond_orbitals',None)
        st.session_state.pop('inspect_selected_bond',None)
        st.session_state.pop('show_signed_ingredients',None)
        st.session_state.pop('show_ideal_vsepr',None)
    except Exception as exc:
        st.error(f'Molecular calculation failed: {exc}')

loaded=st.session_state.get('loaded_molecule')
if loaded is None:
    st.info('Choose a molecule and click **Load molecular orbitals**. Try HF for one polar bond and three F lone pairs, or H₂O for two bonds and two O lone pairs.')
    st.stop()

seed,result,localized,mapping,geometry_origin=calculated_molecule(*loaded)
if not mapping:
    st.error('No occupied valence IBOs found for this molecule.')
    st.stop()

with st.expander('Molecular geometry and calculation provenance',expanded=False):
    a,b=st.columns([2,3])
    with a:st.plotly_chart(geometry_diagram(seed),use_container_width=True)
    with b:
        st.write(f'**Geometry:** {geometry_origin}')
        st.write(f'**Electronic method:** RHF/{BASIS.upper()}, followed by IBO localization.')
        st.write(f'**RHF total energy:** {result.energy_hartree:.8f} Hartree (includes nuclear repulsion).')
        rows=[{'Bond':f'{seed.symbols[bd["a"]]}{bd["a"]+1}–{seed.symbols[bd["b"]]}{bd["b"]+1}',
               'Distance (Å)':round(float(np.linalg.norm(seed.xyz_angstrom[bd['a']]-seed.xyz_angstrom[bd['b']])),4)}
              for bd in bond_sites(seed)]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        angles=bond_angles(seed,seed.xyz_angstrom)
        if angles:
            st.dataframe(pd.DataFrame(angles),hide_index=True,use_container_width=True)

control_left,control_right=st.columns([1.3,2.4],gap='large')
with control_left:
    mode=st.radio('Bonding picture',('MO (one doubly occupied IBO)','GVB-PP (all-orbital GVB-SCF)'),
                  horizontal=False,key='bonding_picture')
    st.caption('Both pictures use the same atomic geometry and molecular Hamiltonian.')
with control_right:
    options=list(range(len(mapping)))
    descriptions=[f'{m.label} · IBO {m.orbital_index+1}' for m in mapping]
    selected=st.multiselect('Bonding orbitals and lone pairs',options,
        default=[i for i,item in enumerate(mapping) if item.kind in ('σ','π')][:4] or options[:2],
        format_func=lambda k:descriptions[k],key='selected_bond_orbitals',
        help='Choose bonds and/or localized nonbonding electron pairs. Each selected pair is included in the same GVB-PP optimization.')
    if any(mapping[k].kind in ('σ','π') and mapping[k].confidence<.22 for k in selected):
        st.warning('One or more bond-to-IBO assignments has low overlap confidence. Interpret their atomic decomposition carefully.')
    if any(mapping[k].kind=='unassigned' for k in selected):
        st.warning('One or more occupied IBOs could not be classified confidently as a bond or lone pair. They can be inspected in MO mode, but cannot be correlated in GVB-PP.')

if not selected:
    st.info('Select at least one occupied bond or lone-pair orbital.')
    st.stop()

if len(selected)>6:
    st.warning('At most six simultaneous pairs are supported in this interactive version.')
    st.stop()

selected_matches=[mapping[k] for k in selected]
assignments=tuple((m.orbital_index,m.bond_index if m.bond_index is not None else -1)
                  for m in selected_matches)
is_pp=mode.startswith('GVB-PP')
if is_pp and any(m.kind=='unassigned' for m in selected_matches):
    st.warning('Unassigned occupied IBOs have no chemically justified virtual partner. Select confirmed bonds or lone pairs for GVB-PP.')
    st.stop()
space=ham=reference=gvb=None
if is_pp:
    if st.button('Optimize selected singlet pairs',type='primary',key='run_pp'):
        try:
            with st.spinner(f'Full GVB-SCF: relaxing ALL molecular orbitals for {len(assignments)} correlated singlet pair(s)…'):
                calculated_pairs(*loaded,assignments)
            st.session_state['ready_pp']=(loaded,assignments)
        except Exception as exc:
            st.session_state.pop('ready_pp',None)
            st.error(f'GVB-PP optimization failed: {exc}')
    if st.session_state.get('ready_pp')!=(loaded,assignments):
        st.info('Click **Optimize selected singlet pairs** to generate the GVB-PP orbitals. No optimization runs during MO browsing or while switching the inspected bond.')
        st.stop()
    space,ham,reference,gvb=calculated_pairs(*loaded,assignments)
    a,b,c=st.columns(3)
    a.metric('RHF total energy',f'{reference.energy:.6f} Eh')
    b.metric('GVB-PP total energy',f'{gvb.energy:.6f} Eh')
    c.metric('Pairing energy lowering',f'{(reference.energy-gvb.energy)*1000:.2f} mEh')
    if not gvb.converged:
        st.warning('The optimizer did not confirm convergence. Displayed orbitals are the lowest-energy valid trial state found; inspect before interpreting.')
    st.caption('All-orbital GVB-SCF: selected singlet pairs optimize simultaneously against the complete occupied AND virtual molecular orbital space. Unselected orbitals remain doubly occupied but are optimized, including the core. Pair subspaces stay mutually strongly orthogonal and the spin pairing pattern is fixed (perfect pairing). These are total molecular energies, not individual bond energies.')
    overview=[]
    for i,item in enumerate(selected_matches):
        overlap=float(bond_pair_orbitals(space,gvb,i)[2])
        overview.append({'Selected electron pair':item.label,
                         'Major natural occupation':round(float(gvb.occupations[i,0]),4),
                         'Minor natural occupation':round(float(gvb.occupations[i,1]),4),
                         'Pair-orbital overlap':round(overlap,4)})
    st.dataframe(pd.DataFrame(overview),use_container_width=True,hide_index=True)
else:
    st.caption('Each selected bond or lone pair is a doubly occupied RHF IBO. Localization changes the orbital representation, not the RHF energy or electron density.')

if len(selected)>1:
    with st.expander('All selected electron pairs · atomic/hybrid overview',expanded=True):
        st.dataframe(pd.DataFrame(selected_orbital_summary(result,seed,localized,selected_matches,space,gvb)),
                     use_container_width=True,hide_index=True)
        st.caption('Ingredient percentages are relative to the captured minimal atomic analysis space; IAO coverage gives the represented fraction of the full spatial orbital. View one bond below for signed coefficients and its 3D wavefunction.')

if len(selected)>1:
    inspect=st.selectbox('Inspect selected electron pair',selected,
        format_func=lambda k:descriptions[k],key='inspect_selected_bond')
else:
    inspect=selected[0]

item=mapping[inspect]
orbital=localized[:,item.orbital_index]
is_lone=item.kind=='lone pair'
is_unassigned=item.kind=='unassigned'
if not (is_lone or is_unassigned):
    bond=bond_sites(seed)[item.bond_index]
    siteA,siteB=bond['a'],bond['b']
    idealA,metaA=ideal_site_orbital(result,seed,siteA,siteB,item.kind,orbital)
    idealB,metaB=ideal_site_orbital(result,seed,siteB,siteA,item.kind,orbital)
    atoms=(f'{seed.symbols[siteA]}{siteA+1}',f'{seed.symbols[siteB]}{siteB+1}')
else:
    siteA=siteB=item.atom_index
    atoms=(f'{seed.symbols[siteA]}{siteA+1}',)
    try:
        idealA,metaA=lone_pair_ideal_reference(result,seed,siteA,orbital)
    except ValueError:
        idealA=metaA=None
    idealB=metaB=None

st.subheader(item.label)
# Visualization is not part of the molecular calculation/cache key. Changing
# this merely redraws the existing orbitals, never reoptimizes GVB-SCF.
render_choice = st.radio(
    '3D orbital rendering', ('Isosurface', 'Probability points'),
    horizontal=True, key='molecular_orbital_render',
    help='Isosurfaces show equal-amplitude contours; points are sampled from '
         'the spatial probability density |psi(r)|^2. Blue/red indicate phase, not spin.',
)
render_mode = 'points' if render_choice == 'Probability points' else 'surface'
if is_lone:
    st.caption(f'Localized nonbonding pair on {atoms[0]} · occupied IBO {item.orbital_index+1}. '
               'The orbital is identified from its atomic IAO population; its hybrid orientation is not unique.')
elif is_unassigned:
    st.warning('This occupied IBO is not sufficiently localized to be classified as a lone pair. '
               'Its decomposition is shown without making a chemical assignment.')
else:
    st.caption(f'IBO bond-assignment overlap: {item.confidence:.1%} · '
               f'centers: {atoms[0]} and {atoms[1]}')
actual=[]
if not is_pp:
    actual=[('Localized lone-pair MO' if is_lone else 'Localized bonding MO' if not is_unassigned else 'Localized occupied MO',orbital)]
    raw_pop=atomic_iao_populations(result,orbital)
    st.markdown('**Atomic distribution of the doubly occupied orbital (IAO-projected)**')
    atom_names=tuple(f'{s}{i+1}' for i,s in enumerate(seed.symbols))
    st.dataframe(pd.DataFrame([{'Atom':atom,
        'Orbital probability (%)':round(100*raw_pop.get(atom,0.),1),
        'Doubly occupied electron population':round(2*raw_pop.get(atom,0.),3)}
        for atom in atom_names if raw_pop.get(atom,0.)>.002]),
        hide_index=True,use_container_width=True)
    st.caption('IAO projected populations, not formal oxidation states or ionic fractions.')
    if not (is_lone or is_unassigned) and {seed.symbols[siteA],seed.symbols[siteB]}=={'F','H'}:
        pf=sum(raw_pop.get(k,0.) for k in atoms if k.startswith('F'))
        ph=sum(raw_pop.get(k,0.) for k in atoms if k.startswith('H'))
        if ph>pf+.05:
            st.error('HF bond sanity check failed: the assigned bonding MO is more hydrogen-centered '
                     'than fluorine-centered. This is probably not the bonding IBO.')
            st.stop()
else:
    pair_index=selected.index(inspect)
    left,right,pair_overlap=bond_pair_orbitals(space,gvb,pair_index)
    # Full orbital relaxation can recouple the selected pair *spatially* with
    # occupied and virtual orbitals. Check continuity of its chemical label
    # against the original RHF IBO; never silently relabel a migrated pair.
    start=space.pair_offset+2*pair_index
    natural=space.orbitals_ao@gvb.frame[:,start:start+2]
    metric=np.asarray(result.mol.intor('int1e_ovlp'),float)
    retained=float(np.sum(.5*gvb.occupations[pair_index]*
        (orbital@metric@natural)**2))
    if retained<.23:
        st.warning(f'The optimized geminal has only {retained:.0%} weighted overlap '
                   'with the originally selected occupied IBO. All-space orbital '
                   'relaxation may have changed its chemical assignment; check '
                   'the atomic populations before calling it this bond/lone pair.')
    actual=[('GVB pair orbital 1',left),('GVB pair orbital 2',right)]
    st.caption(f'Two generally nonorthogonal spatial functions make one singlet pair. '
               f'Orbital overlap: {pair_overlap:.4f}. '
               'Lone-pair virtual partners start locally but can relax throughout the full molecular orbital space.')
    populations=pair_atomic_populations(result,space,gvb,pair_index)
    p1=pair_orbital_populations(result,left)
    p2=pair_orbital_populations(result,right)
    report=[]
    for atom in [f'{s}{i+1}' for i,s in enumerate(seed.symbols)]:
        if max(p1.get(atom,0.),p2.get(atom,0.),populations.get(atom,0.)/2)<.005:continue
        report.append({'Atom':atom,
                       'GVB orbital 1 (%)':round(100*p1.get(atom,0.),1),
                       'GVB orbital 2 (%)':round(100*p2.get(atom,0.),1),
                       'Pair electron population':round(populations.get(atom,0.),3)})
    st.markdown('**Atomic distribution of the optimized pair (IAO-projected)**')
    st.dataframe(pd.DataFrame(report),hide_index=True,use_container_width=True)
    coverage=2.-populations.get('unrepresented',0.)
    if coverage<1.5:
        st.warning('Less than 75% of the pair density is represented by the minimal IAO analysis space.')
    st.caption(f'Pair 1-RDM projected IAO population: {coverage:.3f}/2 electrons represented. '
               'These are projection populations, not formal charges.')
    if is_lone and populations.get(atoms[0],0.)<1.10:
        st.warning('The optimized pair is not predominantly on its assigned lone-pair atom. '
                   'The optimized full-space geminal may have changed its chemical assignment; do not interpret it as a validated localized lone pair.')
    if not (is_lone or is_unassigned) and {seed.symbols[siteA],seed.symbols[siteB]}=={'F','H'}:
        pf=sum(populations.get(k,0.) for k in atoms if k.startswith('F'))
        ph=sum(populations.get(k,0.) for k in atoms if k.startswith('H'))
        if ph>pf+.10:
            st.error('HF GVB bond sanity check failed: the selected pair is more hydrogen-centered '
                     'than fluorine-centered. Its bond assignment or active space is unreliable.')
            st.stop()

columns=st.columns(len(actual),gap='medium')
for column,(name,coeff) in zip(columns,actual):
    with column:
        st.markdown(f'**{name}**')
        try:
            st.plotly_chart(orbital_surface(result,seed,coeff,name,render_mode),
                use_container_width=True,key=f'surf-{mode}-{inspect}-{name}')
        except Exception as exc:
            st.warning(f'Could not render orbital: {exc}')
st.caption('Blue and red encode wavefunction phase—not electron spin. '
           + ('Point density samples |ψ(r)|² within the plotting box. '
              if render_mode == 'points' else
              'Isosurfaces mark constant-amplitude contours. ')
           + 'The two GVB pair orbitals are separately normalized and generally nonorthogonal.')

st.markdown('### Calculated orbital composition')
orbital_breakdowns(result,seed,actual)

st.markdown('### VSEPR reference versus calculated orbitals')
if is_lone or is_unassigned:
    st.caption('The VSEPR lone-pair hybrid is a geometric convention, not a unique occupied orbital. '
               'For equivalent lone pairs, rotations within their occupied subspace can change individual shapes. '
               'The actual signed IAO s/p decomposition above is the more direct diagnostic.')
    if idealA is not None:
        overlap=result.mol.intor('int1e_ovlp')
        rows=[]
        for name,coeff in actual:
            local_prob=atomic_iao_populations(result,coeff).get(atoms[0],0.)
            rows.append({'Orbital':name,'Site':atoms[0],
                         'VSEPR convention':metaA['hybrid'],
                         'Ideal s (%)':round(100*metaA['ideal_s_fraction'],1),
                         'Site population (%)':round(100*local_prob,1),
                         'Ideal-hybrid squared overlap (%)':round(100*float(coeff@overlap@idealA)**2,1)})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        with st.expander('Visualize one idealized lone-pair hybrid',expanded=False):
            try:
                st.plotly_chart(orbital_surface(result,seed,idealA,metaA['hybrid'],render_mode),use_container_width=True)
            except Exception as exc:
                st.warning(f'Ideal hybrid surface unavailable: {exc}')
    else:
        st.info('No unoccupied VSEPR reference direction is defined for this atomic site.')
else:
    st.caption('The σ reference is a bond-directed VSEPR s/p hybrid; π orbitals use a perpendicular p reference. '
               'Hybridization assignments depend on the chosen local orbital basis.')
    rows=[]
    for name,coeff in actual:
        for entry in hybrid_comparison(result,seed,bond,coeff,item.kind):
            row={k:v for k,v in entry.items() if k!='Ideal AO'}
            rows.append({'Calculated orbital':name,**row})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    with st.expander('View idealized VSEPR atomic orbitals',expanded=False):
        if st.toggle('Render two ideal hybrid orbitals',value=False,key='show_ideal_vsepr'):
            idealcols=st.columns(2)
            for col,(ideal,atom,meta) in zip(idealcols,
                   [(idealA,atoms[0],metaA),(idealB,atoms[1],metaB)]):
                with col:
                    st.markdown(f'**{atom} · {meta["hybrid"]}**')
                    try:
                        st.plotly_chart(orbital_surface(result,seed,ideal,f'Ideal {atom} {meta["hybrid"]}',render_mode),
                            use_container_width=True,key=f'ideal-{inspect}-{atom}')
                    except Exception as exc:
                        st.warning(f'Ideal hybrid surface unavailable: {exc}')

with st.expander('Wavefunctions, interpretation, and restrictions',expanded=False):
    st.latex(r'\Psi_{\mathrm{MO},i}=\phi_i(1)\phi_i(2)\,\chi_{S}(1,2)')
    st.latex(r'\Psi_{\mathrm{GVB},i}={\cal N}_i[u_i(1)v_i(2)+v_i(1)u_i(2)]\,\chi_S(1,2)')
    st.markdown('Both pictures are spin singlets. GVB-SCF variationally optimizes ALL molecular spatial orbitals (including core and previously external virtual orbitals) with all selected pairs correlated simultaneously. Unselected pairs are constrained to double occupation but their orbitals relax. The **perfect-pairing and strong-orthogonality restrictions remain**: there is no recoupling of different electron pairs or resonance-structure CI. Results are stationary-point candidates, not proven global optima. A unique hybridization is representation-dependent.')

try:
    data=export_selected(seed,result,localized,mapping,actual,space,gvb)
    st.download_button('Export selected orbital data (.npz)',data=data,
                       mime='application/octet-stream',
                       file_name='orbital_studio_bond_orbitals.npz',key='download-selected')
except Exception as exc:
    st.warning(f'Cannot assemble export: {exc}')
