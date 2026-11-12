"""Streamlit molecular bonding explorer: canonical vs localized RHF orbitals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qchem_analysis.molecular_geometry import geometry_from_smiles
from qchem_analysis.molecular_hf import (
    AtomicLibrary, decomposition, optimize_full, optimize_reduced,
    vsepr_single_point, localize_intrinsic_bond_orbitals,
    localize_occupied_standard, save_result,
)
from qchem_analysis.bonding_analysis import (
    HARTREE_TO_EV, assign_bond_labels, validate_density_preservation,
    effective_fock_matrix, overlap_analysis,
    plot_overlap_heatmap, plot_fock_matrix, localized_spin_archive,
)
from qchem_analysis.valence_bond_lab import (
    atomic_hybrid_character, bond_sites, bond_orbital_candidates,
    two_site_bond_functions, mixed_bond_orbital, orbital_fock_expectation,
    selected_pair_casci, HARTREE_TO_EV as EV,
)
from qchem_analysis.molecular_viewer import (
    plot_local_ingredients, plot_combined_with_bonds, show_decomposition_bars,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'saved_atomic_hf'
PRESETS = {
    'Acetylene · C₂H₂': 'C#C',
    'Hydrogen fluoride · HF': 'F',
    'Methane · CH₄': 'C',
    'Water · H₂O': 'O',
    'Carbon dioxide · CO₂': 'O=C=O',
    'Ethene · C₂H₄': 'C=C',
    'Ammonia · NH₃': 'N',
    'Formaldehyde · CH₂O': 'C=O',
    'Methanol · CH₃OH': 'CO',
    'Ethane · C₂H₆': 'CC',
    'Nitrogen · N₂': 'N#N',
    'Custom SMILES': None,
}
ATOM_COLORS = {'H':'#d8dde4','C':'#3b4656','N':'#407ac8','O':'#df4e4e','F':'#4fae76'}

st.markdown('''<style>
.block-container{max-width:2200px;padding-top:1.4rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.02em}
</style>''', unsafe_allow_html=True)


def geometry_figure(seed, xyz, title):
    positions=np.asarray(xyz)
    fig=go.Figure()
    for bond in seed.rdkit_mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        fig.add_trace(go.Scatter3d(
            x=[positions[a,0],positions[b,0]],
            y=[positions[a,1],positions[b,1]],
            z=[positions[a,2],positions[b,2]],
            mode='lines',line=dict(color='#7f8d9c',width=7),
            hoverinfo='skip',showlegend=False))
    for i,symbol in enumerate(seed.symbols):
        pos=positions[i]
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],y=[pos[1]],z=[pos[2]],mode='markers+text',
            marker=dict(size=10 if symbol!='H' else 6,
                        color=ATOM_COLORS.get(symbol,'#9d8eae')),
            text=[f'{symbol}{i+1}'],textposition='top center',
            name=f'{symbol}{i+1}',showlegend=False))
    fig.update_layout(title=title,height=360,
                      scene=dict(xaxis_title='x (Å)',yaxis_title='y (Å)',
                                 zaxis_title='z (Å)',aspectmode='data'),
                      margin=dict(l=0,r=0,b=0,t=50))
    return fig


def bond_lengths(seed, xyz):
    p=np.asarray(xyz)
    rows=[]
    for b in seed.rdkit_mol.GetBonds():
        a,c=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        rows.append({'Bond': f'{seed.symbols[a]}{a+1}–{seed.symbols[c]}{c+1}',
                     'Distance (Å)':round(float(np.linalg.norm(p[a]-p[c])),5)})
    return pd.DataFrame(rows)


def bond_angles(seed, xyz):
    from itertools import combinations
    p=np.asarray(xyz)
    rows=[]
    for atom in seed.rdkit_mol.GetAtoms():
        b=atom.GetIdx()
        neighbors=sorted(a.GetIdx() for a in atom.GetNeighbors())
        for a,c in combinations(neighbors,2):
            u=p[a]-p[b];v=p[c]-p[b]
            t=float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
            rows.append({'Angle':f'{seed.symbols[a]}{a+1}–{seed.symbols[b]}{b+1}–{seed.symbols[c]}{c+1}',
                         'Degrees (°)':round(float(np.degrees(np.arccos(np.clip(t,-1,1)))),3)})
    return pd.DataFrame(rows,columns=['Angle','Degrees (°)'])


def run_models(smiles,basis,charge,frozen,do_reduced,do_full,geometry_mode,
               reduced_steps,full_steps,random_seed):
    seed=geometry_from_smiles(smiles,random_seed)
    electrons=sum(atom.GetAtomicNum() for atom in seed.rdkit_mol.GetAtoms())-charge
    if electrons<2 or electrons%2:
        raise ValueError('Closed-shell RHF requires an even number of electrons (at least 2).')
    lib=AtomicLibrary.prepare(
        (atom.GetAtomicNum() for atom in seed.rdkit_mol.GetAtoms()),
        basis=basis,directory=CACHE)
    results={}
    models=[]
    if do_full: models.append(('Full Gaussian',False,full_steps))
    if do_reduced: models.append(('Saved atomic',True,reduced_steps))
    for name,reduced,steps in models:
        results[name+' · VSEPR start']=vsepr_single_point(
            seed,lib,charge=charge,reduced=reduced,frozen_core=frozen)
        if geometry_mode=='Optimize + compare VSEPR':
            optimizer=optimize_reduced if reduced else optimize_full
            results[name+' · optimized']=optimizer(
                seed,lib,charge=charge,frozen_core=frozen,maxsteps=steps)
    return seed,results


def get_localized(result, seed, mode):
    indices=result.valence_spatial_indices
    canonical=result.coefficients_ao[:,indices]
    if mode=='Canonical RHF':
        return canonical,[(f'MO {i+1} (canonical)',1.0) for i in indices]
    if mode=='Intrinsic bond orbitals (IBO)':
        C=localize_intrinsic_bond_orbitals(result)
    elif mode=='Boys localized':
        C=localize_occupied_standard(result,'boys')
    elif mode=='Pipek–Mezey localized':
        C=localize_occupied_standard(result,'pipek')
    else:
        raise ValueError(f'Unsupported representation {mode}')
    error=validate_density_preservation(result,C)
    if error>3e-5:
        raise RuntimeError(f'Localized state changed the occupied density ({error:.3g})')
    return C,assign_bond_labels(result,seed,C)


def save_as_bytes(result):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as path:
        dest=Path(path)/'valence_spin_orbitals.npz'
        save_result(result,dest)
        return dest.read_bytes()


st.title('Molecular Bonding Explorer')
st.caption('VSEPR geometries · RHF localized bonds · calculated hybridization · spin-coupled pair analysis')
with st.expander('How to read an orbital: start here', expanded=False):
    st.markdown("""
    **1. Choose an IBO-localized occupied orbital.** A C–H σ bond, C–C σ bond,
    or C–C π bond should often appear separately instead of being merged into
    several symmetry-adapted canonical orbitals. For acetylene, try the C–C π
    orbital and show C1:2py alongside C2:2py (or both pz functions).

    **2. Compare signed ingredients and the full orbital.** Matching wavefunction
    phase in the overlap region indicates constructive interference. An overall
    change from blue to red has no physical meaning; only relative phase matters.

    **3. Inspect the overlap matrix.** Positive and negative off-diagonal
    entries show how selected atomic terms contribute to orbital normalization,
    not how much energy a bond releases. For a full-basis calculation, inspect
    the **projection fraction** before drawing conclusions from saved atomic functions.

    **4. Inspect the Fock matrix.** An IBO has a diagonal Fock expectation and
    off-diagonal couplings. These are not independent bond dissociation energies.
    IBO and canonical orbitals represent the *same* RHF occupied density.
    """)
with st.sidebar:
    st.header('Molecule and calculation')
    with st.form('molecular_settings'):
        preset=st.selectbox('Molecule',list(PRESETS),index=0)
        custom=st.text_input('Custom SMILES',help='Only used for Custom SMILES.')
        basis=st.selectbox('Gaussian basis',['cc-pVDZ','cc-pVTZ','6-31g','sto-3g'])
        charge=st.number_input('Charge',min_value=-3,max_value=3,value=0,step=1)
        frozen=st.checkbox('Freeze occupied core',value=False)
        full=st.checkbox('Full Gaussian HF',value=True)
        reduced=st.checkbox('Reduced saved-atom HF',value=True)
        geometry_mode=st.radio('Geometry treatment',
            ['Optimize + compare VSEPR','VSEPR starting geometry only'])
        with st.expander('Optimization settings'):
            reduced_steps=st.slider('Reduced geometry iterations',1,80,20)
            full_steps=st.slider('Full geometry iterations',1,120,45)
            random_seed=st.number_input('Geometry seed',1,1000000,11)
        go_clicked=st.form_submit_button('Calculate molecular orbitals',type='primary',use_container_width=True)
    st.caption('All electronic and geometry calculations are performed locally.')

if go_clicked:
    smiles=custom.strip() if preset=='Custom SMILES' else PRESETS[preset]
    if not smiles:
        st.error('Provide a molecular SMILES string.')
    elif not (full or reduced):
        st.error('Choose at least one electronic model.')
    else:
        st.session_state.pop('molecule_data',None)
        st.session_state.pop('bonding_localizations',None)
        st.session_state.pop('bonding_fock',None)
        st.session_state.pop('pair_result',None)
        try:
            with st.spinner('Calculating atomic references, HF orbitals, and molecular geometries…'):
                seed,results=run_models(smiles,basis,int(charge),frozen,
                                         reduced,full,geometry_mode,
                                         reduced_steps,full_steps,int(random_seed))
            st.session_state['molecule_data']={
                'seed':seed,'results':results,'smiles':smiles,'basis':basis,
                'charge':charge,'frozen':frozen,
            }
        except Exception as exc:
            st.exception(exc)

state=st.session_state.get('molecule_data')
if state is None:
    st.info('Choose a molecule, then click **Calculate molecular orbitals**. Start with acetylene or HF to see localized bonds.')
    cols=st.columns(3)
    for col,(label,smiles) in zip(cols,list(PRESETS.items())[:3]):
        with col:
            st.markdown(f'**{label}**')
            try:
                initial=geometry_from_smiles(smiles)
                st.plotly_chart(geometry_figure(initial,initial.xyz_angstrom,'VSEPR start'),
                                use_container_width=True,key=f'preview-{smiles}')
            except Exception:
                pass
    st.stop()

seed=state['seed'];results=state['results'];model_names=list(results)
optimized_names=[name for name in model_names if 'optimized' in name]
default_model=optimized_names[0] if optimized_names else model_names[0]

bond_tab,lab_tab,geom_tab,export_tab=st.tabs([
    'Bonding orbitals','Bond Formation Lab','Geometry & energy','Results & export',
])

with bond_tab:
    st.subheader('Explore an individual bonding orbital')
    selected_model=st.selectbox('Electronic model / geometry',model_names,
                                index=model_names.index(default_model))
    result=results[selected_model]
    localization=st.selectbox('Occupied-orbital representation',[
        'Intrinsic bond orbitals (IBO)','Canonical RHF',
        'Boys localized','Pipek–Mezey localized',
    ])
    cache_key=(selected_model,localization)
    cache=st.session_state.setdefault('bonding_localizations',{})
    if cache_key not in cache:
        try:
            with st.spinner('Localizing occupied valence orbitals…'):
                cache[cache_key]=get_localized(result,seed,localization)
        except Exception as exc:
            st.error(f'Orbital localization failed: {exc}')
    if cache_key not in cache:
        st.stop()
    Cval,orbital_names=cache[cache_key]
    valence_indices=result.valence_spatial_indices
    chosen=st.selectbox('Bonding orbital / lone pair',range(len(valence_indices)),
        format_func=lambda i:(f'{i+1}. {orbital_names[i][0]}' +
           (f' · match {orbital_names[i][1]:.0%}' if localization!='Canonical RHF' else '')),
        key=f'orbital-choice::{selected_model}::{localization}')
    actual=Cval[:,chosen]
    mo_index=valence_indices[chosen]
    st.caption('IBO bond names are inferred by matching calculated orbitals to bond-aligned trial functions; they are not enforced. A low match or “mixed” name indicates ambiguity. Each orbital has an arbitrary overall sign; relative signs are what matter.')
    with st.expander('Display settings and atomic component selection', expanded=False):
        st.markdown('**Visual representation**')
        ingredient_basis=st.radio('Constituent atomic functions',
                                  ['Atomic s/p','Bond-aligned hybrids'],horizontal=False)
        kind='atomic' if ingredient_basis=='Atomic s/p' else 'hybrid'
        data=decomposition(result,mo_index,kind,coefficients_ao=actual)
        ranked=np.argsort(data['weights'])[::-1]
        # Start with the largest signed contribution from each substantially
        # represented atom. This exposes both ends of a localized bond, rather
        # than showing four functions on the same atomic center by default.
        first_by_atom={}
        if len(ranked):
            largest=float(data['weights'][ranked[0]])
            for k in ranked:
                name=data['labels'][int(k)]
                atom=name.split(':',1)[0]
                if atom not in first_by_atom and data['weights'][k]>=max(.004,.05*largest):
                    first_by_atom[atom]=name
        default_terms=list(first_by_atom.values())[:4]
        if len(default_terms)<4:
            for k in ranked:
                name=data['labels'][int(k)]
                if name not in default_terms and data['weights'][k]>=max(.008,.13*largest):
                    default_terms.append(name)
                if len(default_terms)>=4:
                    break
        selected_components=st.multiselect(
            'Select atomic contributions to the chosen bond',options=data['labels'],
            default=default_terms,
            key=f'components::{selected_model}::{localization}::{chosen}::{kind}')
        cutoff=st.slider('Common isosurface amplitude cutoff (%)',0.5,25.0,4.0,0.5)/100
        resolution=st.slider('3D grid resolution',25,65,39,2)
        extent=st.slider('Plot half-width (Bohr)',2.0,17.0,7.0,0.5)
        composite_mode=st.radio('Complete molecular orbital',
                                 ['Isosurface','Probability point cloud'])
        points=st.slider('Point count',2000,80000,18000,2000) if composite_mode=='Probability point cloud' else 18000
        show_overlay=st.checkbox('Overlay full projected sum on ingredient plot',value=False)
        st.caption('Blue/red = wavefunction phase, not electron spin. All ingredient surfaces use a *common amplitude threshold*.')

    st.subheader(orbital_names[chosen][0])
    if localization=='Canonical RHF':
        st.caption(f'Canonical Fock eigenvalue: {result.orbital_energies[mo_index]:.6f} Ha')
    else:
        st.caption('An occupied-space rotation of RHF: total density and HF energy are unchanged.')
    st.caption('Both 3D views use the full page width. Use the chart fullscreen control for an even larger view.')
    col_left,col_right=st.columns(2,gap='small')
    with col_left:
        st.markdown('**Signed atomic ingredients**')
        try:
            if selected_components:
                fig=plot_local_ingredients(
                    result,data,selected_components,seed=seed,
                    extent_bohr=extent,grid_points=resolution,
                    amplitude_cutoff=cutoff,overlay_total=show_overlay)
                fig.update_layout(height=760)
                st.plotly_chart(fig,use_container_width=True,key='bond-ingredients')
            else:
                st.info('Choose one or more local-orbital terms to display.')
        except Exception as exc:
            st.error(f'Ingredient rendering failed: {exc}')
    with col_right:
        st.markdown('**Combined molecular orbital**')
        try:
            figure=plot_combined_with_bonds(
                result,mo_index,seed=seed,coefficients_ao=actual,
                mode='points' if composite_mode=='Probability point cloud' else 'surface',
                grid_points=resolution,extent_bohr=extent,point_count=points)
            figure.update_layout(height=760)
            st.plotly_chart(figure,use_container_width=True,key='bond-combined')
        except Exception as exc:
            st.error(f'Combined MO plot failed: {exc}')
    completeness=float(data['probability_represented'])
    st.progress(np.clip(completeness,0,1),
                text=f'Projection onto saved atomic orbitals: {completeness:.1%}')
    if completeness<.90:
        st.warning('A substantial fraction of this full-basis MO is outside the saved atomic-orbital subspace. Interpret the ingredient plots as a partial reconstruction.')
    st.caption('Each ingredient is a **signed amplitude** cᵢ φᵢ(r), not a separate electron cloud. The lobes add coherently, including cancellation. The right panel shows the complete RHF orbital; the left panel reconstructs its atomic projection. Blue/red represent relative wavefunction phase, never spin. An overall sign flip has no physical effect.')
    st.markdown('#### Calculated atomic hybridization')
    st.caption('Derived from the orbital’s saved-atom s/p projection—not assigned from VSEPR. '
               'The reported sp exponent is a representation-dependent descriptor, not a unique observable.')
    try:
        atom_projection=decomposition(result,mo_index,'atomic',coefficients_ao=actual)
        hybrid_rows=atomic_hybrid_character(atom_projection)
        if hybrid_rows:
            st.dataframe(pd.DataFrame(hybrid_rows),hide_index=True,use_container_width=True)
        else:
            st.info('No significant s/p contributions were identified in the saved atomic basis.')
    except Exception as exc:
        st.warning(f'Hybrid character could not be computed: {exc}')
    st.markdown('#### Signed local-orbital coefficients')
    t_hyb,t_sp=st.tabs(['Hybrid amplitudes','Atomic s/p amplitudes'])
    for t,representation in ((t_hyb,'hybrid'),(t_sp,'atomic')):
        with t:
            p=decomposition(result,mo_index,representation,coefficients_ao=actual)
            bars=show_decomposition_bars(result,mo_index,representation,
                                          coefficients_ao=actual)
            bars.update_layout(height=370)
            st.plotly_chart(bars,use_container_width=True,
                            key=f'bar-{representation}')
            rows=pd.DataFrame({'Orbital':p['labels'],
                               'Signed amplitude':p['coefficients'],
                               'Löwdin weight':p['weights']})
            st.dataframe(rows.sort_values('Löwdin weight',ascending=False),
                         hide_index=True,use_container_width=True)

    st.divider()
    st.subheader('Orbital overlap & energy diagnostics')
    st.caption('These diagnostics refer to the currently selected bond orbital and localization. The local coefficient basis does not affect the SCF solution.')
    overlap_tab,fock_tab=st.tabs(['Constructive/destructive overlap','Fock energies & couplings'])
    with overlap_tab:
        overlap_rep=st.radio('Decomposition basis for overlap',
                              ['Atomic s/p','Bond-aligned hybrids'],horizontal=True,
                              key='overlap_basis')
        rep='atomic' if overlap_rep=='Atomic s/p' else 'hybrid'
        p=decomposition(result,mo_index,rep,coefficients_ao=actual)
        terms=overlap_analysis(p)
        a,b,c=st.columns(3)
        a.metric('Reconstructed norm',f"{terms['total']:.4f}")
        b.metric('Self terms',f"{terms['self_terms']:+.4f}")
        c.metric('Total cross terms',f"{terms['cross_terms']:+.4f}")
        st.caption('The full sum is 1 only for a complete projection. Each off-diagonal heatmap entry appears twice in the normalization sum; the pair table counts each pair once as 2aᵢaⱼSᵢⱼ.')
        left,right=st.columns([1,1.65])
        with left:
            st.plotly_chart(plot_overlap_heatmap(p,grouped=True),
                            use_container_width=True,key='atom-interference')
        with right:
            st.plotly_chart(plot_overlap_heatmap(p,grouped=False),
                            use_container_width=True,key='basis-interference')
        cross=pd.DataFrame(terms['top_interatomic'])
        st.markdown('**Largest interatomic overlap terms**')
        st.caption('A positive cross term is constructive for the chosen orbital normalization, not a uniquely defined bond-energy contribution.')
        if len(cross):
            st.dataframe(cross,hide_index=True,use_container_width=True)
        else:
            st.info('No interatomic terms in the displayed decomposition.')
    with fock_tab:
        fock_cache=st.session_state.setdefault('bonding_fock',{})
        if cache_key not in fock_cache:
            try:
                with st.spinner('Evaluating the AO Fock operator at the converged density…'):
                    fock_cache[cache_key]=effective_fock_matrix(result,Cval)
            except Exception as exc:
                st.error(f'Could not evaluate Fock matrix: {exc}')
        if cache_key in fock_cache:
            F=fock_cache[cache_key]
            m1,m2=st.columns(2)
            m1.metric('Selected orbital Fock expectation',
                      f'{F[chosen,chosen]:+.5f} Ha')
            m2.metric('Selected orbital expectation',
                      f'{F[chosen,chosen]*HARTREE_TO_EV:+.3f} eV')
            st.plotly_chart(plot_fock_matrix(F,[name for name,score in orbital_names]),
                            use_container_width=True,key='fock-plot')
            st.markdown('**Bond-by-bond orbital energy expectations**')
            bond_rows=[]
            for idx,(name,score) in enumerate(orbital_names):
                bond_rows.append({
                    'Localized orbital / assignment':name,
                    'Assignment confidence':round(float(score),3),
                    'Fock expectation (Ha)':round(float(F[idx,idx]),6),
                    'Fock expectation (eV)':round(float(F[idx,idx])*HARTREE_TO_EV,4),
                })
            st.dataframe(pd.DataFrame(bond_rows),hide_index=True,use_container_width=True)
            st.caption('These are diagonal Fock *expectation values* for each chosen localized orbital, '
                       'not additive bond energies or energies required to break bonds. '
                       'The selected-pair CASCI comparison is on the Bond Formation Lab tab.')
            st.caption('Diagonal terms are ⟨φᵢ|F|φᵢ⟩ and off-diagonal terms are couplings. They are representation-dependent and are NOT bond dissociation energies. The eigenvalues of the occupied Fock block are invariant under occupied-space rotations.')
            energies=np.linalg.eigvalsh(F)
            st.dataframe(pd.DataFrame({
                'Occupied-block eigenvalue (Ha)':energies,
                'Original canonical valence energy (Ha)':np.sort(result.orbital_energies[result.valence_spatial_indices]),
            }),hide_index=True,use_container_width=True)

with lab_tab:
    st.subheader('Construct a bond from two atom-centered orbital contributions')
    st.caption('This lab changes coefficients and relative phase in a two-center trial function. '
               'It does not recalculate the self-consistent molecular energy. The electron-pair '
               'calculation below is an independent CASCI(2,2) singlet, not full GVB-PP.')
    chosen_bonds=bond_sites(seed)
    model_for_lab=st.selectbox('Electronic model',list(results),
        index=list(results).index(default_model),key='lab-model')
    lab_result=results[model_for_lab]
    lab_mode=st.selectbox('Bond orbital representation',
        ['Intrinsic bond orbitals (IBO)','Boys localized','Pipek–Mezey localized','Canonical RHF'],
        key='lab-localization')
    lab_key=(model_for_lab,lab_mode)
    lab_cache=st.session_state.setdefault('bonding_localizations',{})
    if lab_key not in lab_cache:
        try:
            lab_cache[lab_key]=get_localized(lab_result,seed,lab_mode)
        except Exception as exc:
            st.error(f'Localization failed: {exc}')
    if lab_key in lab_cache and chosen_bonds:
        lab_C,lab_names=lab_cache[lab_key]
        lab_bond_idx=st.selectbox('Choose bond',range(len(chosen_bonds)),
            format_func=lambda k:chosen_bonds[k]['name'], key='lab-bond')
        lab_bond=chosen_bonds[lab_bond_idx]
        lab_candidates=bond_orbital_candidates(lab_names,lab_bond)
        orbital_options=lab_candidates if lab_candidates else list(range(lab_C.shape[1]))
        lab_index=st.selectbox('Orbital to decompose',orbital_options,
            format_func=lambda i:lab_names[i][0],key='lab-orbital')
        lab_mo_index=lab_result.valence_spatial_indices[lab_index]
        lab_orbital=lab_C[:,lab_index]
        lab_proj=decomposition(lab_result,lab_mo_index,'atomic',coefficients_ao=lab_orbital)
        a_name=f"{seed.symbols[lab_bond['a']]}{lab_bond['a']+1}"
        b_name=f"{seed.symbols[lab_bond['b']]}{lab_bond['b']+1}"
        try:
            from qchem_analysis.molecular_viewer import plot_combined_with_bonds
            left,right=two_site_bond_functions(lab_result,lab_proj,a_name,b_name)
            S=lab_result.mol.intor('int1e_ovlp')
            left=left/np.sqrt(left@S@left)
            right=right/np.sqrt(right@S@right)
            actual_overlap=float(left@S@right)
            c1,c2=st.columns(2)
            with c1:
                angle=st.slider('Mixing angle (degrees)',0.,90.,45.,1.,key='lab-angle')
            with c2:
                relative_phase=st.radio('Relative wavefunction phase',['In phase','Out of phase'],
                                        horizontal=True,key='lab-phase')
            phase=1 if relative_phase=='In phase' else -1
            trial=mixed_bond_orbital(left,right,S,angle,phase)
            st.metric('Atomic-function overlap S_AB',f'{actual_overlap:+.5f}')
            st.latex(r'\phi=\mathcal N[\cos(\theta)\,h_A\;\pm\;\sin(\theta)\,h_B]')
            v1,v2=st.columns(2)
            with v1:
                st.plotly_chart(plot_combined_with_bonds(lab_result,lab_mo_index,
                    seed=seed,coefficients_ao=trial,mode='surface',grid_points=39),
                    use_container_width=True,key='lab-trial')
                st.caption('Your trial combination of the two atomic contributions')
            with v2:
                st.plotly_chart(plot_combined_with_bonds(lab_result,lab_mo_index,
                    seed=seed,coefficients_ao=lab_orbital,mode='surface',grid_points=39),
                    use_container_width=True,key='lab-reference')
                st.caption('Actual localized occupied RHF orbital')
            try:
                f_trial=orbital_fock_expectation(lab_result.mol,
                    lab_result.coefficients_ao[:,:lab_result.nocc],trial)
                f_actual=orbital_fock_expectation(lab_result.mol,
                    lab_result.coefficients_ao[:,:lab_result.nocc],lab_orbital)
                st.dataframe(pd.DataFrame([
                    {'Orbital':'Constructed trial','Fock expectation (Ha)':f_trial,
                     'Fock expectation (eV)':f_trial*EV},
                    {'Orbital':'Localized RHF','Fock expectation (Ha)':f_actual,
                     'Fock expectation (eV)':f_actual*EV},
                ]),hide_index=True,use_container_width=True)
                st.caption('Fock expectations are one-electron mean-field diagnostics, not bond strengths.')
            except Exception as exc:
                st.warning(f'Fock expectation unavailable: {exc}')
            st.divider()
            st.subheader('Spin-coupled electron pair: selected-bond CASCI(2,2)')
            st.markdown('Allow two electrons to occupy a bonding and antibonding spatial orbital '
                        'in an optimized singlet **configuration mixture**, with all other occupied '
                        'orbitals held inactive. This models correlation within one selected pair '
                        'and includes ionic/covalent mixing, but is not a global GVB perfect-pairing wavefunction.')
            allowed=(lab_result.mode=='full' and not lab_result.frozen_core and
                     lab_mode!='Canonical RHF')
            if not allowed:
                st.info('For this pair calculation, choose full Gaussian RHF, no frozen core, and a localized orbital representation.')
            if st.button('Calculate selected-pair singlet',disabled=not allowed,key='calculate-pair'):
                try:
                    with st.spinner('Diagonalizing the two-electron singlet pair…'):
                        pair=selected_pair_casci(lab_result,lab_C,lab_index,
                                                 lab_proj,a_name,b_name)
                    st.session_state['pair_result']=(model_for_lab,lab_mode,lab_bond_idx,lab_index,pair)
                except Exception as exc:
                    st.error(f'Selected-pair calculation failed: {exc}')
            stored=st.session_state.get('pair_result')
            if stored and stored[:4]==(model_for_lab,lab_mode,lab_bond_idx,lab_index):
                pair=stored[4]
                d1,d2,d3=st.columns(3)
                d1.metric('RHF total E',f"{pair['rhf_total_energy_hartree']:.7f} Ha")
                d2.metric('Selected-pair total E',f"{pair['total_energy_hartree']:.7f} Ha")
                d3.metric('Pair correlation lowering',f"{pair['pair_correlation_lowering_ev']:+.4f} eV")
                st.caption('This is the difference between **two total molecular energies** at '
                    'one fixed geometry and active space. It is not a bond dissociation energy.')
                st.write('Pair natural occupations (two active spatial orbitals):',
                         ', '.join(f'{k:.4f}' for k in pair['active_natural_occupations']))
        except Exception as exc:
            st.error(f'Bond lab unavailable for this selection: {exc}')

with geom_tab:
    st.subheader('Electronic energy at the VSEPR seed versus optimized geometry')
    rows=[]
    for model in ('Full Gaussian','Saved atomic'):
        start=results.get(model+' · VSEPR start')
        opt=results.get(model+' · optimized')
        if start is None:continue
        rows.append({
            'Electronic model':model,
            'VSEPR start E (Ha)':start.energy_hartree,
            'Optimized E (Ha)':opt.energy_hartree if opt is not None else None,
            'Δ E (optimized − seed), Ha':opt.energy_hartree-start.energy_hartree if opt else None,
            'Optimization converged':opt.geometry_converged if opt else None,
        })
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('VSEPR supplies only a nuclear geometry, not its own quantum energy. The starting-energy column is a fully self-consistent RHF calculation at fixed VSEPR-seeded coordinates, with the *same electronic model* as the optimized column.')
    cols=st.columns(min(3,len(results)))
    for j,(name,res) in enumerate(results.items()):
        with cols[j%len(cols)]:
            st.plotly_chart(geometry_figure(seed,res.xyz_angstrom,name),
                            use_container_width=True,key=f'geo-{j}')
            st.markdown('**Bond lengths**')
            st.dataframe(bond_lengths(seed,res.xyz_angstrom),
                         hide_index=True,use_container_width=True)
            st.markdown('**Bond angles**')
            angles=bond_angles(seed,res.xyz_angstrom)
            if len(angles):
                st.dataframe(angles,hide_index=True,use_container_width=True)
            else:
                st.caption('No bond angle in a diatomic molecule.')
            if not res.converged:
                st.warning('Electronic SCF did not converge')
            if not res.geometry_converged:
                st.warning('Geometry optimization did not converge')
    st.caption('Molecules are aligned consistently: the primary heavy-atom bond is placed along x; planar systems are oriented in xy, making perpendicular p_y/p_z labels meaningful for linear molecules.')

with export_tab:
    st.subheader('Calculation results')
    st.dataframe(pd.DataFrame([{
        'Model/geometry':name,'Total RHF energy (Ha)':res.energy_hartree,
        'Valence spatial orbitals':len(res.valence_spatial_indices),
        'SCF converged':res.converged,'Frozen core':res.frozen_core,
    } for name,res in results.items()]),hide_index=True,use_container_width=True)
    st.markdown('**Export canonical occupied valence spin orbitals**')
    st.caption('These exports contain the converged canonical RHF solution, including the α and β spin records. Localized orbitals are rotations of the same occupied subspace.')
    for i,(name,res) in enumerate(results.items()):
        try:
            st.download_button('Download '+name+' (.npz)',
                data=save_as_bytes(res),mime='application/octet-stream',
                file_name=f'molecule_{res.mode}_{i}_{state["basis"]}.npz',
                key=f'export-{i}')
        except Exception as exc:
            st.error(f'Could not export {name}: {exc}')
    st.markdown('**Export current localized occupied valence orbitals**')
    st.caption('Select a model and orbital representation under Bonding orbitals first. The entire valence orbital set is exported with labels and the rotation-specific occupied Fock matrix when available.')
    cache=st.session_state.get('bonding_localizations',{})
    for (model,localization),(C,names) in cache.items():
        F=st.session_state.get('bonding_fock',{}).get((model,localization))
        payload=localized_spin_archive(results[model],C,names,localization,
                                       fock_matrix=F)
        st.download_button(f'Download {model} · {localization}',payload,
                           file_name=('localized_'+model.lower().replace(' ','_').replace('·','_')
                                      +'_'+localization.lower().replace(' ','_')+'.npz'),
                           mime='application/octet-stream',key=f'localized-export-{model}-{localization}')

st.divider()
st.caption('RHF localization preserves the density and total RHF energy. The optional selected-pair CASCI(2,2) adds a two-electron correlated singlet at fixed nuclear geometry; it is not global GVB-PP. Neither localized Fock expectations nor pair-correlation lowering are dissociation energies.')
