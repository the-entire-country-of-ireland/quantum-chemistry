"""Phase-colored 3D MO visualization and signed atomic/hybrid components."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from types import SimpleNamespace
from .orbital_sampling import _surface_level_for_probability, _sample_probability_points

BLUE, RED = '#2878b5', '#d34a4a'


def grid_for_orbital(mol, c, extent_bohr=6., grid_points=100):
    if grid_points<25:
        raise ValueError('grid_points must be >= 25')
    axis=np.linspace(-extent_bohr,extent_bohr,grid_points)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    vals=np.empty(len(grid))
    for a in range(0,len(grid),8192):
        b=min(len(grid),a+8192)
        vals[a:b]=mol.eval_gto('GTOval_sph',grid[a:b])@c
    return grid, vals.reshape((grid_points,)*3)


def _draw_surface(figure,grid,values,coverage,name='',opacity=0.55, legend=True):
    threshold=_surface_level_for_probability(values,coverage)
    for sign,phase,color in [(1,'+ phase',BLUE),(-1,'− phase',RED)]:
        val=sign*values
        if val.max()<=threshold:continue
        figure.add_trace(go.Isosurface(
            x=grid[:,0],y=grid[:,1],z=grid[:,2],value=val.ravel(),
            isomin=threshold,isomax=threshold*(1+1e-5),surface_count=1,
            colorscale=[[0,color],[1,color]],showscale=False,
            opacity=opacity,caps=dict(x_show=False,y_show=False,z_show=False),
            name=f'{name} {phase}',showlegend=legend))



def plot_molecular_orbital(result, mo_index, mode='points',
                           extent_bohr=None,grid_points=70,
                           point_count=30000,probability_coverage=.95,
                           seed=12345,coefficients_ao=None):
    """Plot a complete canonical or localized MO as a signed surface or |ψ|² cloud.

    Individual *atomic orbital* terms are rendered by plot_local_ingredients;
    atom-summed surfaces are intentionally excluded to avoid hiding σ/π
    contributions behind the total contribution of each atom.
    """
    if mode not in ('points','surface'):
        raise ValueError('mode must be points or surface')
    if not 0<probability_coverage<1:
        raise ValueError('probability_coverage must be between 0 and 1')
    mol=result.mol
    c=(result.coefficients_ao[:,mo_index] if coefficients_ao is None
       else np.asarray(coefficients_ao,dtype=float))
    if extent_bohr is None:
        extent_bohr=float(np.max(np.abs(mol.atom_coords()))+4.0)
    if extent_bohr<=0:
        raise ValueError('extent_bohr must be positive')
    figure=go.Figure()
    grid,val=grid_for_orbital(mol,c,extent_bohr,grid_points)
    if mode=='surface':
        _draw_surface(figure,grid,val,probability_coverage,'MO',.55)
        subtitle=f'phase isosurface; ~{probability_coverage:.0%} of in-box grid density'
    else:
        adapter=SimpleNamespace(molecule=mol)
        pts, amplitudes, probability = _sample_probability_points(
            adapter,c,val,extent_bohr,point_count,seed)
        for positive,phase,color in ((True,'+ phase',BLUE),(False,'− phase',RED)):
            mask=amplitudes>=0 if positive else amplitudes<0
            if not mask.any():continue
            subset=pts[mask]
            figure.add_trace(go.Scatter3d(
                x=subset[:,0],y=subset[:,1],z=subset[:,2],mode='markers',
                marker=dict(size=1.5,color=color,opacity=.55),
                name=phase,hoverinfo='skip'))
        subtitle=f'{point_count:,} samples of |ψ|²; estimated in-box mass={probability:.2f}'
    for i in range(mol.natm):
        pos=mol.atom_coord(i)
        figure.add_trace(go.Scatter3d(x=[pos[0]],y=[pos[1]],z=[pos[2]],
            mode='markers+text',marker=dict(color='black',size=5),
            text=[mol.atom_symbol(i)],textposition='top center',
            name=f'{mol.atom_symbol(i)}{i+1}',showlegend=False))
    figure.update_layout(
        title=f'{result.mode} orbital {mo_index+1} | {mode}<br><sup>{subtitle}</sup>',
        scene=dict(xaxis_title='x (Bohr)',yaxis_title='y (Bohr)',
                   zaxis_title='z (Bohr)',aspectmode='data'),
        margin=dict(l=0,r=0,b=0,t=85))
    return figure


def show_decomposition_bars(result,mo_index,representation='hybrid',top=15,
                            coefficients_ao=None):
    from .molecular_hf import decomposition
    data=decomposition(result,mo_index,representation,coefficients_ao)
    inds=np.argsort(data['weights'])[::-1][:top][::-1]
    # Plot coefficients are signed, bars are colored by coefficient sign.
    return go.Figure(go.Bar(
        x=data['coefficients'][inds],y=[data['labels'][i] for i in inds],
        orientation='h',marker_color=[BLUE if a>=0 else RED for a in data['coefficients'][inds]],
        customdata=data['weights'][inds],
        hovertemplate='%{y}<br>signed amplitude=%{x:.3f}<br>Lowdin weight=%{customdata:.1%}<extra></extra>'
    )).update_layout(title=f'MO {mo_index+1} — signed {representation} amplitudes',
                     xaxis_title='Coefficient in normalized nonorthogonal local functions',
                     margin=dict(l=135,r=10,t=60,b=40))


def _add_molecular_skeleton(fig, result, seed=None, *, atom_size=5):
    """Bond lines and atom numbers in the exact frame of the orbital grid."""
    p=np.asarray(result.mol.atom_coords())
    if seed is not None:
        for bond in seed.rdkit_mol.GetBonds():
            i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            fig.add_trace(go.Scatter3d(
                x=[p[i,0],p[j,0]], y=[p[i,1],p[j,1]], z=[p[i,2],p[j,2]],
                mode='lines',line=dict(color='#7a7d83',width=4),
                hoverinfo='skip',showlegend=False))
    for k in range(result.mol.natm):
        q=p[k]
        label=f'{result.mol.atom_symbol(k)}{k+1}'
        fig.add_trace(go.Scatter3d(
            x=[q[0]],y=[q[1]],z=[q[2]],mode='markers+text',
            marker=dict(size=atom_size,color='#343434'),
            text=[label],textposition='top center',name=label,showlegend=False))


def plot_local_ingredients(result, projection, selection, *, seed=None,
                           extent_bohr=7.0,grid_points=70,
                           amplitude_cutoff=0.045,opacity=0.40,
                           overlay_total=False):
    """Draw *individual signed AO/hybrid terms*, not atom-summed contributions.

    All components are evaluated on a common spatial grid, with the same
    absolute isovalue. This is essential for interpreting their relative
    contributions and spatial interference. The molecular orbital is formed
    by coherent addition, not by adding probabilities.
    """
    if not 0 < amplitude_cutoff < 1:
        raise ValueError('Amplitude cutoff must be a fraction between 0 and 1')
    label_to_index={name:i for i,name in enumerate(projection['labels'])}
    indices=[label_to_index[name] for name in selection if name in label_to_index]
    if not indices:
        raise ValueError('Select one or more orbital components')
    axis=np.linspace(-extent_bohr,extent_bohr,grid_points)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    coeffs=np.asarray(projection['components_ao'])[:,indices]
    values=np.empty((len(grid),len(indices)),dtype=float)
    for start in range(0,len(grid),4096):
        stop=min(start+4096,len(grid))
        ao=result.mol.eval_gto('GTOval_sph',grid[start:stop])
        values[start:stop,:]=ao@coeffs
    max_abs=float(np.max(np.abs(values)))
    if not np.isfinite(max_abs) or max_abs<=0:
        raise RuntimeError('Selected AO contributions vanish on the grid')
    level=amplitude_cutoff*max_abs
    fig=go.Figure()
    for j,k in enumerate(indices):
        name=projection['labels'][k]
        orbital_values=values[:,j]
        for sign,phase,color in ((1,'positive',BLUE),(-1,'negative',RED)):
            signed=sign*orbital_values
            if float(np.max(signed))<=level:
                continue
            fig.add_trace(go.Isosurface(
                x=grid[:,0],y=grid[:,1],z=grid[:,2],value=signed,
                isomin=level,isomax=level*(1+1e-5),surface_count=1,
                colorscale=[[0,color],[1,color]],showscale=False,
                opacity=opacity, caps=dict(x_show=False,y_show=False,z_show=False),
                name=f'{name} · {phase}',legendgroup=name,showlegend=True,
                hovertemplate=f'{name} · {phase}<extra></extra>'))
    if overlay_total:
        # The full projected sum shows interference between all saved local
        # functions, including those hidden by the component selection.
        _,full_val=grid_for_orbital(result.mol,projection['projected_ao'],extent_bohr,grid_points)
        _draw_surface(fig,grid,full_val,0.88,'Projected sum',0.13)
    _add_molecular_skeleton(fig,result,seed)
    fig.update_layout(
        title='Signed contributions · common absolute amplitude isovalue',
        scene=dict(xaxis_title='x (Bohr)',yaxis_title='y (Bohr)',
                   zaxis_title='z (Bohr)',aspectmode='data'),
        margin=dict(l=0,r=0,t=65,b=0),
        legend=dict(orientation='h',y=-0.10),
    )
    return fig


def plot_combined_with_bonds(result,mo_index,*,seed=None,coefficients_ao=None,
                             mode='surface',extent_bohr=7.0,
                             grid_points=70,probability_coverage=.95,
                             point_count=30000):
    """Full localized MO on the same aligned coordinate frame."""
    fig=plot_molecular_orbital(
        result,mo_index,mode=mode,
        extent_bohr=extent_bohr,grid_points=grid_points,
        probability_coverage=probability_coverage,point_count=point_count,
        coefficients_ao=coefficients_ao)
    # The base renderer already adds unlabeled nuclei; replace them with
    # labeled nuclei and bond lines to make p_x/p_y/p_z interpretable.
    for trace in list(fig.data):
        if trace.type=='scatter3d' and trace.showlegend is False:
            fig.data=tuple(t for t in fig.data if t is not trace)
    _add_molecular_skeleton(fig,result,seed)
    return fig
