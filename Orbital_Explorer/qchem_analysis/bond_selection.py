"""Bond connectivity and atomic ingredient helpers used by both MO and GVB views."""
from __future__ import annotations
import numpy as np

def bond_sites(seed):
    """Selectable bond list keyed by the actual connectivity."""
    output = []
    for bond in seed.rdkit_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        name = f'{seed.symbols[i]}{i+1}–{seed.symbols[j]}{j+1}'
        output.append({'index':len(output), 'a':i, 'b':j,
                       'name':name, 'order':float(bond.GetBondTypeAsDouble())})
    return output

def bond_map_figure(seed, xyz, selection=None):
    """3D molecular stick drawing with selectable marker at each bond center."""
    import plotly.graph_objects as go
    xyz=np.asarray(xyz,dtype=float)
    fig=go.Figure()
    bonds=bond_sites(seed)
    for b in bonds:
        i,j=b['a'],b['b']
        fig.add_trace(go.Scatter3d(x=[xyz[i,0],xyz[j,0]],y=[xyz[i,1],xyz[j,1]],
            z=[xyz[i,2],xyz[j,2]],mode='lines',line={'width':8,'color':'#84929b'},
            showlegend=False,hoverinfo='skip'))
    for k,name in enumerate(seed.symbols):
        r=xyz[k]
        fig.add_trace(go.Scatter3d(x=[r[0]],y=[r[1]],z=[r[2]],
            mode='markers+text',marker={'size':9 if name!='H' else 6,'color':'#4e96ba'},
            text=[f'{name}{k+1}'],textposition='top center',showlegend=False,
            hovertemplate=f'{name}{k+1}<extra></extra>'))
    centers=np.array([(xyz[b['a']]+xyz[b['b']])/2 for b in bonds]) if bonds else np.zeros((0,3))
    fig.add_trace(go.Scatter3d(
        x=centers[:,0],y=centers[:,1],z=centers[:,2],mode='markers',
        marker={'size':11,'symbol':'diamond','color':['#ec8b38' if b['index']==selection else '#5bceae' for b in bonds],
                'line':{'color':'#10212a','width':1}},
        customdata=[b['index'] for b in bonds],
        hovertemplate='Bond center · %{customdata}<extra>Click to select bond if supported</extra>',
        name='Select bond centers',showlegend=True))
    fig.update_layout(height=370,margin={'l':0,'r':0,'b':0,'t':15},
        scene={'xaxis_title':'x (Å)','yaxis_title':'y (Å)','zaxis_title':'z (Å)', 'aspectmode':'data'})
    return fig

def two_site_bond_functions(result, projected, atom_a, atom_b):
    """Return local contributions on two atoms, allowing coherent σ/π assembly."""
    labels=projected['labels']
    components=np.asarray(projected['components_ao'])
    left=[i for i,s in enumerate(labels) if s.split(':',1)[0]==atom_a]
    right=[i for i,s in enumerate(labels) if s.split(':',1)[0]==atom_b]
    if not left or not right:
        raise ValueError('No saved atomic contributions for this bond')
    return components[:,left].sum(axis=1), components[:,right].sum(axis=1)
