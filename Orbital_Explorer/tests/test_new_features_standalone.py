"""Check the bond-angle UI helper and basis grouping with no SCF installation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]


def extract(path,name,namespace):
    tree=ast.parse(path.read_text())
    item=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[item],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def test_water_angle_helper():
    angle=extract(ROOT/'qchem_analysis/fixed_geometry_data.py','bond_angles',{'np':np,'MolecularSeed':object})
    mol=Chem.AddHs(Chem.MolFromSmiles('O'))
    seed=SimpleNamespace(rdkit_mol=mol,symbols=['O','H','H'])
    theta=np.deg2rad(104.478)
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[np.cos(theta),np.sin(theta),0.]])
    tab=angle(seed,xyz)
    assert len(tab)==1 and abs(tab[0]['Degrees (°)']-104.478)<.001



def test_ingredient_isosurfaces_remain_function_resolved():
    """Selected terms are plotted independently, rather than summed by atom."""
    import plotly.graph_objects as go
    source=ROOT/'qchem_analysis/molecular_viewer.py'
    # Pure visualization helper, avoiding a PySCF dependency in tests.
    namespace={'np':np,'go':go,'BLUE':'#2878b5','RED':'#d34a4a'}
    namespace['_add_molecular_skeleton']=lambda fig,result,seed:None
    namespace['grid_for_orbital']=lambda *args,**kwargs: (None,None)
    view=extract(source,'plot_local_ingredients',namespace)
    class Mol:
        def eval_gto(self,kind,coordinates):
            x,y,z=coordinates.T
            return np.column_stack((np.exp(-((x+1)**2+y*y+z*z)),
                                    np.exp(-((x-1)**2+y*y+z*z))))
    result=SimpleNamespace(mol=Mol())
    proj={'labels':['C1:2py','C2:2py'],
          'components_ao':np.array([[1.,0.],[0.,1.]]),
          'projected_ao':np.array([1.,1.])}
    fig=view(result,proj,['C1:2py','C2:2py'],extent_bohr=3.,
             grid_points=25,amplitude_cutoff=.05,opacity=.4)
    labels=[t.name for t in fig.data if t.type=='isosurface']
    assert any('C1:2py' in x for x in labels)
    assert any('C2:2py' in x for x in labels)
    assert len(labels)>=2
