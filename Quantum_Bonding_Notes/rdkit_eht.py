import numpy as np
from rdkit import Chem
from rdkit.Chem import rdEHTTools
from full_eht import geometry
smiles={'methane':'C','ethane':'CC','ethylene':'C=C','methanol':'CO'}
for which in smiles:
 m=Chem.AddHs(Chem.MolFromSmiles(smiles[which]))
 g=geometry(which)
 assert [a.GetSymbol() for a in m.GetAtoms()]==[s for s,x in g]
 conf=Chem.Conformer(len(g))
 for i,(s,pos) in enumerate(g):conf.SetAtomPosition(i,pos/1.88972612462577)
 m.AddConformer(conf)
 status,res=rdEHTTools.RunMol(m,keepOverlapAndHamiltonianMatrices=True)
 print('\n',which,'status',status)
 print('charges',list(np.round(res.GetAtomicCharges(),3)))
 print('energies',list(np.round(res.GetOrbitalEnergies(),2)))
 mtrx=np.asarray(res.GetReducedOverlapPopulationMatrix())
 print('Reduced OP',mtrx.shape, np.round(mtrx,2))
