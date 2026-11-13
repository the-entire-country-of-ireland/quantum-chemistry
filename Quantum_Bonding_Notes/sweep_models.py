import numpy as np, json
from full_eht import PARAM,solve
from localize_eht import localize
from screen_models import sto_pair,e_ht

def bond_orbs(model,loc,indices):
    q=loc['q']
    return [(i,float(q[indices[0],i]),float(q[indices[1],i]),float(q[indices[0],i]+q[indices[1],i])) for i in range(q.shape[1]) if q[indices[0],i]+q[indices[1],i]>.85]

rows=[]
for K in [1.25,1.4,1.55,1.75,1.95,2.1]:
 for which in ['methane','ethane','ethylene','methanol']:
  m=solve(which,K); loc=localize(m)
  if which=='methane':
   data={'bond_pair_C':min([x[1] for x in bond_orbs(m,loc,(0,1))],default=0)}
   # C-H orbital pairing choose max atom H1
   j=np.argmax(loc['q'][1]);data['bond_pair_C']=float(loc['q'][0,j]);data['bond_pair_H']=float(loc['q'][1,j]);data['charge_C']=m['charges'][0]
  if which=='ethane':data={'BO_CC':m['BO'][(0,1)]}
  if which=='ethylene':
   c0=[n for n,(a,o) in enumerate(m['orbital_map']) if a==0 and o==3][0]
   c1=[n for n,(a,o) in enumerate(m['orbital_map']) if a==1 and o==3][0]
   data={'BO_CC':m['BO'][(0,1)],'BO_pi':m['PS'][c0,c1]**2}
  if which=='methanol':
   arr=bond_orbs(m,loc,(0,1));v=max(arr,key=lambda x:x[1]*x[2]);data={'bond_pair_O':v[2], 'bond_pair_C':v[1], 'BO_CO':m['BO'][(0,1)],'charge_O':m['charges'][1]}
  rows.append({'K':K,'which':which,**data})
  print(rows[-1],flush=True)
with open('sweep_results.json','w') as f:json.dump(rows,f,indent=2)
print('Orbital overlap sensitivity at bond lengths')
for f in [.9,1,1.1]:
 sCO,*_=sto_pair(1.427*1.889726,1.625*f,2.275*f,'sp3','sp3')
 sCH,*_=sto_pair(1.087*1.889726,1.625*f,1.30*f,'sp3','s1')
 print(f,'CO',sCO,e_ht(sCO,-13.9,-19.18)[2],'CH',sCH,e_ht(sCH,-13.9,-13.6)[2])
