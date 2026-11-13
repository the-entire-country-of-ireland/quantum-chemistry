import numpy as np
from screen_models import sto_pair
from full_eht import PARAM,BOHR

def hybrid_projection(R,sa,sb,hybA='sp3',hybB='sp3',K=1.75,za=None,zb=None):
    za=PARAM[sa]['z'] if za is None else za;zb=PARAM[sb]['z'] if zb is None else zb
    def cv(hyb,sign):
        if hyb=='sp3':return np.array([.5,sign*np.sqrt(3)/2])
        if hyb=='sp2':return np.array([1/np.sqrt(3),sign*np.sqrt(2/3)])
        if hyb=='p':return np.array([0.,sign])
        if hyb=='s1':return np.array([1.])
    ca=cv(hybA,1);cb=cv(hybB,-1 if hybB not in ('p','s1') else 1)
    # p-p lateral uses px, nonhybrid
    ka=('s2','pz') if sa!='H' else ('s1',)
    kb=('s2','pz') if sb!='H' else ('s1',)
    ss=np.array([[sto_pair(R,za,zb,ia,ib)[0] for ib in kb] for ia in ka])
    EA=np.array([PARAM[sa]['on'][0],PARAM[sa]['on'][1]]) if sa!='H' else np.array([-13.6])
    EB=np.array([PARAM[sb]['on'][0],PARAM[sb]['on'][1]]) if sb!='H' else np.array([-13.6])
    S=float(ca@ss@cb)
    HAB=float(ca@(.5*K*(EA[:,None]+EB[None,:])*ss)@cb)
    HAA=float(np.sum(ca*ca*EA));HBB=float(np.sum(cb*cb*EB))
    t=(S*(HAA+HBB)/2-HAB)/(1-S*S)
    delta=(HBB-HAA)/np.sqrt(1-S*S)
    wB=.5*(1-delta/np.sqrt(delta*delta+4*t*t))
    return dict(S=S,HAA=HAA,HBB=HBB,HAB=HAB,t=t,delta=delta,wB=wB,S_element=ss)

if __name__=='__main__':
 for name,R,a,b,ha,hb in [('CO',1.427*BOHR,'C','O','sp3','sp3'),('CH',1.087*BOHR,'C','H','sp3','s1'),('CCsp3',1.543*BOHR,'C','C','sp3','sp3'),('CCsp2',1.339*BOHR,'C','C','sp2','sp2')]:
  d=hybrid_projection(R,a,b,ha,hb)
  print(name,d)
