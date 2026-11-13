"""Minimal extended Hückel for H,C,O using independently normalized Slater valence orbitals.
No self-consistency, core, electron correlation, geometry optimization, or rigorous dissociation energies.
"""
import sys, numpy as np
from scipy.linalg import eigh,fractional_matrix_power
from functools import lru_cache
from screen_models import sto_pair
BOHR=1.88972612462577
PARAM={'H': {'z':1.30,'on':[-13.6]},'C':{'z':1.625,'on':[-21.4,-11.4,-11.4,-11.4]},'O':{'z':2.275,'on':[-32.38,-15.85,-15.85,-15.85]}}
ELECTRONS={'H':1,'C':4,'O':6}

def geometry(which):
    def atom(sym,xy):return (sym,np.array(xy,dtype=float)*BOHR)
    tet=[np.array([np.cos(2*np.pi*j/3),np.sin(2*np.pi*j/3),-1/np.sqrt(8)]) for j in range(3)]
    tet=[v/np.linalg.norm(v) for v in tet]
    if which=='methane':
        vec=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],float)/np.sqrt(3)
        return [atom('C',[0,0,0])]+[atom('H',1.087*v) for v in vec]
    if which=='ethane':
        c1=np.array([0,0,-1.543/2]); c2=np.array([0,0,1.543/2]); L=1.09
        v1=[np.array([np.sqrt(8)/3*np.cos(2*np.pi*j/3),np.sqrt(8)/3*np.sin(2*np.pi*j/3),-1/3]) for j in range(3)]
        v2=[np.array([np.sqrt(8)/3*np.cos(np.pi/3+2*np.pi*j/3),np.sqrt(8)/3*np.sin(np.pi/3+2*np.pi*j/3),1/3]) for j in range(3)]
        return [atom('C',c1),atom('C',c2)]+[atom('H',c1+L*v) for v in v1]+[atom('H',c2+L*v) for v in v2]
    if which=='ethylene':
        c1=np.array([-1.339/2,0,0]);c2=np.array([1.339/2,0,0]);L=1.087
        vv1=[np.array([-.5,t*np.sqrt(3)/2,0]) for t in [-1,1]]
        vv2=[np.array([+.5,t*np.sqrt(3)/2,0]) for t in [-1,1]]
        return [atom('C',c1),atom('C',c2)]+[atom('H',c1+L*v) for v in vv1]+[atom('H',c2+L*v) for v in vv2]
    if which=='methanol':
        c=np.array([0,0,-1.427/2]);o=np.array([0,0,1.427/2]);L=1.09
        v1=[np.array([np.sqrt(8)/3*np.cos(2*np.pi*j/3),np.sqrt(8)/3*np.sin(2*np.pi*j/3),-1/3]) for j in range(3)]
        vOH=np.array([np.sin(np.deg2rad(108.5)),0,-np.cos(np.deg2rad(108.5))])
        # angle O-C vs O-H 108.5: O-C=-z, dot=-z=cos108.5=-.317 -> +z .317
        vOH=np.array([np.sin(np.deg2rad(108.5)),0,-np.cos(np.deg2rad(108.5))])
        return [atom('C',c),atom('O',o)]+[atom('H',c+L*v) for v in v1]+[atom('H',o+.96*vOH)]
    raise ValueError(which)

@lru_cache(None)
def get_basic(dist,za,zb,kindA,kindB):
    return sto_pair(dist,za,zb,kindA,kindB)[0]

def overlap_pair(symA,symB,v):
    dist=np.linalg.norm(v); n=v/dist
    za=PARAM[symA]['z'];zb=PARAM[symB]['z']
    qa=4 if symA!='H' else 1
    qb=4 if symB!='H' else 1
    out=np.zeros((qa,qb))
    ka='s1' if symA=='H' else 's2';kb='s1' if symB=='H' else 's2'
    out[0,0]=get_basic(round(dist,10),za,zb,ka,kb)
    if qb>1:
        s_p=get_basic(round(dist,10),za,zb,ka,'pz')
        out[0,1:]=n*s_p
    if qa>1:
        p_s=get_basic(round(dist,10),za,zb,'pz',kb)
        out[1:,0]=n*p_s
    if qa>1 and qb>1:
        pzpz=get_basic(round(dist,10),za,zb,'pz','pz')
        pxpx=get_basic(round(dist,10),za,zb,'px','px')
        out[1:,1:]=pxpx*np.eye(3)+(pzpz-pxpx)*np.outer(n,n)
    return out

def solve(which,K=1.75):
    geo=geometry(which)
    lookup=[];start=[];P=[]
    for i,(sym,loc) in enumerate(geo):
        start.append(len(lookup))
        for j,E in enumerate(PARAM[sym]['on']):
            lookup.append((i,j));P.append(E)
    na=len(P);S=np.eye(na)
    for i,(sa,xa) in enumerate(geo):
        for j,(sb,xb) in enumerate(geo):
            if j<=i:continue
            block=overlap_pair(sa,sb,xb-xa)
            ia=start[i];ib=start[j]
            S[ia:ia+block.shape[0],ib:ib+block.shape[1]]=block
            S[ib:ib+block.shape[1],ia:ia+block.shape[0]]=block.T
    E=np.array(P)
    H=K*.5*(E[:,None]+E[None,:])*S
    np.fill_diagonal(H,E)
    ev,C=eigh(H,S)
    nv=sum(ELECTRONS[sa] for sa,loc in geo)
    occ=nv//2
    density=2*C[:,:occ]@C[:,:occ].T
    charges=[]
    for i,(sym,loc) in enumerate(geo):
        inds=[n for n,(ii,j) in enumerate(lookup) if i==ii]
        pop=np.sum(density[inds,:]*S[inds,:])
        charges.append(ELECTRONS[sym]-pop)
    Ssq=np.linalg.eigh(S)
    ss=(Ssq[1]*np.sqrt(Ssq[0]))@Ssq[1].T
    PS=ss@density@ss
    bond_orders={}
    for i in range(len(geo)):
        for j in range(i+1,len(geo)):
            ii=[n for n,(ix,jx) in enumerate(lookup) if ix==i]
            jj=[n for n,(ix,jx) in enumerate(lookup) if ix==j]
            bond_orders[(i,j)]=np.sum(PS[np.ix_(ii,jj)]**2)
    return {'which':which,'geo':geo,'ev':ev,'C':C,'S':S,'H':H,'dens':density,'charges':charges,'BO':bond_orders,'occ':occ,'orbital_map':lookup,'PS':PS}

if __name__=='__main__':
    for which in ['methane','ethane','ethylene','methanol']:
        m=solve(which)
        print('\n',which,'occ',m['occ'],'low S eig',np.min(np.linalg.eigvalsh(m['S'])))
        print('Energies',m['ev'].round(2))
        print('charges', [(s,round(q,3)) for (s,x),q in zip(m['geo'],m['charges'])])
        print('BO',[(k,round(v,3)) for k,v in m['BO'].items() if v>.1])
