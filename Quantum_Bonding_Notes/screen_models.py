import numpy as np
from scipy.special import roots_laguerre
from numpy.polynomial.legendre import leggauss
L,WL=roots_laguerre(72);NU,WN=leggauss(88)
HA2EV=27.211386245988

def sto_pair(R,zA,zB,kindA='sp3',kindB='sp3',dirA=1,dirB=-1):
    # STO n=2: ns radial r exp(-zeta r), p spatial coordinate exp(-zeta r).
    def f(kind,rr,zz,rho,zeta,d):
        if kind=='s1':return np.sqrt(zeta**3/np.pi)*np.ones_like(rr)
        A=np.sqrt(zeta**5/(3*np.pi))
        if kind=='sp3':return A*.5*(rr+3*d*zz)
        if kind=='sp2':return A*(rr+np.sqrt(6)*d*zz)/np.sqrt(3)
        if kind=='pz':return np.sqrt(zeta**5/np.pi)*zz
        if kind=='px':return np.sqrt(zeta**5/np.pi)*rho
        if kind=='s2':return A*rr
        raise ValueError(kind)
    la=zA; lb=zB
    aA=0.0
    for mode in ['ab','aa','bb']:
        pass
    def integ(mode):
        if mode=='ab':al,be=la,lb
        if mode=='aa':al,be=2*la,0.
        if mode=='bb':al,be=0.,2*lb
        c=(al+be)*R/2; d=(al-be)*R/2
        mu=1+L[:,None]/c;nu=NU[None,:]
        rA=R/2*(mu+nu);rB=R/2*(mu-nu)
        zz=R/2*mu*nu; rho=R/2*np.sqrt(np.maximum(0,(mu*mu-1)*(1-nu*nu)))
        av=f(kindA,rA,zz+R/2,rho,zA,dirA)
        bv=f(kindB,rB,zz-R/2,rho,zB,dirB)
        p= av*bv if mode=='ab' else (av*av if mode=='aa' else bv*bv)
        fac=.5 if (mode=='ab' and (kindA=='px' or kindB=='px')) or (mode=='aa' and kindA=='px') or (mode=='bb' and kindB=='px') else 1
        return float(2*np.pi*fac*np.exp(-c)/c*np.sum(WL[:,None]*WN[None,:]*np.exp(-d*nu)*p*(R**3/8)*(mu**2-nu**2)))
    return integ('ab'),integ('aa'),integ('bb')

def e_ht(S,EA,EB,K=1.75):
    Haa=EA; Hbb=EB;Hab=K*S*(EA+EB)/2
    t=(S*(Haa+Hbb)/2-Hab)/(1-S*S)
    delta=(Hbb-Haa)/np.sqrt(1-S*S)
    wB=.5*(1-delta/np.sqrt(delta*delta+4*t*t))
    val=np.linalg.eigh(np.linalg.multi_dot([np.array([[1.,S],[S,1.]]),np.eye(2)]))[0]
    return t,delta,wB,Hab

def run():
 print('equivalent Z=1 STO2, E=-3.4 eV:')
 for R in [2.5,3,4,5,6,7,9]:
  ss,n1,n2=sto_pair(R,.5,.5,'sp3','sp3')
  sp,n3,n4=sto_pair(R,.5,.5,'px','px')
  print('R %.2f Sigma S %.5f norm%.5f Pi S %.5f ratio abs S %.3f'%(R,ss,n1,sp,abs(ss/sp)))
 print('equivalent carbon-like zeta=1.6:')
 for R in [2.5,2.8,3.0,3.5,4,5]:
  ss,*_=sto_pair(R,1.6,1.6,'sp3','sp3')
  sp,*_=sto_pair(R,1.6,1.6,'px','px')
  print('R %.2f Sigma S %.5f Pi S %.5f ratio %.3f'%(R,ss,sp,abs(ss/sp)))
 print('CO lower state B=oxygen, E_C -13.9 E_O -19.18:')
 for R in [2.5,2.7,3.0,3.5]:
  s,*_=sto_pair(R,1.625,2.275,'sp3','sp3')
  print('R %.2f S %.4f vals %s'%(R,s,str(e_ht(s,-13.9,-19.18))))
 print('CH E_C -13.9 H-13.6')
 for R in [1.9,2.06,2.2,2.5]:
  s,*_=sto_pair(R,1.625,1.0,'sp3','s1')
  print('R %.2f S %.4f vals %s'%(R,s,str(e_ht(s,-13.9,-13.6))))
if __name__=='__main__':run()
