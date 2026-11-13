import numpy as np
from scipy.optimize import minimize_scalar
from full_eht import solve

def localize(result,exponent=4,seed=0,random_start=False,passes=25):
    S=result['S']; C=result['C'][:,:result['occ']]
    e,V=np.linalg.eigh(S); O=(V*np.sqrt(e))@V.T
    B=O@C
    n=B.shape[1]
    Qa=[]
    for i in range(len(result['geo'])):
        inds=[j for j,(ia,_) in enumerate(result['orbital_map']) if ia==i]
        Qa.append(B[inds,:].T@B[inds,:])
    Q=np.array(Qa)
    U=np.eye(n)
    if random_start:
        rng=np.random.default_rng(seed)
        x,_=np.linalg.qr(rng.normal(size=(n,n)))
        U=x
        Q=np.einsum('ij,ajk,kl->ail',U.T,Q,U)
    def objective(Q):return float(np.sum(np.diagonal(Q,axis1=1,axis2=2)**exponent))
    last=objective(Q)
    for it in range(passes):
        gain=0
        for i in range(n):
            for j in range(i+1,n):
                # rotation i'=c i+s j, j'=-s i+c j
                def score(theta):
                    c=np.cos(theta);s=np.sin(theta)
                    a=Q[:,i,i]; b=Q[:,j,j];d=Q[:,i,j]
                    qii=c*c*a+s*s*b+2*c*s*d
                    qjj=s*s*a+c*c*b-2*c*s*d
                    return np.sum(qii**exponent+qjj**exponent)
                test=np.linspace(-np.pi/2,np.pi/2,37)
                scores=np.array([score(q) for q in test]); best=int(np.argmax(scores))
                if best==0 or best==len(test)-1:th=float(test[best])
                else:
                    r=minimize_scalar(lambda x:-score(x),bounds=(test[best-1],test[best+1]),method='bounded',options={'xatol':1e-10})
                    th=float(r.x)
                if score(th)<=score(0)+1e-13:continue
                c=np.cos(th);s=np.sin(th)
                V=np.array([[c,-s],[s,c]])
                inds=[i,j]
                Q[:,inds,:]=np.einsum('kj,akl->ajl',V,Q[:,inds,:])  # Q rows new = V^T rows; V indices? 
                Q[:,:,inds]=np.einsum('ajk,kl->ajl',Q[:,:,inds],V)
                U[:,inds]=U[:,inds]@V
        curr=objective(Q)
        if abs(curr-last)<1e-11:break
        last=curr
    # recompute exact Q
    q=[]
    Bl=B@U
    for atom in range(len(result['geo'])):
        inds=[j for j,(i,k) in enumerate(result['orbital_map']) if atom==i]
        q.append(np.sum(Bl[inds,:]**2,axis=0))
    q=np.array(q)
    assert np.max(abs(q.sum(axis=0)-1))<1e-6
    return {'q':q,'B_loc':Bl,'C_loc':C@U,'U':U,'objective':np.sum(q**exponent),'passes':it+1}

if __name__=='__main__':
 for which in ['methane','ethane','ethylene','methanol']:
  r=solve(which)
  best=None
  for seed in range(4):
   p=localize(r,random_start=(seed>0),seed=seed)
   if best is None or p['objective']>best['objective']:best=p
  print('\n',which,'OBJ',best['objective'],'passes',best['passes'])
  for i in range(r['occ']):
   v=best['q'][:,i]
   print('orbital',i,', '.join('%s%d:%.3f'%(r['geo'][j][0],j,float(v[j])) for j in np.argsort(-v)[:4]),'top2',round(sum(sorted(v,reverse=True)[:2]),3))
