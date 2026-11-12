"""Reproducible numerical figures for quantum bonding notes.
Atomic units except for the explicitly dimensional 1D derivation.
"""
import os, json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_laguerre
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE=os.path.dirname(os.path.abspath(__file__))
FIG=os.path.join(HERE,'figures')
os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':165,'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.13})
C1='#2368a0'; C2='#cf7045'; C3='#37967a'; C4='#9465ae'

def save(name,fig):
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG,name+'.png'),bbox_inches='tight')
    plt.close(fig)

def delta_root(R, odd=False, k=1):
    if odd:
        if R*k<=1: return np.nan
        f=lambda q: q-k*(1-np.exp(-q*R))
        return brentq(f,1e-10,k,xtol=1e-13)
    f=lambda q: q-k*(1+np.exp(-q*R))
    return brentq(f,k,2*k,xtol=1e-13)

def delta_exact(x,R,odd=False):
    q=delta_root(R,odd)
    if np.isnan(q):return np.full_like(x,np.nan)
    s=-1 if odd else 1
    N=np.sqrt(q/(2*(1+s*(1+q*R)*np.exp(-q*R))))
    return N*(np.exp(-q*np.abs(x+R/2))+s*np.exp(-q*np.abs(x-R/2)))

def delta_lcao(x,R,odd=False):
    S=(1+R)*np.exp(-R)
    s=-1 if odd else 1
    return (np.exp(-np.abs(x+R/2))+s*np.exp(-np.abs(x-R/2)))/np.sqrt(2*(1+s*S))

def gauss_a(x,R,sigma):
    return (2*np.pi*sigma*sigma)**(-.25)*np.exp(-(x+R/2)**2/(4*sigma*sigma))

def gauss_b(x,R,sigma):
    return (2*np.pi*sigma*sigma)**(-.25)*np.exp(-(x-R/2)**2/(4*sigma*sigma))

def gauss_matrix(R,sigma,g=1):
    S=np.exp(-R*R/(8*sigma*sigma))
    A=g/(np.sqrt(2*np.pi)*sigma)
    haa=1/(8*sigma*sigma)-A*(1+np.exp(-R*R/(2*sigma*sigma)))
    hab=S*(1/(8*sigma*sigma)-R*R/(32*sigma**4))-2*A*np.exp(-R*R/(4*sigma*sigma))
    return S,haa,hab

def gauss_energy(R,sigma,odd=False):
    S,aa,ab=gauss_matrix(R,sigma)
    s=-1 if odd else 1
    return (aa+s*ab)/(1+s*S)

def opt_gauss(R,odd=False):
    res=minimize_scalar(lambda log_sigma:gauss_energy(R,np.exp(log_sigma),odd),bounds=(np.log(.16),np.log(7)),method='bounded',options={'xatol':1e-11})
    return np.exp(res.x),res.fun

def hydro1s(x,R):
    return np.exp(-np.abs(x+R/2))/np.sqrt(np.pi),np.exp(-np.abs(x-R/2))/np.sqrt(np.pi)

def h2plus(R):
    S=np.exp(-R)*(1+R+R*R/3)
    J=(1-(1+R)*np.exp(-2*R))/R
    K=np.exp(-R)*(1+R)
    alpha=-.5-J
    beta=-S*.5-K
    ep=(alpha+beta)/(1+S)+1/R
    em=(alpha-beta)/(1-S)+1/R
    t=(alpha*S-beta)/(1-S*S)
    return S,J,K,t,ep,em

# Axis slices and electron exchange for 1s H2
R=1.6
x=np.linspace(-5,5,420)
a,b=hydro1s(x,R)
S=np.exp(-R)*(1+R+R*R/3)
fig,ax=plt.subplots(1,1,figsize=(8.7,3.6))
ax.plot(x,a,color=C1,label=r'$1s_A$');ax.plot(x,b,color=C2,label=r'$1s_B$')
ax.plot(x,(a+b)/np.sqrt(2*(1+S)),color=C3,label=r'$(a+b)/\sqrt{2(1+S)}$')
ax.plot(x,(a-b)/np.sqrt(2*(1-S)),color=C4,label=r'$(a-b)/\sqrt{2(1-S)}$')
for v in [-R/2,R/2]:ax.axvline(v,color='gray',ls=':',lw=1)
ax.set(xlabel=r'Bond-axis coordinate $z/a_0$',ylabel='Orbital amplitude on axis',title=r'One-electron orbital symmetry ($R=1.6a_0$)');ax.legend(fontsize=7,loc='upper right')
xx=np.linspace(-3.5,3.5,180)
aa,bb=hydro1s(xx,R)
A=aa[:,None]*bb[None,:]; B=bb[:,None]*aa[None,:]
psiS=(A+B)/np.sqrt(2*(1+S*S));psiT=(A-B)/np.sqrt(2*(1-S*S))
# show full signed wavefunctions on separate figure
save('01_h2_atomic_orbital_axis',fig)
fig,axs=plt.subplots(1,2,figsize=(10.4,4.4),sharex=True,sharey=True)
for ax,data,title in zip(axs,[psiS,psiT],['Covalent singlet: symmetric space','Covalent triplet: antisymmetric space']):
    mx=max(abs(psiS).max(),abs(psiT).max()); im=ax.imshow(data.T,extent=[-3.5,3.5,-3.5,3.5],origin='lower',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx),aspect='equal',interpolation='bilinear')
    ax.plot([-3.5,3.5],[-3.5,3.5],ls=':',color='black',lw=1,alpha=.6)
    ax.set(xlabel=r'$z_1/a_0$',title=title)
axs[0].set_ylabel(r'$z_2/a_0$')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.80,label='Signed 3D 1s orbital amplitudes, axial slice')
fig.subplots_adjust(left=.08,right=.88,bottom=.12,top=.85,wspace=.15)
fig.savefig(os.path.join(FIG,'02_h2_exchange_nodes.png'),bbox_inches='tight',dpi=165);plt.close(fig)

rs=np.linspace(.8,7,200)
vals=np.array([h2plus(r) for r in rs])
fig,axs=plt.subplots(1,2,figsize=(10.8,3.8))
for j,lab,col in [(0,'Overlap S',C1),(1,'Cross attraction J',C2),(2,'Resonance K',C3),(3,'Orthogonalized hopping t',C4)]:axs[0].plot(rs,vals[:,j],label=lab,color=col)
axs[0].legend(fontsize=8);axs[0].set(xlabel=r'$R/a_0$',ylabel='Atomic units',title=r'$\mathrm{H}_2^+$: overlap versus Hamiltonian coupling')
axs[1].plot(rs,vals[:,4],color=C1,label='Bonding total energy')
axs[1].plot(rs,vals[:,5],color=C2,label='Antibonding total energy')
axs[1].axhline(-.5,color='black',ls='--',lw=1,label='H + H$^+$ limit')
axs[1].set(ylim=(-.7,.35),xlabel=r'$R/a_0$',ylabel='Energy (Ha)',title=r'Fixed-$1s$ LCAO molecular energies');axs[1].legend(fontsize=8)
save('03_h2plus_overlap_energy',fig)

# two-site eps and S; energy reference can remain nonzero without hopping
es=np.linspace(0,.9,150)
ss=2*es/(1+es**2)
fig,axs=plt.subplots(1,2,figsize=(10.5,3.5))
axs[0].plot(es,ss,color=C1,lw=2);axs[0].plot(es,es,color='gray',ls='--',label=r'$S=\epsilon$ (not true)')
axs[0].set(xlabel=r'Atomic leakage amplitude $\epsilon$',ylabel='Overlap S',title=r'$S=2\epsilon/(1+\epsilon^2)$');axs[0].legend()
ts=np.linspace(0,1.1,100)
axs[1].plot(ts,-.5-ts,label=r'$E_+=E_0-t$',color=C1)
axs[1].plot(ts,-.5+ts,label=r'$E_-=E_0+t$',color=C2)
axs[1].set(xlabel='Physical hopping t',ylabel='Eigenenergy',title='Splitting is controlled by H, not by S');axs[1].legend()
save('04_discrete_overlap_and_hopping',fig)

# delta exact vs LCAO
fig,axs=plt.subplots(2,3,figsize=(11.2,6.4),sharex=True)
xx=np.linspace(-7,7,800)
for j,R in enumerate([1.3,2.5,5.0]):
    for row,odd in [(0,False),(1,True)]:
        ax=axs[row,j]
        ax.plot(xx,delta_exact(xx,R,odd),color=C1,lw=2.3,label='Exact')
        ax.plot(xx,delta_lcao(xx,R,odd),color=C2,ls='--',lw=1.9,label='LCAO')
        for a0 in [-R/2,R/2]:ax.axvline(a0,color='gray',ls=':',lw=.8)
        ax.axhline(0,color='gray',lw=.5)
        ax.set(title=(r'Even (bonding), ' if row==0 else r'Odd (antibonding), ')+rf'$R={R}$',xlabel=r'$x\kappa$',ylabel=r'$\psi(x)/\sqrt{\kappa}$')
        if j==0 and row==0:ax.legend(fontsize=8)
save('05_delta_exact_vs_lcao',fig)
rs=np.linspace(.65,7,250)
qp=np.array([delta_root(r) for r in rs]);qm=np.array([delta_root(r,True) for r in rs])
fig,axs=plt.subplots(1,2,figsize=(10.8,3.7))
axs[0].plot(rs,qp,color=C1,label=r'$q_+$ exact');axs[0].plot(rs,qm,color=C2,label=r'$q_-$ exact')
axs[0].axhline(1,color='gray',ls='--',label=r'Atomic $\kappa=1$');axs[0].axvline(1,color='gray',ls=':',lw=1)
axs[0].set(xlabel='Separation R',ylabel='Decay constant q',title='Exact orbital relaxation');axs[0].legend(fontsize=8)
axs[1].plot(rs,-.5*qp**2,color=C1,label='Exact bonding')
axs[1].plot(rs,-.5*qm**2,color=C2,label='Exact antibonding')
axs[1].axhline(-.5,color='gray',ls='--',label='Isolated well')
axs[1].axvline(1,color='gray',ls=':',lw=1)
axs[1].set(xlabel='Separation R',ylabel='Bound-state energy',title='Odd bound state exists only for R > 1');axs[1].legend(fontsize=8)
save('06_delta_decay_energies',fig)

sigmas=np.linspace(.25,2,300)
EG=1/(8*sigmas**2)-1/(np.sqrt(2*np.pi)*sigmas)
sig0=np.sqrt(2*np.pi)/4
fig,axs=plt.subplots(1,2,figsize=(10.8,3.6))
axs[0].plot(sigmas,EG,color=C1,label='Gaussian trial energy');axs[0].axhline(-.5,color=C2,ls='--',label='Exact one-delta energy')
axs[0].axvline(sig0,color='gray',ls=':',lw=1)
axs[0].set(xlabel=r'Gaussian width $\sigma$',ylabel='Energy (a.u.)',ylim=(-.55,.2),title='Variational principle in an isolated delta well');axs[0].legend(fontsize=8)
R=2.5
for odd,color,label in [(False,C1,'even'),(True,C2,'odd')]:
    sigma,En=opt_gauss(R,odd)
    sg=1 if not odd else -1
    ss=np.exp(-R*R/(8*sigma*sigma))
    a=gauss_a(xx,R,sigma);b=gauss_b(xx,R,sigma)
    trial=(a+sg*b)/np.sqrt(2*(1+sg*ss))
    axs[1].plot(xx,delta_exact(xx,R,odd),color=color,label=f'Exact {label}')
    axs[1].plot(xx,trial,color=color,ls='--',label=f'Optimized Gaussian {label}')
axs[1].set(xlabel='Position x',ylabel='Normalized amplitude',title='Exact vs optimized Gaussian, R=2.5');axs[1].legend(fontsize=7,loc='upper right')
save('07_gaussian_variational',fig)

tr=np.linspace(0,.8,160)
Es=(1-np.sqrt(1+16*tr*tr))/2
pert=-4*tr**2
ionic=.5*(1-1/np.sqrt(1+16*tr**2))
fig,axs=plt.subplots(1,2,figsize=(10.5,3.5))
axs[0].plot(tr,Es,color=C1,label='Exact singlet');axs[0].plot(tr,pert,color=C2,ls='--',label=r'Second order: $-4t^2/U$');axs[0].axhline(0,color=C4,ls=':',label='Triplet')
axs[0].set(xlabel=r'$t/U$',ylabel=r'$E/U$',title='Hubbard dimer: exact versus perturbative');axs[0].legend(fontsize=8)
axs[1].plot(tr,ionic,color=C3);axs[1].set(xlabel=r'$t/U$',ylabel='Ionic probability',ylim=(0,.5),title='Increasing charge-transfer admixture')
save('08_hubbard_exchange',fig)

# 3D atomic orbitals for n=1,2, n=2 hybrids
# Axis along z; rho perpendicular cylindrical coordinate
c=np.sqrt(1/(32*np.pi))
def orb2(z,rho,Z,kind,center=0,direction=1):
    zz=z-center
    rr=np.sqrt(zz*zz+rho*rho)
    pref=np.sqrt(Z**3/(32*np.pi))*np.exp(-Z*rr/2)
    s=pref*(2-Z*rr)
    pz=pref*(Z*zz)
    px=pref*(Z*rho) # x=rho cos(phi), evaluated at phi=0
    if kind=='s2':return s
    if kind=='pz':return pz
    if kind=='px':return px
    if kind=='sp3':return .5*(s+np.sqrt(3)*direction*pz)
    if kind=='s1':return np.sqrt(Z**3/np.pi)*np.exp(-Z*rr)
    raise ValueError(kind)

# Gaussian quadrature cylindrical z/rho, phi integrated analytically.
# For sigma and H 1s: phi independent. For p_x p_x: angular average cos^2(phi)=1/2.
# Prolate-spheroidal quadrature for two-center Coulomb integrals.
# Here mu=(r_A+r_B)/R, nu=(r_A-r_B)/R and
# dV = (R^3/8)(mu^2-nu^2) dmu dnu dphi.
# The mu exponent is integrated with Gauss-Laguerre quadrature.
LAG,WLAG=roots_laguerre(90)
NU,WNU=leggauss(110)

def two_center(R,Za,Zb,kindA='sp3',kindB='sp3',dirA=1,dirB=-1):
    lamA=Za/(1 if kindA=='s1' else 2)
    lamB=Zb/(1 if kindB=='s1' else 2)
    def polyn(kind,zz,rr,rho,Z,direction):
        if kind=='s1':return np.sqrt(Z**3/np.pi)*np.ones_like(rr)
        pref=np.sqrt(Z**3/(32*np.pi))
        if kind=='sp3':return pref*.5*((2-Z*rr)+np.sqrt(3)*direction*Z*zz)
        if kind=='px':return pref*Z*rho
        if kind=='s2':return pref*(2-Z*rr)
        if kind=='pz':return pref*Z*zz
        raise ValueError(kind)
    def integrate(mode,power):
        if mode=='ab':alpha,beta=lamA,lamB
        elif mode=='aa':alpha,beta=2*lamA,0.
        elif mode=='bb':alpha,beta=0.,2*lamB
        else:raise ValueError(mode)
        c=(alpha+beta)*R/2
        d=(alpha-beta)*R/2
        mu=1+LAG[:,None]/c
        nu=NU[None,:]
        rA=(R/2)*(mu+nu)
        rB=(R/2)*(mu-nu)
        zz=(R/2)*mu*nu
        rho=(R/2)*np.sqrt(np.maximum(0,(mu*mu-1)*(1-nu*nu)))
        a=polyn(kindA,zz+R/2,rA,rho,Za,dirA)
        b=polyn(kindB,zz-R/2,rB,rho,Zb,dirB)
        if mode=='aa':v=a*a
        elif mode=='bb':v=b*b
        else:v=a*b
        fac=.5 if kindA=='px' and kindB=='px' and mode=='ab' else (0.5 if (mode=='aa' and kindA=='px') or (mode=='bb' and kindB=='px') else 1.)
        if power=='A':v=v/rA
        elif power=='B':v=v/rB
        measure=(R**3/8)*(mu*mu-nu*nu)
        return float(2*np.pi*fac*np.exp(-c)/c*np.sum(WLAG[:,None]*WNU[None,:]*np.exp(-d*nu)*v*measure))
    S=integrate('ab',None)
    normA=integrate('aa',None);normB=integrate('bb',None)
    KA=integrate('ab','A');KB=integrate('ab','B')
    JA=integrate('aa','B');JB=integrate('bb','A')
    ea=-Za**2/(8 if kindA!='s1' else 2)
    eb=-Zb**2/(8 if kindB!='s1' else 2)
    haa=ea-Zb*JA
    hbb=eb-Za*JB
    habA=eb*S-Za*KA
    habB=ea*S-Zb*KB
    hab=(habA+habB)/2
    t=(S*(haa+hbb)/2-hab)/(1-S*S)
    delta=(hbb-haa)/np.sqrt(1-S*S)
    lower_weight_B=.5*(1-delta/np.sqrt(delta*delta+4*t*t))
    mu=(haa+hbb-2*S*hab)/(2*(1-S*S))
    return {'S':S,'normA':normA,'normB':normB,'KA':KA,'KB':KB,'JA':JA,'JB':JB,'haa':haa,'hbb':hbb,'hab':hab,'habA':habA,'habB':habB,'t_signed':t,'delta':delta,'weight_B':lower_weight_B,'e_low':mu-np.sqrt((delta/2)**2+t*t),'e_high':mu+np.sqrt((delta/2)**2+t*t)}

# orbital contour visualization: sigma and pi
z=np.linspace(-7.5,7.5,300); x=np.linspace(-6,6,240)
zz,xx2=np.meshgrid(z,x,indexing='xy')
R=5
sigA=orb2(zz,abs(xx2),1,'sp3',-R/2,1)
sigB=orb2(zz,abs(xx2),1,'sp3',+R/2,-1)
piA=orb2(zz,xx2,1,'px',-R/2,1)
piB=orb2(zz,xx2,1,'px',+R/2,1)
fig,axs=plt.subplots(2,2,figsize=(10.7,6.4),sharex=True,sharey=True)
for ax,arr,title in [(axs[0,0],sigA,'Bond-directed $sp^3$ on A'),(axs[0,1],sigB,'Bond-directed $sp^3$ on B'),(axs[1,0],piA,'Transverse $2p_x$ on A'),(axs[1,1],piB,'Transverse $2p_x$ on B')]:
    vmax=max(np.max(np.abs(sigA)),np.max(np.abs(sigB)),np.max(np.abs(piA)),np.max(np.abs(piB)))
    ax.contourf(z,x,arr,levels=np.linspace(-vmax,vmax,31),cmap='RdBu_r',extend='both')
    ax.axhline(0,color='black',alpha=.45,lw=.65)
    ax.plot([-R/2,R/2],[0,0],'ko',ms=3)
    ax.set(title=title,xlabel=r'Bond-axis coordinate $z/a_0$',ylabel=r'Transverse coordinate $x/a_0$',xlim=(-7.5,7.5),ylim=(-6,6))
save('09_sigma_pi_orbital_shapes',fig)

# Numeric data for all instances
pairs=[('equal_sigma',1,1,'sp3','sp3',1,-1,5.0),('equal_pi',1,1,'px','px',1,1,5.0),('carbon_oxygen',3.25,4.55,'sp3','sp3',1,-1,2.7),('carbon_hydrogen',3.25,1,'sp3','s1',1,1,2.1)]
results={}
for name,za,zb,ka,kb,da,db,r0 in pairs:
    v=two_center(r0,za,zb,ka,kb,da,db)
    results[name]={'R':r0,'Z_A':za,'Z_B':zb,**{k:float(x) for k,x in v.items()}}
    print(name, 'R=',r0, 'S=',v['S'], 'HAB=',v['hab'], 't=',v['t_signed'],'Bweight=',v['weight_B'],'hAB_error=',abs(v['habA']-v['habB']),'norms=',v['normA'],v['normB'])
rs=np.linspace(1.8,13,70)
vp=np.array([[[two_center(r,1,1,kindA=kind,kindB=kind,dirA=1,dirB=(-1 if kind=='sp3' else 1))[key] for key in ['S','t_signed','e_low','e_high']] for r in rs] for kind in ['sp3','px']])
fig,axs=plt.subplots(1,2,figsize=(11,3.7))
for i,name,col in [(0,r'$sp^3$-$sp^3$ $\sigma$',C1),(1,r'$p_x$-$p_x$ $\pi$',C2)]:
    axs[0].plot(rs,vp[i,:,0],color=col,label=name)
    axs[1].plot(rs,vp[i,:,1],color=col,label=name)
axs[0].set(xlabel=r'$R/a_0$',ylabel='Signed overlap S',title='Z = 1 hydrogenic n=2 trial orbitals')
axs[1].set(xlabel=r'$R/a_0$',ylabel='Signed orthogonalized hopping t (Ha)',title='Hopping is not a universal function of overlap')
for ax in axs:ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=9)
save('10_equal_charge_sigma_pi_coupling',fig)

fig,axs=plt.subplots(1,2,figsize=(10.8,3.8))
for i,name,col in [(0,r'$sp^3$ $\sigma$',C1),(1,r'$p_x$ $\pi$',C2)]:
    axs[0].plot(rs,vp[i,:,2],color=col,label=name+' lower')
    axs[0].plot(rs,vp[i,:,3],color=col,ls='--',label=name+' upper')
axs[0].set(xlabel=r'$R/a_0$',ylabel='Electronic eigenenergy (Ha)',title='Projected two-orbital energies, Z=1')
axs[0].legend(fontsize=8)
# show center-plane orbitals for sigma and pi at R=5
xx=np.linspace(-6,6,401)
for kind,col,lab in [('sp3',C1,r'$sp^3$-$sp^3$ $\sigma$'),('px',C2,r'$p_x$-$p_x$ $\pi$, at x=1')]:
    rho=np.zeros_like(xx) if kind=='sp3' else np.ones_like(xx)
    a=orb2(xx,rho,1,kind,-2.5,1)
    b=orb2(xx,rho,1,kind,+2.5,(-1 if kind=='sp3' else 1))
    sgn=(-1 if kind=='sp3' else 1)
    ov=two_center(5,1,1,kindA=kind,kindB=kind,dirA=1,dirB=(-1 if kind=='sp3' else 1))['S']
    axs[1].plot(xx,(a+sgn*b)/np.sqrt(2*(1+sgn*ov)),color=col,label=lab)
axs[1].set(xlabel='Bond-axis coordinate z',ylabel='Normalized MO axial amplitude',title='Lower-energy signed combinations: orbital sections')
axs[1].legend(fontsize=8)
save('11_sigma_pi_levels',fig)

fig,axs=plt.subplots(1,2,figsize=(10.8,3.7))
for name,Za,Zb,R0,col,lab in [('carbon_oxygen',3.25,4.55,2.7,C1,'C-O hybrid pair'),('carbon_hydrogen',3.25,1.,2.1,C2,'C-H hybrid/1s')]:
    kindB='sp3' if name=='carbon_oxygen' else 's1'
    rarr=np.linspace(1.2,6,65)
    v=[two_center(r,Za,Zb,'sp3',kindB,1,-1) for r in rarr]
    axs[0].plot(rarr,[q['S'] for q in v],color=col,label=lab)
    axs[1].plot(rarr,[q['weight_B'] for q in v],color=col,label=lab)
    axs[1].axvline(R0,color=col,ls=':',alpha=.5)
axs[0].set(xlabel=r'$R/a_0$',ylabel='Signed atomic overlap S',title='Screened effective-charge overlap')
axs[1].set(xlabel=r'$R/a_0$',ylabel='Lower-state weight on B (Lowdin)',ylim=(0,1),title='Orbital polarization in two-state truncation')
for ax in axs:ax.legend(fontsize=8)
save('12_polarization_carbon_oxygen_ch',fig)

fig,axs=plt.subplots(1,2,figsize=(10.6,3.6))
for ax,name,ZA,ZB,R,kindB in [(axs[0],'C-O',3.25,4.55,2.7,'sp3'),(axs[1],'C-H',3.25,1,2.1,'s1')]:
    xx=np.linspace(-5,5,650)
    av=orb2(xx,np.zeros_like(xx),ZA,'sp3',-R/2,1)
    bv=orb2(xx,np.zeros_like(xx),ZB,kindB,R/2,-1)
    ax.plot(xx,av,color=C1,label='C hybrid')
    ax.plot(xx,bv,color=C2,label=('O hybrid' if name=='C-O' else 'H 1s'))
    for v in [-R/2,R/2]:ax.axvline(v,color='gray',ls=':',lw=.8)
    ax.axhline(0,color='gray',lw=.65)
    ax.set(title=name+f' signed orbital axis slices, R={R}',xlabel=r'$z/a_0$',ylabel='Amplitude on axis')
    ax.legend(fontsize=8)
save('13_heteronuclear_orbital_sections',fig)

with open(os.path.join(HERE,'numerical_results.json'),'w') as f:json.dump(results,f,indent=2)
print('Gaussian opt at R=2.5',opt_gauss(2.5,False),opt_gauss(2.5,True),'isolated sigma',sig0)
print('H2plus at R=2',h2plus(2))
