"""Additional plots centered on spin-coupled valence bond (VB) theory.

Plots are explicitly labeled illustrative/slice as needed; no Hubbard
parameters are inferred from extended Huckel one-electron calculations.
"""
import os
import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from screen_models import sto_pair
from full_eht import solve, BOHR
from localize_eht import localize
OUT='figures'
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180,'axes.grid':False})
blue='#2368a0'; orange='#cf7045'; green='#37967a'

def sf(z,x,z0,zeta,kind,ori):
    zz=z-z0
    r=np.sqrt(zz*zz+x*x)
    A=np.sqrt(zeta**5/(3*np.pi))*np.exp(-zeta*r)
    if kind=='sp3': return A/2*(r+3*ori*zz)
    if kind=='sp2': return A/np.sqrt(3)*(r+np.sqrt(6)*ori*zz)
    if kind=='px': return A*np.sqrt(3)*x
    raise ValueError(kind)

# 15: actual 3D HL spatial wavefunction evaluated on fixed transverse cuts.
fig, axs=plt.subplots(2,2,figsize=(10.2,7.7),sharex=False,sharey=False)
settings=[('Equal-exponent $sp^3$--$sp^3$ $\\sigma$',6.0,.5,'sp3',0.0),
          ('$p_x$--$p_x$ $\\pi$',6.0,.5,'px',1.8)]
for i,(title,R,zeta,kind,xx) in enumerate(settings):
    S=float(sto_pair(R,zeta,zeta,kind,kind)[0]); z=np.linspace(-R/2-5,R/2+5,170)
    z1,z2=np.meshgrid(z,z)
    a1=sf(z1,xx,-R/2,zeta,kind,1);b1=sf(z1,xx,R/2,zeta,kind,-1)
    a2=sf(z2,xx,-R/2,zeta,kind,1);b2=sf(z2,xx,R/2,zeta,kind,-1)
    sp=(a1*b2+b1*a2)/np.sqrt(2*(1+S*S))
    sm=(a1*b2-b1*a2)/np.sqrt(2*(1-S*S))
    lim=max(np.max(abs(sp)),np.max(abs(sm)))
    for j,(arr,label) in enumerate([(sp,'Singlet (symmetric space)'),(sm,'Triplet (antisymmetric space)')]):
        ax=axs[i,j]
        ax.contourf(z1,z2,arr,levels=np.linspace(-lim,lim,31),norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),cmap='RdBu_r',extend='both')
        ax.plot(z,z,ls='--',lw=1,color='k',alpha=.6)
        ax.plot([-R/2,+R/2],[+R/2,-R/2],'k.',ms=5)
        ax.set(xlabel=r'Electron 1 bond-axis position $z_1/a_0$',ylabel=r'Electron 2 bond-axis position $z_2/a_0$',title=f'{title}\n{label}')
        ax.set_aspect('equal',adjustable='box')
        if kind=='px': ax.text(.02,.02,r'Slice: $x_1=x_2=+1.8a_0$',transform=ax.transAxes,fontsize=8,ha='left',va='bottom',bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
        else:ax.text(.02,.02,r'Slice: $x_1=x_2=0$',transform=ax.transAxes,fontsize=8,ha='left',va='bottom',bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
fig.suptitle('Spin-coupled VB pairs: exchange symmetry in two-electron position space',fontsize=12,y=0.995)
fig.tight_layout(pad=1.0,rect=(0,0,1,.968))
fig.savefig(os.path.join(OUT,'15_vb_sigma_pi_pair_amplitudes.png'),bbox_inches='tight');plt.close(fig)

# 16: heteronuclear effective VB covalent/ionic configuration mixing.
def ground(delta,tt=.18,U=1):
    # relative to covalent E_C=0; AA has gap U+delta, BB U-delta;
    # physical: delta=eps_A-eps_B, so delta>0 favours BB ionic.
    M=np.array([[0.,-np.sqrt(2)*tt,-np.sqrt(2)*tt],[-np.sqrt(2)*tt,U+delta,0.],[-np.sqrt(2)*tt,0.,U-delta]])
    ew,ev=eigh(M);return ew[0],ev[:,0]**2
Ds=np.linspace(0,.85,180)
eng=[];ww=[]
for d in Ds:
    e,w=ground(d);eng.append(e);ww.append(w)
ww=np.array(ww)
fig,axs=plt.subplots(1,2,figsize=(10.3,3.9))
axs[0].plot(Ds,ww[:,0],color=blue,label='Covalent $|C;S=0\\rangle$')
axs[0].plot(Ds,ww[:,1],color=green,label='Ionic $|AA\\rangle$')
axs[0].plot(Ds,ww[:,2],color=orange,label='Ionic $|BB\\rangle$')
axs[0].set(xlabel=r'Atomic site-energy mismatch $\delta/(U-V)$',ylabel='Configuration probability',ylim=(0,1),title='VB singlet: covalent and ionic mixing')
axs[0].legend(fontsize=8,loc='center left')
axs[1].plot(Ds,-np.array(eng),color=blue,lw=2.2,label='Singlet stabilization relative to triplet')
axs[1].plot(Ds,4*.18**2/(1-Ds**2),color=orange,ls='--',label=r'Second order: $2t^2(1/\Delta_A+1/\Delta_B)$')
axs[1].set(xlabel=r'Atomic site-energy mismatch $\delta/(U-V)$',ylabel=r'Energy / $(U-V)$',title='Perturbation theory fails near ionic resonance',ylim=(0,.48))
axs[1].legend(fontsize=8,loc='upper left')
fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT,'16_vb_polarization_ionic_mixing.png'),bbox_inches='tight');plt.close(fig)

# 17: explicit orbital symmetry labels for C-C bond pairs in full-valence EHT;
# important: only a reference for VB orbital selection, NOT a correlated VB calculation.
r=solve('ethylene');l=localize(r);q=l['q'];Cl=l['C_loc']
cc=np.argsort(-(q[0]*q[1]))[:2]
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for idx,j in enumerate(cc):
    c=Cl[:,j];d=np.array([c[k] for k in [3,7]])
    # true pz norm in nonorthogonal subblock
    pzwt=float(d@r['S'][np.ix_([3,7],[3,7])]@d)
    title='C--C $\\pi$' if pzwt>.85 else 'C--C $\\sigma$'
    # AO component representation as coefficients, not populations: AO basis nonorthogonal
    atom_names=['C1: 2s','C1: 2px','C1: 2py','C1: 2pz','C2: 2s','C2: 2px','C2: 2py','C2: 2pz']
    axs[idx].barh(np.arange(8),c[:8],color=[blue if k<4 else orange for k in range(8)])
    axs[idx].set(yticks=np.arange(8),yticklabels=atom_names,ylim=(-.7,7.7),xlabel='Signed AO expansion coefficient',title=f'{title} localized occupied orbital\ntransverse $p_z$ block norm = {pzwt:.3f}')
    axs[idx].invert_yaxis();axs[idx].axvline(0,color='k',lw=.8)
fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT,'17_ethene_sigma_pi_localized_orbitals.png'),bbox_inches='tight');plt.close(fig)
print('C-C ethylene occupied localized indices',cc,'pzweights',[float(Cl[[3,7],j]@r['S'][np.ix_([3,7],[3,7])]@Cl[[3,7],j]) for j in cc])
