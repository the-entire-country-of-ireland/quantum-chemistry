"""Extra figures for physically audited semiempirical bond notes.
Generates figures 09-14, JSON results; requires only NumPy/SciPy/Matplotlib/RDKit (optional check).
"""
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from screen_models import sto_pair,e_ht
from full_eht import solve,PARAM,BOHR
from localize_eht import localize
from scipy.linalg import eigh
F='figures';os.makedirs(F,exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':170,'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14})
colors=['#2368a0','#cf7045','#37967a','#9465ae']

def save(name,fig):
    fig.tight_layout(pad=1.3)
    fig.savefig(os.path.join(F,name+'.png'),bbox_inches='tight')
    plt.close(fig)

def orb_sto(z,x,zeta,kind,center,orient=1):
    zz=z-center; r=np.sqrt(zz*zz+x*x)
    A=np.sqrt(zeta**5/(3*np.pi))*np.exp(-zeta*r)
    if kind=='sp3':return A/2*(r+3*orient*zz)
    if kind=='sp2':return A/np.sqrt(3)*(r+np.sqrt(6)*orient*zz)
    if kind=='px':return A*np.sqrt(3)*x
    if kind=='pz':return A*np.sqrt(3)*zz
    if kind=='s1':return np.sqrt(zeta**3/np.pi)*np.exp(-zeta*r)

# compare nodeless STO vs hydrogenic n2 for equal Z=1
Rvec=np.linspace(2.5,10,90)
Ssig=np.array([sto_pair(R,.5,.5,'sp3','sp3')[0] for R in Rvec])
Spi=np.array([sto_pair(R,.5,.5,'px','px')[0] for R in Rvec])
Rcarb=np.linspace(2.25,4,65)
Ssp2=np.array([sto_pair(R,1.625,1.625,'sp2','sp2')[0] for R in Rcarb])
Spcarb=np.array([sto_pair(R,1.625,1.625,'px','px')[0] for R in Rcarb])
fig,axs=plt.subplots(1,2,figsize=(10.8,3.7))
axs[0].plot(Rvec,Ssig,c=colors[0],label=r'$sp^3-sp^3$ head-on $\sigma$')
axs[0].plot(Rvec,Spi,c=colors[1],label=r'$p_x-p_x$ side-on $\pi$')
axs[0].axvline(6,ls=':',c='gray')
axs[0].set(xlabel=r'Separation $R$ ($a_0$)',ylabel='Signed overlap $S$',title=r'Fictitious equal $Z_{\mathrm{eff}}=1$, $\zeta=0.5$')
axs[0].legend(fontsize=8)
axs[1].plot(Rcarb,Ssp2,c=colors[0],label=r'$sp^2-sp^2$ ethene $\sigma$')
axs[1].plot(Rcarb,Spcarb,c=colors[1],label=r'$2p_x-2p_x$ ethene $\pi$')
axs[1].axvline(1.339*BOHR,ls=':',c='gray',label='Ethene C=C geometry')
axs[1].set(xlabel=r'Separation $R$ ($a_0$)',ylabel='Signed overlap $S$',title=r'Carbon-like radial scale, $\zeta_C=1.625$')
axs[1].legend(fontsize=7.6)
save('09_revised_sigma_pi_overlap',fig)

z=np.linspace(-4.8,4.8,330);xx=np.linspace(-3.6,3.6,240);Z,X=np.meshgrid(z,xx)
R=1.339*BOHR
cases=[('Ethene $sp^2$ sigma orbital on left',orb_sto(Z,X,1.625,'sp2',-R/2,1)),('Ethene $sp^2$ sigma orbital on right',orb_sto(Z,X,1.625,'sp2',+R/2,-1)),('Transverse $p_x$ orbital on left',orb_sto(Z,X,1.625,'px',-R/2)),('Transverse $p_x$ orbital on right',orb_sto(Z,X,1.625,'px',+R/2))]
mmax=max(abs(ar).max() for name,ar in cases)
fig,axs=plt.subplots(2,2,figsize=(10.2,6.6),sharex=True,sharey=True)
for ax,(name,arr) in zip(axs.flat,cases):
    ax.contourf(Z,X,arr,levels=np.linspace(-mmax,mmax,30),norm=TwoSlopeNorm(vmin=-mmax,vcenter=0,vmax=mmax),cmap='RdBu_r',extend='both')
    ax.plot([-R/2,R/2],[0,0],'ko',markersize=3)
    ax.axhline(0,color='k',lw=.5,alpha=.5)
    ax.set(xlabel=r'Bond axis $z/a_0$',ylabel=r'Transverse $x/a_0$',title=name)
save('10_revised_sigma_pi_shapes',fig)

names=['methane','ethane','ethylene','methanol']
rs={k:solve(k) for k in names};ls={k:localize(rs[k]) for k in names}
fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,k in zip(axs,['methane','ethylene','methanol']):
    result=rs[k];loc=ls[k];q=loc['q']; n=loc['q'].shape[1]
    # reorder methane methane all equivalent; ethylene CH then CC; methanol CH, CO, OH, LP
    labels=[];vals=[]
    for j in range(n):
        top=np.argsort(-q[:,j]); ii=top[0];jj=top[1]
        if q[jj,j]<.08: lab='LP'+str(ii)
        else: lab=result['geo'][ii][0]+str(ii)+'-'+result['geo'][jj][0]+str(jj)
        labels.append(lab)
        vals.append((q[ii,j],q[jj,j]))
    # distinguish double C-C by counting
    x=np.arange(n)
    # show atom-specific colored composition instead of arbitrary top2
    bottom=np.zeros(n)
    for kind,color in [('C',colors[0]),('O',colors[1]),('H',colors[2])]:
        vals1=np.array([np.sum(q[[i for i,(sym,p) in enumerate(result['geo']) if sym==kind],j]) for j in range(n)])
        ax.bar(x,vals1,bottom=bottom,color=color,label=kind,width=.7)
        bottom+=vals1
    ax.set_xticks(x,labels,rotation=58,ha='right',fontsize=7)
    ax.set(ylim=(0,1.05),ylabel='Löwdin atomic population per occupied localized MO',title=k.capitalize())
    ax.legend(fontsize=7,ncol=3,loc='upper right')
save('11_revised_localized_bonds',fig)

# Independent NBO reference only for local bond population; NBO differs by population definition.
labels=['C-C ethane','C=C ethene','C-O methanol']
values=[rs['ethane']['BO'][(0,1)],rs['ethylene']['BO'][(0,1)],rs['methanol']['BO'][(0,1)]]
expected=[1,2,1]
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
x=np.arange(3);w=.34
axs[0].bar(x-w/2,values,width=w,color=colors[0],label='Calculated Löwdin/Wiberg index')
axs[0].bar(x+w/2,expected,width=w,color='gray',alpha=.5,label='Lewis bond order (not observed index)')
axs[0].set(xticks=x,xticklabels=labels,ylabel='Bond index',ylim=(0,2.35),title='Full-valence bond-order pattern')
axs[0].legend(fontsize=7)
ks=np.array([1.55,1.65,1.75,1.85,1.95,2.1])
cc=[];ce=[];co=[]
for kval in ks:
    cc.append(solve('ethane',kval)['BO'][(0,1)])
    ce.append(solve('ethylene',kval)['BO'][(0,1)])
    co.append(solve('methanol',kval)['BO'][(0,1)])
axs[1].plot(ks,cc,'-o',c=colors[0],label='Ethane C-C')
axs[1].plot(ks,ce,'-o',c=colors[1],label='Ethylene C=C')
axs[1].plot(ks,co,'-o',c=colors[2],label='Methanol C-O')
axs[1].axvline(1.75,ls=':',color='gray',label='Standard K')
axs[1].set(xlabel=r'Extended Hückel $K$',ylabel='Bond index',title='Sensitivity to off-diagonal scaling',ylim=(.6,2.25))
axs[1].legend(fontsize=7)
save('12_revised_bondorder_robustness',fig)

# compare native 2-site and EHT localized bond with independent NBO for CO/CH
co_q=ls['methanol']['q'];co_idx=int(np.argmax(co_q[0,:]*co_q[1,:]))
ch_q=ls['methanol']['q'];ch_idx=int(np.argmax(ch_q[0,:]*ch_q[2,:]))
co_O=co_q[1,co_idx]; ch_C=ch_q[0,ch_idx]
modelvals=[.9880,.8573,.6542,.5080,co_O,ch_C,.82**2,.77**2]
fig,axs=plt.subplots(1,2,figsize=(10.7,3.75))
co=[.9880,.6542,co_O,.82**2]; ch=[1-.8573,.5080,ch_C,.77**2]
styles=['Original screened Coulomb','Two-site extended Hückel','Full-valence localized','Published NBO reference']
for ax,v,title,target in [(axs[0],co,'C-O bond: O contribution',.82**2),(axs[1],ch,'C-H bond: C contribution',.77**2)]:
    ys=np.arange(4)
    ax.barh(ys,v,color=['#b35d61',colors[3],colors[0],'gray'],height=.65)
    ax.set_yticks(ys,styles,fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0,1.07),xlabel='Orbital population fraction',title=title)
    for i,b in enumerate(v):ax.text(b+.01,i,f'{100*b:.1f}%',va='center',fontsize=8)
save('13_revised_polarity_audit',fig)

# Parameter sensitivity in localized CO and CH at reasonable K
fig,axs=plt.subplots(1,2,figsize=(10.5,3.5))
Ktest=np.array([1.55,1.65,1.75,1.85,1.95,2.1]); co=[],
valsO=[];valsC=[]
for kk in Ktest:
    m=solve('methanol',kk);q=localize(m)['q'];j=np.argmax(q[0,:]*q[1,:]);valsO.append(q[1,j]);
    m=solve('methane',kk);q=localize(m)['q'];j=np.argmax(q[0,:]*q[1,:]);valsC.append(q[0,j]);
axs[0].plot(Ktest,valsO,'o-',c=colors[1],label='C-O: O orbital population')
axs[0].axhline(.82**2,c='gray',ls='--',label='Methanol NBO approx. 67%')
axs[0].axvline(1.75,color='gray',ls=':')
axs[0].set(ylim=(.45,.85),xlabel='Extended Hückel K',ylabel='Localized orbital population',title='C-O: moderate polarization, K-dependent')
axs[0].legend(fontsize=7)
axs[1].plot(Ktest,valsC,'o-',c=colors[0],label='C-H: C orbital population')
axs[1].axhline(.77**2,c='gray',ls='--',label='Methanol NBO approx. 59%')
axs[1].axvline(1.75,color='gray',ls=':')
axs[1].set(ylim=(.45,.68),xlabel='Extended Hückel K',ylabel='Localized orbital population',title='C-H: model underpolarizes')
axs[1].legend(fontsize=7)
save('14_revised_polarity_sensitivity',fig)

# numerical reports
report={}
for name in names:
    m=rs[name];q=ls[name]['q']
    report[name]={'energies_eV':[float(v) for v in m['ev']], 'charges':[float(v) for v in m['charges']], 'bond_indices':{f'{i}-{j}':float(v) for (i,j),v in m['BO'].items() if v>.05}, 'localized_populations':q.tolist(), 'min_S_eigen':float(np.linalg.eigvalsh(m['S']).min())}
report['two_center']={}
for name,za,zb,R,kinda,kindb,ea,eb in [
('Z1_sigma',.5,.5,6,'sp3','sp3',-3.40142,-3.40142),('Z1_pi',.5,.5,6,'px','px',-3.40142,-3.40142),
('ethane_sigma',1.625,1.625,1.543*BOHR,'sp3','sp3',-13.9,-13.9),('ethene_sigma',1.625,1.625,1.339*BOHR,'sp2','sp2',-14.733333,-14.733333),('ethene_pi',1.625,1.625,1.339*BOHR,'px','px',-11.4,-11.4),
('methanol_CO',1.625,2.275,1.427*BOHR,'sp3','sp3',-13.9,-19.1125),('methane_CH',1.625,1.3,1.087*BOHR,'sp3','s1',-13.9,-13.6)]:
    S,na,nb=sto_pair(R,za,zb,kinda,kindb);t,d,wb,hab=e_ht(S,ea,eb)
    report['two_center'][name]={'R_bohr':R,'S':S,'t_eV':t,'B_weight':wb,'normA':na,'normB':nb}
with open('physical_revision_results.json','w') as f:json.dump(report,f,indent=2)
print('SUMMARY')
for name,ob in report['two_center'].items():print(name, 'S',round(ob['S'],4),'t',round(ob['t_eV'],3),'wB',round(ob['B_weight'],4))
for name in names:print(name,'charges',np.round(rs[name]['charges'],3),'CCBO',rs[name]['BO'].get((0,1)))
print('Methanol localized CO',co_idx,co_q[0,co_idx],co_q[1,co_idx],'CH',ch_C)
