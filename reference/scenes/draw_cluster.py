import numpy as np, pickle, datetime as dt, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York")
S='C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Claude-g3-market-research/2ce66efc-6ae1-4e6c-9607-28949911ca17/scratchpad/'
R='data/market/NQ/'
hi=np.load(R+'high.npy',mmap_mode='r');lo=np.load(R+'low.npy',mmap_mode='r')
op=np.load(R+'open.npy',mmap_mode='r');cl=np.load(R+'close.npy',mmap_mode='r')
ts=np.load(R+'close_ts_utc_ns.npy',mmap_mode='r')
rows=pickle.load(open(S+'rows.pkl','rb')); good=pickle.load(open(S+'good.pkl','rb'))
PINE=set(range(1,60))|{60,120,240,420,1440}
def uz(v,tol=2.0):
    keep=[]
    for tf,zt,zb,sd in sorted(v,key=lambda z:-z[0]):
        if not any(abs(zt-a)<=tol and abs(zb-b)<=tol for a,b,_ in keep): keep.append((zt,zb,tf))
    return keep

target='2023-10-31 10:11'
sel=[g for g in good if dt.datetime.fromtimestamp(int(ts[g[0]])/1e9,ET).strftime('%Y-%m-%d %H:%M')==target][0]
p=sel[0]; before,after=30,75
zones=uz([x for x in rows[p] if x[0] in PINE])
side=sel[2]; near=sel[5]
a,b=p-before,p+after; x=np.arange(a,b)
o=np.array(op[a:b]);h=np.array(hi[a:b]);l=np.array(lo[a:b]);c=np.array(cl[a:b])
fig,ax=plt.subplots(figsize=(15,8))
for i in range(len(x)):
    col='#26a69a' if c[i]>=o[i] else '#ef5350'
    ax.plot([x[i],x[i]],[l[i],h[i]],color=col,lw=0.9,zorder=3)
    ax.add_patch(Rectangle((x[i]-0.34,min(o[i],c[i])),0.68,max(abs(c[i]-o[i]),0.03),
                 facecolor=col,edgecolor=col,lw=0.4,zorder=4))
for zt,zb,tf in zones:
    ax.add_patch(Rectangle((p,zb),b-p-1,max(zt-zb,0.3),facecolor='#2962FF',alpha=0.16,
                 edgecolor='#2962FF',lw=0.9,zorder=2))
    ax.text(p+1.5,zt,f" {tf}m",fontsize=8.5,color='#0b3d91',va='bottom',ha='left',zorder=6)
ax.axhline(near,color='#0b3d91',lw=1.8,zorder=5)
ax.axvline(p,color='#111',lw=1.5,ls='--',zorder=5)
top=float(h.max()); ax.text(p+0.6,top,f"T0 {target[-5:]} ET — {len(zones)} zony",
        fontsize=11,va='top',ha='left',fontweight='bold',zorder=7)
ax.annotate('',xy=(p+30,float(hi[p+1:p+31].max())),xytext=(p+30,near),
            arrowprops=dict(arrowstyle='<->',color='#444',lw=1.4),zorder=6)
ax.text(p+31,(near+float(hi[p+1:p+31].max()))/2,f"ushla {sel[3]:.1f} p",fontsize=9.5,color='#333',va='center')
ax.plot([p+sel[4]],[near],marker='v',color='#111',ms=11,zorder=8)
ax.text(p+sel[4]+1,near-1.5,f"vozvrat k blizhney granice: +{sel[4]} min",fontsize=9.5,va='top')
tk=np.arange(a,b,15); ax.set_xticks(tk)
ax.set_xticklabels([dt.datetime.fromtimestamp(int(ts[i])/1e9,ET).strftime('%H:%M') for i in tk],fontsize=8.5)
ax.set_title(f"NQ 2023-10-31 — odna minuta zazhigaet {len(zones)} zon na TF {sorted(z[2] for z in zones)}; "
             f"sinyaya liniya — blizhnyaya granica klastera",fontsize=11.5,loc='left')
ax.set_ylabel('NQ, punkty'); ax.grid(alpha=0.18,lw=0.5); ax.set_xlim(a-1,b)
fig.tight_layout(); fig.savefig(S+'scene_pine.png',dpi=115); plt.close(fig)
print("ok", target, "zones:",[(round(z[0],2),round(z[1],2),z[2]) for z in zones], "near",near)
