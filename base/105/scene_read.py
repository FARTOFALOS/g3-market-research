import sys, numpy as np, pandas as pd
from tape import load_minutes
D = int(sys.argv[1]); prev = int(sys.argv[2])
df = load_minutes(str(prev)[:4]+'-'+str(prev)[4:6]+'-'+str(prev)[6:], str(D)[:4]+'-'+str(D)[4:6]+'-'+str(int(str(D)[6:])+1).zfill(2))
night = df[((df.date == prev) & (df['mod'] >= 1080)) | ((df.date == D) & (df['mod'] < 120))]
H, L = night.h.max(), night.l.min()
print(f'night 18:00-02:00: H {H} at {night.loc[night.h.idxmax(),"mod"]//60:02d}:{night.loc[night.h.idxmax(),"mod"]%60:02d}, L {L}, width {H-L:.2f}, bars {len(night)}')
day = df[(df.date == D) & (df['mod'] >= 120) & (df['mod'] < 960)].copy()
day['rng'] = day.h - day.l
for m0 in range(120, 960, 15):
    g = day[(day['mod'] >= m0) & (day['mod'] < m0 + 15)]
    if len(g) == 0: continue
    flag = ''
    if g.h.max() > H: flag += ' >H'
    if g.l.min() < L: flag += ' <L'
    print(f'{m0//60:02d}:{m0%60:02d}  O {g.o.iloc[0]:9.2f} H {g.h.max():9.2f} L {g.l.min():9.2f} C {g.c.iloc[-1]:9.2f}  medbar {g.rng.median():4.1f}{flag}')
