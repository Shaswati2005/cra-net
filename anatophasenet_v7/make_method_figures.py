"""Figures of methodology_v7: fig_init_cycles.png (detector) and fig_geometry.png (Table of posture signals)."""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import anato_phasenet_v7 as M

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREY = '#2F6DB5', '#C8691C', '#868E96'

# ---------------------------------------------------------------- synthetic limping walk, real detector
T, N = 66, 200
n = np.arange(N)
r = 0.40 + 0.11 * np.cos(2 * np.pi * n / T - 0.6) ** 2 * (1 + 0.55 * np.cos(2 * np.pi * (n / T) / 2 * 2 - 0.6))
r = r + 0.006 * np.sin(2 * np.pi * n / 23)
h = np.full(N, 300.0)
cycles = M.detect_init_cycles(r * h, h)

ra = M._moving_average(r, 5)
w = ra - M._moving_average(ra, 61)
w = w / (w.std() + 1e-8)
lags = np.arange(40, min(110, N // 2) + 1)
ac = np.array([np.dot(w[:-l], w[l:]) / (N - l) for l in lags])
pk = [i for i in range(1, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
top = max(ac[i] for i in pk)
period = lags[next(i for i in pk if ac[i] >= 0.8 * top)]
lm = [t for t in range(1, N - 1) if w[t] >= w[t - 1] and w[t] >= w[t + 1]]
ev = []
for t in sorted(lm, key=lambda t: -w[t]):
    if all(abs(t - u) >= 0.35 * period for u in ev):
        ev.append(t)
ev.sort()
print('period', period, 'events', ev, 'kept', cycles, 'candidates', max(0, len(ev) - 2))

fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 3.3), dpi=300, gridspec_kw=dict(hspace=0.62))
a.plot(n, w, color=BLUE, lw=0.9)
for s, e in cycles:
    a.axvspan(s, e, color=BLUE, alpha=0.10, lw=0)
starts = {s for s, _ in cycles}
for t in ev:
    a.plot(t, w[t], 'o', ms=4, mfc=BLUE if w[t] > 1 else 'white', mec=BLUE, mew=0.8)
s0, e0 = cycles[0]
for k in range(9):
    f = int(round(s0 + k * (e0 - s0) / 8))
    a.plot(f, -2.0, 's', ms=2.4, color=ORANGE)
a.set_xlabel('frame $n$', labelpad=1)
a.set_ylabel(r'$\hat{r}_n$', labelpad=1)
a.set_title('(a) Normalised aspect ratio, step events and kept cycles', fontsize=8.5, pad=3)
b.plot(lags, ac, color=BLUE, lw=0.9)
b.axhline(0.8 * top, color=GREY, lw=0.7, ls='--')
b.text(74, 0.8 * top + 0.05, r'$0.8\times$ highest maximum', fontsize=7.5, color='#52606D')
for i in pk:
    b.plot(lags[i], ac[i], 'o', ms=3.6, mfc='white', mec=BLUE, mew=0.8)
b.axvline(period, color=ORANGE, lw=0.8)
b.text(period + 1.5, b.get_ylim()[0] + 0.1, rf'$\hat{{T}}={period}$', fontsize=8, color=ORANGE)
b.set_xlabel(r'lag $\ell$ (frames)', labelpad=1)
b.set_ylabel(r'$R(\ell)$', labelpad=1)
b.set_title('(b) Autocorrelation in the search window', fontsize=8.5, pad=3)
fig.savefig(os.path.join(HERE, 'fig_init_cycles.png'), bbox_inches='tight')

# ---------------------------------------------------------------- one real INIT frame
d = os.path.join(M.PARENT, 'Gait_Dataset_silhoutte_based', 'INIT', 'INIT_GaitDB', 'INIT_8', 'silhouettes', 'nm')
seq = os.path.join(d, sorted(os.listdir(d))[0])
files = M.list_frames(seq)
st = [M.frame_stats(os.path.join(seq, f)) for f in files]
ok = [(i, s) for i, s in enumerate(st) if s is not None]
h_ref = float(np.median([s['h'] for _, s in ok]))
y_ref = float(np.median([s['y0'] for _, s in ok]))
idx = max((i for i, s in ok if 0.3 * len(files) < i < 0.7 * len(files)), key=lambda i: st[i]['w'])
s = st[idx]
sign = -1.0  # every INIT walk in this dataset moves right to left
feat = M.geometry_features(s, h_ref, y_ref, sign)
print(os.path.basename(seq), 'frame', idx, 'of', len(files), dict(zip(M.GEOM_NAMES, feat.round(3).tolist())), 'h', s['h'], 'w', s['w'], 'h_ref', h_ref)

a_img = np.array(Image.open(os.path.join(seq, files[idx])).convert('L')) > 127
x0 = s['cx_box'] - s['w'] / 2
y0, hh, ww = s['y0'], s['h'], s['w']
pad = 0.55 * ww
fig, ax = plt.subplots(figsize=(3.45, 3.05), dpi=300)
ax.imshow(np.where(a_img, 0.12, 1.0), cmap='gray', vmin=0, vmax=1, interpolation='nearest', zorder=1)
ax.add_patch(Rectangle((x0, y0), ww, 0.30 * hh, fc=ORANGE, alpha=0.20, lw=0, zorder=2))
ax.add_patch(Rectangle((x0, y0 + 0.70 * hh), ww, 0.30 * hh, fc=BLUE, alpha=0.20, lw=0, zorder=2))
ax.add_patch(Rectangle((x0, y0), ww, hh, fc='none', ec='#52606D', lw=0.7, ls='--', zorder=3))
for f in (0.15, 0.55):
    ax.plot([x0, x0 + ww], [y0 + f * hh] * 2, color='#52606D', lw=0.6, ls=':', zorder=3)
ytop, ybot = y0 + 0.15 * hh, y0 + 0.85 * hh
ax.plot([s['cx_top'], s['cx_bot']], [ytop, ybot], color='#C92A2A', lw=1.0, zorder=4)
ax.plot(s['cx'], s['cy'], 'o', ms=4, mfc='white', mec='#C92A2A', mew=0.9, zorder=5)
tx = x0 + ww + 0.1 * ww
for yy, lab in [(y0 + 0.15 * hh, r'top 30 %: $c_{x,[0,0.3]}$'), (y0 + 0.35 * hh, r'15–55 %: $w_{[0.15,0.55]}$'),
                (s['cy'] + 0.06 * hh, r'centroid $(c_x,c_y)$'), (y0 + 0.85 * hh, r'bottom 30 %: $c_{x,[0.7,1]}$, $w_{[0.7,1]}$')]:
    ax.text(tx, yy, lab, fontsize=7.6, va='center')
ax.annotate('', (x0 - 0.1 * ww, y0), (x0 - 0.1 * ww, y0 + hh), arrowprops=dict(arrowstyle='<->', lw=0.6, color='#52606D'))
ax.text(x0 - 0.18 * ww, y0 + hh / 2, '$h$', fontsize=9, ha='right', va='center')
ax.annotate('', (x0, y0 + hh + 0.035 * hh), (x0 + ww, y0 + hh + 0.035 * hh), arrowprops=dict(arrowstyle='<->', lw=0.6, color='#52606D'))
ax.text(x0 + ww / 2, y0 + hh + 0.075 * hh, '$w$', fontsize=9, ha='center', va='top')
ax.set_xlim(x0 - pad * 0.6, x0 + ww + 2.1 * ww)
ax.set_ylim(y0 + hh + 0.15 * hh, y0 - 0.05 * hh)
ax.axis('off')
fig.savefig(os.path.join(HERE, 'fig_geometry.png'), bbox_inches='tight', pad_inches=0.02)
print('saved')
