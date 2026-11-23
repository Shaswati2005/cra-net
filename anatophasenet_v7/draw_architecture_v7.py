"""Draws the AnatoPhaseNet v7 architecture: architecture_v7.png and architecture_v7.pdf."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})

INK, MUTED, LINE = '#1F2933', '#52606D', '#7B8794'
GREY = ('#F1F3F5', '#868E96')
BLUE = ('#E3EEF9', '#2F6DB5')
ORANGE = ('#FCE9D9', '#C8691C')
GREEN = ('#E2F2E4', '#3B8A47')
VIOLET = ('#ECE7F6', '#6B4FA8')

fig = plt.figure(figsize=(17, 11.4), dpi=200, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 17)
ax.set_ylim(0, 11.4)
ax.axis('off')


def box(x, y, w, h, title, lines=(), pal=GREY, ts=10.5, ls=8.8, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.07', fc=pal[0], ec=pal[1],
                                lw=1.3, ls='--' if dashed else '-', zorder=2))
    n = len(lines)
    step = 0.215
    top = y + h / 2 + (n * step) / 2
    ax.text(x + w / 2, top, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=INK, zorder=3)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - (i + 1) * step, s, ha='center', va='center', fontsize=ls, color=INK, zorder=3)


def arrow(p, q, color=LINE, lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, lw=lw, color=color, zorder=1,
                                 shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


def path(pts, color=LINE, lw=1.5):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle='round')
    arrow(pts[-2], pts[-1], color, lw)


def lane(x, y, w, h, label, pal, lx=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1', fc='none', ec=pal[1], lw=0.9,
                                ls=(0, (4, 3)), zorder=0))
    ax.text(x + lx, y + h - 0.17, label, ha='left', va='center', fontsize=9.6, color=pal[1], fontweight='bold')


ax.text(0.3, 11.03, 'AnatoPhaseNet v7', fontsize=19, fontweight='bold', color=INK, va='center')
ax.text(3.62, 11.0, 'side-view silhouette walk  →  gait class per cycle and per walk', fontsize=11.5, color=MUTED, va='center')

# ------------------------------------------------------------------ preprocessing
Y = 9.5
box(0.3, Y, 2.3, 1.05, 'Input walk', ['silhouette frames $I_1,\\dots,I_N$', 'of one subject'])
box(3.0, Y, 5.6, 1.05, 'B1  Direction and gait cycles',
    ['mirror to one walking direction', 'GAIT-IT: 9 key-frame entries per cycle, read in blocks of 9',
     'INIT: cycles detected from the bounding-box aspect ratio'])
box(9.0, Y, 5.0, 1.05, 'B2  Phase sampling',
    ['nine phase frames per cycle, $t=0,\\dots,8$', 'jitter offsets $j\\in\\{-2,0,+2\\}$ frames',
     'train: drawn per phase; test: 3 aligned views'])
arrow((2.6, Y + .52), (3.0, Y + .52))
arrow((8.6, Y + .52), (9.0, Y + .52))

# ------------------------------------------------------------------ appearance stream
lane(0.3, 5.75, 10.95, 3.4, 'Appearance stream (cropped silhouettes)', BLUE, lx=2.0)
Ya, Ha = 6.0, 2.62
xs = [0.5, 2.3, 3.72, 5.14, 6.56, 8.0, 9.4]
box(xs[0], Ya, 1.55, Ha, 'Image input', ['crop to 96×96', '$2x_t-1$', '$x_t-x_{t-1}$', '$x_t-\\bar{x}$', '', '3×9×96²'], BLUE)
box(xs[1], Ya, 1.2, Ha, 'B3 stem', ['1×7×7 conv', 'max pool', 'ImageNet', '', '64×9×24²'], BLUE)
box(xs[2], Ya, 1.2, Ha, 'B3 layer 1', ['2 × (2+1)D', 'blocks', '', '', '64×9×24²'], BLUE)
box(xs[3], Ya, 1.2, Ha, 'B3 layer 2', ['2 × (2+1)D', 'blocks', 'stride 2', '', '128×9×12²'], BLUE)
box(xs[4], Ya, 1.2, Ha, 'B3 layer 3', ['2 × (2+1)D', 'blocks', 'stride 2', '', '256×9×6²'], BLUE)
box(xs[5], Ya, 1.2, Ha, 'B4 parts', ['6 strips', '+ global', 'avg ∥ max', 'own $\\Theta_p$', '', '7×9×128'], BLUE)
box(xs[6], Ya, 1.65, Ha, 'B5 temporal', ['phase embedding', 'circular conv', 'att ∥ max ∥ sym', 'BN', '', '$g^{app}$: 7×384'], BLUE)
for a, b, wa in zip(xs[:-1], xs[1:], [1.55, 1.2, 1.2, 1.2, 1.2, 1.2]):
    arrow((a + wa, Ya + Ha / 2), (b, Ya + Ha / 2))

# ------------------------------------------------------------------ geometry stream
lane(0.3, 3.5, 10.95, 2.05, 'Geometry stream (raw frames, measured before cropping)', ORANGE)
Yg, Hg = 3.7, 1.42
box(0.5, Yg, 3.1, Hg, 'Posture signals', ['per phase, on the raw frame', '9 signals on INIT, 7 on GAIT-IT', '$G$: 9 phases × $S$ signals'], ORANGE)
box(4.0, Yg, 3.2, Hg, 'B6 temporal encoder', ['BN per signal', 'two circular convs, 64 channels', 'second is residual'], ORANGE)
box(7.6, Yg, 3.45, Hg, 'B6 pooling', ['mean ∥ max ∥ sym over phases', 'BN', '$g^{geo}\\in\\mathbb{R}^{192}$'], ORANGE)
arrow((3.6, Yg + Hg / 2), (4.0, Yg + Hg / 2))
arrow((7.2, Yg + Hg / 2), (7.6, Yg + Hg / 2))

# B2 feeds both streams
bx = 11.5
path([(14.0, Y + .52), (14.25, Y + .52), (14.25, 9.28), (1.28, 9.28), (1.28, Ya + Ha)])
path([(0.14, 9.28), (0.14, Yg + Hg / 2), (0.5, Yg + Hg / 2)])
ax.plot([0.14, 1.28], [9.28, 9.28], color=LINE, lw=1.5, zorder=1)

# ------------------------------------------------------------------ descriptor
box(bx, 3.7, 1.15, 4.92, 'Cycle', ['descriptor', '', '$g=[g^{app}\\,\\|\\,g^{geo}]$', '', '$\\mathbb{R}^{2880}$', '', '2688 + 192'], VIOLET, ts=10.5, ls=8.5)
arrow((11.05, Ya + Ha / 2), (bx, Ya + Ha / 2))
arrow((11.05, Yg + Hg / 2), (bx, Yg + Hg / 2))

# ------------------------------------------------------------------ heads
hx, hw = 13.05, 2.1
box(hx, 7.0, hw, 1.62, 'B7 classifiers', ['7 part classifiers', '+ 1 geometry classifier', 'dropout 0.3, logits', 'averaged'], GREEN)
box(hx, 5.3, hw, 1.42, 'Severity head', ['linear, 3 levels', 'auxiliary, training only', 'not used for the class'], GREY, dashed=True)
box(hx, 3.7, hw, 1.32, 'B8 walk head', ['$e_k\\in\\mathbb{R}^{256}$ per cycle', '$[\\mu\\,\\|\\,\\sigma\\,\\|\\,\\delta]$ → LN → MLP'], GREEN)
for yy in (7.81, 6.01, 4.36):
    arrow((bx + 1.15, yy), (hx, yy))
box(15.5, 7.33, 1.25, 0.96, 'Cycle class', ['$\\hat{y}$'], GREEN)
box(15.5, 3.88, 1.25, 0.96, 'Walk class', ['$C$ scores'], GREEN)
arrow((hx + hw, 7.81), (15.5, 7.81))
arrow((hx + hw, 4.36), (15.5, 4.36))

# ------------------------------------------------------------------ detail panels
ax.text(0.3, 3.15, 'Inside the blocks', fontsize=11.5, fontweight='bold', color=INK, va='center')
Yd, Hd = 0.3, 2.6
# (2+1)D block
ax.add_patch(FancyBboxPatch((0.3, Yd), 7.45, Hd, boxstyle='round,pad=0,rounding_size=0.08', fc='#F7FAFD', ec=BLUE[1], lw=0.9, zorder=0))
ax.text(0.45, Yd + Hd - 0.2, '(2+1)D residual block (B3)', fontsize=9.8, fontweight='bold', color=BLUE[1], va='center')
yb, hb = Yd + 1.05, 0.72
cells = [(0.95, 1.4, ['1×3×3 conv', 'BN, ReLU']), (2.8, 1.65, ['3×1×1 circular', 'conv, BN ($\\gamma_0=0$)']), (5.0, 1.4, ['1×3×3 conv', 'BN'])]
for x, w, t in cells:
    box(x, yb, w, hb, t[0], [t[1]], BLUE, ts=8.6, ls=8.2)
ax.text(0.55, yb + hb / 2, '$x$', fontsize=11, ha='center', va='center', color=INK)
ax.text(7.35, yb + hb / 2, '$y$', fontsize=11, ha='center', va='center', color=INK)
arrow((0.68, yb + hb / 2), (0.95, yb + hb / 2))
arrow((2.35, yb + hb / 2), (2.8, yb + hb / 2))
arrow((4.45, yb + hb / 2), (5.0, yb + hb / 2))
arrow((6.4, yb + hb / 2), (6.78, yb + hb / 2))
ax.text(6.9, yb + hb / 2, '⊕', fontsize=15, ha='center', va='center', color=INK, zorder=3)
arrow((7.03, yb + hb / 2), (7.25, yb + hb / 2))
# temporal skip
path([(2.57, yb + hb / 2), (2.57, yb + hb + 0.2), (4.72, yb + hb + 0.2), (4.72, yb + hb / 2 + 0.13)], color=BLUE[1], lw=1.1)
ax.text(4.72, yb + hb / 2, '⊕', fontsize=13, ha='center', va='center', color=INK, zorder=3,
        bbox=dict(boxstyle='circle,pad=0.05', fc='#F7FAFD', ec='none'))
ax.text(3.65, yb + hb + 0.33, 'residual around the temporal conv, then ReLU', fontsize=8, color=MUTED, ha='center', va='center')
# shortcut
path([(0.68, yb + hb / 2), (0.68, yb - 0.32), (6.9, yb - 0.32), (6.9, yb + hb / 2 - 0.14)], color=BLUE[1], lw=1.1)
ax.text(3.8, yb - 0.46, 'shortcut $S(x)$: identity or pretrained 1×1×1 projection; ReLU after the sum', fontsize=8, color=MUTED, ha='center', va='center')
ax.text(4.0, Yd + 0.17, 'phase axis wraps around: phase 0 is next to phase 8; never strided', fontsize=8, color=MUTED, ha='center', va='center')

# part head
ax.add_patch(FancyBboxPatch((7.95, Yd), 4.3, Hd, boxstyle='round,pad=0,rounding_size=0.08', fc='#F7FAFD', ec=BLUE[1], lw=0.9, zorder=0))
ax.text(8.1, Yd + Hd - 0.2, 'Three pooled summaries (B5, B6)', fontsize=9.8, fontweight='bold', color=BLUE[1], va='center')
for i, s in enumerate(['att: learned softmax over the nine phases',
                       'max: strongest response over the phases',
                       'sym: $\\frac{1}{4}\\sum_{t=0}^{3}|f_t-f_{t+4}|$',
                       'phase $t$ against phase $t+4$, about',
                       'half a cycle later; grows with a',
                       'one-sided impairment']):
    ax.text(8.15, Yd + Hd - 0.55 - i * 0.31, s, fontsize=8.9, color=INK, va='center')

# walk head
ax.add_patch(FancyBboxPatch((12.45, Yd), 4.3, Hd, boxstyle='round,pad=0,rounding_size=0.08', fc='#F6FBF6', ec=GREEN[1], lw=0.9, zorder=0))
ax.text(12.6, Yd + Hd - 0.2, 'Walk statistics (B8)', fontsize=9.8, fontweight='bold', color=GREEN[1], va='center')
for i, s in enumerate(['$\\mu$: the typical cycle of the walk',
                       '$\\sigma$: how much the cycles differ',
                       '$\\delta$: mean $|e_{k+1}-e_k|$, stride to stride',
                       'train: $K\\sim U\\{2,\\dots,6\\}$ consecutive cycles',
                       'test: all cycles of the walk, in order',
                       '768 → 128 → $C$ perceptron']):
    ax.text(12.65, Yd + Hd - 0.55 - i * 0.31, s, fontsize=8.9, color=INK, va='center')

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'architecture_v7.{ext}'), facecolor='white')
print('saved')
