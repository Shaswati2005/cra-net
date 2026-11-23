"""
Renders methodology_v7.tex to methodology_v7.pdf without a LaTeX install.
Handles only the LaTeX used in that file: maths through matplotlib, layout through headless Chrome.
For the IEEEtran original, compile methodology_v7.tex with pdflatex (e.g. Overleaf).
"""
import base64
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser, math_to_image

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'mathtext.fontset': 'stix', 'svg.fonttype': 'path'})
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
SIZE_TEXT = 10
parser = MathTextParser('path')
ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']


def fix_math(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(r'\dots', r'\ldots').replace(r'\boldsymbol', r'\mathbf')
    s = s.replace(r'\lfloor', r'\lfloor ').replace(r'\rfloor', r'\rfloor ')
    s = re.sub(r"(''?)_\{([^{}]*)\}", lambda m: '_{' + m.group(2) + '}^{' + r'\prime' * len(m.group(1)) + '}', s)
    for a, b in (('le', 'leq'), ('ne', 'neq'), ('ge', 'geq'), ('to', 'rightarrow')):
        s = re.sub(r'\\' + a + r'(?![a-zA-Z])', lambda m: '\\' + b + ' ', s)
    return s


cache = {}


def math_svg(src, size=SIZE_TEXT, display=False):
    key = (src, size)
    if key in cache:
        return cache[key]
    s = '$' + fix_math(src) + '$'
    prop = FontProperties(size=size)
    w, h, depth, *_ = parser.parse(s, dpi=72, prop=prop)
    buf = io.BytesIO()
    math_to_image(s, buf, prop=prop, dpi=72, format='svg')
    b = base64.b64encode(buf.getvalue()).decode()
    style = f'width:{w:.2f}pt;height:{h:.2f}pt;vertical-align:{-depth:.2f}pt'
    tag = f'<img class="m" style="{style}" src="data:image/svg+xml;base64,{b}">'
    cache[key] = tag
    return tag


tex = (HERE / 'methodology_v7.tex').read_text(encoding='utf-8')
tex = re.sub(r'(?m)^%.*\n', '', tex)
title = re.search(r'\\title\{(.*?)\}\n', tex).group(1)
body = tex[tex.index(r'\maketitle') + len(r'\maketitle'):tex.index(r'\end{document}')]

# ---------------------------------------------------------------- numbering pass
refs = {}
sec = 0
for m in re.finditer(r'\\(subsection|section)\*?\{|\\label\{(.*?)\}|\\begin\{(equation|table|figure\*?|algorithm)\}', body):
    pass
counters = dict(eq=0, tab=0, fig=0, alg=0, sub=0)
cur_sub = None
pos = 0
token = re.compile(r'\\subsection\{|\\begin\{(equation|table|figure\*?|algorithm)\}|\\label\{(.*?)\}')
last = None
for m in token.finditer(body):
    if m.group(0).startswith(r'\subsection'):
        counters['sub'] += 1
        last = 'I-' + chr(64 + counters['sub'])
    elif m.group(1):
        kind = m.group(1)
        if kind == 'equation':
            counters['eq'] += 1
            last = str(counters['eq'])
        elif kind == 'table':
            counters['tab'] += 1
            last = ROMAN[counters['tab']]
        elif kind.startswith('figure'):
            counters['fig'] += 1
            last = str(counters['fig'])
        else:
            counters['alg'] += 1
            last = str(counters['alg'])
    else:
        refs[m.group(2)] = last
cites = {k: i + 1 for i, k in enumerate(re.findall(r'\\bibitem\{(.*?)\}', body))}


def inline(t, size=SIZE_TEXT):
    t = re.sub(r'\\label\{.*?\}', '', t)
    parts = re.split(r'(\$.*?\$)', t, flags=re.S)
    out = []
    for p in parts:
        if len(p) > 1 and p.startswith('$') and p.endswith('$'):
            inner = p[1:-1]
            inner = re.sub(r'\\ref\{(.*?)\}', lambda m: refs[m.group(1)], inner)
            out.append(math_svg(inner, size))
        else:
            out.append(text(p))
    res = ''.join(out)
    return re.sub(r'(\(?)(<img class="m"[^>]*>)([,.;:) %]*)',
                  lambda m: f'<span class="nb">{m.group(0)}</span>' if m.group(1) or m.group(3) else m.group(0), res)


def text(t):
    t = t.replace('&', '&amp;') if False else t
    t = re.sub(r'\\cite\{(.*?)\}', lambda m: f'[{cites[m.group(1)]}]', t)
    t = re.sub(r'\\ref\{(.*?)\}', lambda m: refs[m.group(1)], t)
    for _ in range(3):
        t = re.sub(r'\\textbf\{([^{}]*)\}', r'<b>\1</b>', t)
        t = re.sub(r'\\(?:emph|textit)\{([^{}]*)\}', r'<i>\1</i>', t)
        t = re.sub(r'\\texttt\{([^{}]*)\}', r'<code>\1</code>', t)
    rep = [("``", '“'), ("''", '”'), ('---', '—'), ('--', '–'), ('~', '\u00a0'), (r'\,', '\u202f'), (r'\%', '%'), (r'\_', '_'),
           (r"\'a", 'á'), (r'\\', '<br>'), ('\n', ' '), ("'", '’')]
    for a, b in rep:
        t = t.replace(a, b)
    return t


def caption(t, size=9):
    return inline(t.strip(), size)


def brace_arg(s, name):
    i = s.index('\\' + name + '{') + len(name) + 2
    d, j = 1, i
    while d:
        d += {'{': 1, '}': -1}.get(s[j], 0)
        j += 1
    return s[i:j - 1]


def table(block):
    cap = brace_arg(block, 'caption')
    lab = re.search(r'\\label\{(.*?)\}', block).group(1)
    tab = re.search(r'\\begin\{tabular\}\{.*?\}\n(.*?)\\end\{tabular\}', block, re.S)
    colspec = re.search(r'\\begin\{tabular\}\{(.*?)\}', block).group(1)
    rows = []
    head_done = False
    for line in tab.group(1).split('\n'):
        line = line.strip()
        if not line or line == r'\toprule':
            continue
        if line in (r'\midrule', r'\bottomrule'):
            if rows:
                rows[-1] = rows[-1].replace('<tr', '<tr class="rule"', 1)
            head_done = True
            continue
        line = line.rstrip('\\').strip()
        mc = re.match(r'\\multicolumn\{(\d+)\}\{.\}\{(.*)\}$', line)
        tagc = 'td' if head_done else 'th'
        if mc:
            rows.append(f'<tr><td colspan="{mc.group(1)}" class="l">{inline(mc.group(2), 8)}</td></tr>')
            continue
        cells = [c.strip() for c in line.split('&')]
        r = ''.join(f'<{tagc} class="{colspec[i] if i < len(colspec) else "l"}">{inline(c, 8)}</{tagc}>' for i, c in enumerate(cells))
        rows.append(f'<tr>{r}</tr>')
    note = ''
    after = block[tab.end():]
    nm = re.search(r'\\footnotesize\s(.*)', after, re.S)
    if nm:
        note = f'<div class="note">{inline(nm.group(1).strip(), 8)}</div>'
    return (f'<div class="tbl"><div class="tcap">TABLE {refs[lab]}</div><div class="tcap sc">{caption(cap, 8.5)}</div>'
            f'<table>{"".join(rows)}</table>{note}</div>')


def figure(block):
    src = re.search(r'\\includegraphics\[.*?\]\{(.*?)\}', block).group(1)
    cap = brace_arg(block, 'caption')
    lab = re.search(r'\\label\{(.*?)\}', block).group(1)
    data = base64.b64encode((HERE / src).read_bytes()).decode()
    return (f'<div class="fig"><img class="pic" src="data:image/png;base64,{data}">'
            f'<div class="fcap">Fig. {refs[lab]}. {caption(cap, 8.5)}</div></div>')


def algorithm(block):
    cap = brace_arg(block, 'caption')
    lab = re.search(r'\\label\{(.*?)\}', block).group(1)
    lines, n, ind = [], 0, 0
    for line in re.search(r'\\begin\{algorithmic\}\[1\]\n(.*?)\\end\{algorithmic\}', block, re.S).group(1).split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith(r'\Require'):
            lines.append(f'<div class="al"><b>Require:</b> {inline(line[8:].strip(), 9)}</div>')
            continue
        if line.startswith(r'\EndFor'):
            ind -= 1
            n += 1
            lines.append(f'<div class="al" style="padding-left:{ind * 10 + 16}pt"><span class="n">{n}:</span><b>end for</b></div>')
            continue
        n += 1
        if line.startswith(r'\For'):
            c = brace_arg(line, 'For')
            lines.append(f'<div class="al" style="padding-left:{ind * 10 + 16}pt"><span class="n">{n}:</span><b>for</b> {inline(c, 9)} <b>do</b></div>')
            ind += 1
            continue
        lines.append(f'<div class="al" style="padding-left:{ind * 10 + 16}pt"><span class="n">{n}:</span>{inline(line[6:].strip(), 9)}</div>')
    return f'<div class="alg"><div class="acap"><b>Algorithm {refs[lab]}</b> {cap}</div>{"".join(lines)}</div>'


def equation(block):
    lab = re.search(r'\\label\{(.*?)\}', block)
    src = re.sub(r'\\label\{.*?\}', '', block)
    return src, lab.group(1) if lab else None


# ---------------------------------------------------------------- body
html = []
top = ''
env = re.compile(r'\\begin\{(figure\*?|table|algorithm|equation|thebibliography)\}(?:\[.*?\])?(?:\{.*?\})?\n?(.*?)\\end\{\1\}', re.S)
idx = 0
eq_n = 0


def prose(chunk):
    for para in re.split(r'\n\s*\n', chunk):
        p = para.strip()
        if not p:
            continue
        p = re.sub(r'\\label\{.*?\}', '', p).strip()
        if not p:
            continue
        m = re.match(r'\\section\*?\{(.*?)\}\s*(.*)', p, re.S)
        if m:
            num = '' if m.group(1) == 'References' else 'I. '
            html.append(f'<h2>{num}{m.group(1)}</h2>')
            p = m.group(2).strip()
            if not p:
                continue
        m = re.match(r'\\subsection\{(.*?)\}\s*(.*)', p, re.S)
        if m:
            prose.sub += 1
            html.append(f'<h3>{chr(64 + prose.sub)}. {inline(m.group(1))}</h3>')
            p = m.group(2).strip()
            if not p:
                continue
        if p.startswith(r'\vspace') or p.startswith(r'\noindent'):
            html.append('<p class="sig">Signature of Supervisor: Prof. Samit Ari<br><br>Date:</p>')
            continue
        html.append(('<p class="cont">' if prose.cont else '<p>') + inline(p) + '</p>')
        prose.cont = False


prose.sub = 0
prose.cont = False

for m in env.finditer(body):
    chunk = body[idx:m.start()]
    # text directly after a display equation continues its paragraph
    prose(chunk)
    kind, block = m.group(1), m.group(2)
    if kind == 'figure*':
        top = figure(block)
    elif kind == 'figure':
        html.append(figure(block))
    elif kind == 'table':
        html.append(table(block))
    elif kind == 'algorithm':
        html.append(algorithm(block))
    elif kind == 'equation':
        eq_n += 1
        src, _ = equation(block)
        html.append(f'<div class="eq"><span class="eqb">{math_svg(src, 10)}</span><span class="eqn">({eq_n})</span></div>')
        prose.cont = True
    else:
        items = re.findall(r'\\bibitem\{(.*?)\}\s*(.*?)(?=\\bibitem|\Z)', block, re.S)
        html.append('<h2>References</h2><div class="bib">' + ''.join(f'<div class="bi"><span>[{cites[k]}]</span><div>{inline(v.strip(), 8.5)}</div></div>' for k, v in items) + '</div>')
    idx = m.end()
prose(body[idx:])

authors = ('<div class="auth">Shaswati Mishra (123EC0564)</div><div class="aff">Department of Electronics and Communication Engineering, '
           'National Institute of Technology, Rourkela</div><div class="aff">Supervisor: Prof. Samit Ari</div>')
css = '''
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
html { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.17; color: #000; }
body { margin: 0; }
h1 { font-weight: normal; font-size: 22pt; line-height: 1.12; text-align: center; margin: 2mm 0 3mm; }
.auth { text-align: center; font-size: 11pt; } .aff { text-align: center; font-size: 10pt; }
.top { margin: 4mm 0 4mm; }
.cols { column-count: 2; column-gap: 5mm; text-align: justify; hyphens: auto; orphans: 2; widows: 2; }
p { margin: 0; text-indent: 1em; } p.cont { text-indent: 0; } p.sig { text-indent: 0; margin-top: 6mm; }
h2 { font-weight: normal; font-variant: small-caps; font-size: 10.5pt; text-align: center; margin: 9pt 0 5pt; break-after: avoid; }
h3 { font-weight: normal; font-style: italic; font-size: 10pt; margin: 8pt 0 3pt; break-after: avoid; text-align: left; }
img.m { display: inline; } .nb { white-space: nowrap; }
.eq { display: flex; align-items: center; margin: 5pt 0; break-inside: avoid; }
.eqb { flex: 1; text-align: center; } .eqn { padding-left: 4pt; }
.fig, .tbl, .alg { break-inside: avoid; margin: 7pt 0 8pt; }
img.pic { width: 100%; display: block; }
.fcap { font-size: 8.5pt; margin-top: 3pt; text-align: justify; }
.tcap { text-align: center; font-size: 8.5pt; } .tcap.sc { font-variant: small-caps; margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; border-top: 1px solid #000; border-bottom: 1px solid #000; }
th, td { padding: 1pt 3pt; font-weight: normal; vertical-align: baseline; }
tr:first-child th { border-bottom: .5px solid #000; }
tr.rule td, tr.rule th { border-bottom: .5px solid #000; }
.l { text-align: left; } .r { text-align: right; } .c { text-align: center; }
.note { font-size: 8pt; margin-top: 2pt; text-align: justify; }
.alg { border-top: 1px solid #000; border-bottom: 1px solid #000; font-size: 9pt; padding: 2pt 0; text-align: left; hyphens: none; }
.acap { border-bottom: .5px solid #000; padding-bottom: 2pt; margin-bottom: 2pt; }
.al { position: relative; padding-left: 16pt; margin: .7pt 0; } .al .n { position: absolute; left: 0; font-size: 8pt; }
.bib { font-size: 8.5pt; text-align: left; } .bi { display: flex; gap: 4pt; margin: 1.4pt 0; break-inside: avoid; } .bi > span { min-width: 16pt; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
'''
doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>AnatoPhaseNet Methodology</title><style>{css}</style></head><body>'
       f'<h1>{title}</h1>{authors}<div class="top">{top}</div><div class="cols">{"".join(html)}</div></body></html>')
tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'methodology_v7.html'
tmp.write_text(doc, encoding='utf-8')
out = HERE / 'methodology_v7.pdf'
subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={out}', tmp.as_uri()],
               check=True, capture_output=True, timeout=120)
print('saved', out, out.stat().st_size, 'bytes; equations', eq_n, 'math snippets', len(cache))
