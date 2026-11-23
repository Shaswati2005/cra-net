import base64
import os

with open('st_cranet_architecture_diagram.png', 'rb') as f:
    b64_fig1 = base64.b64encode(f.read()).decode('utf-8')

with open('st_cranet_detailed_modules.png', 'rb') as f:
    b64_fig2 = base64.b64encode(f.read()).decode('utf-8')

artifact_dir = r"C:\Users\Asus\.gemini\antigravity\brain\9d2370a3-794c-4570-b805-58317a7724b3"
os.makedirs(artifact_dir, exist_ok=True)

html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .tab-btn.active {{ background: #2563eb !important; color: #ffffff !important; border-color: #2563eb !important; }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-3 shadow-md max-w-5xl mx-auto">
    <div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] pb-2 mb-2">
      <div>
        <h2 class="text-sm font-bold text-[var(--foreground)]">ST-CRANet Architecture Figures (300 DPI Publication Quality)</h2>
        <p class="text-xs text-[var(--muted-foreground)]">Switch tabs to view Figure 1 or Figure 2</p>
      </div>
      <div class="flex gap-2">
        <button id="btn-fig1" onclick="showFig('fig1')" class="tab-btn active px-3 py-1 text-xs font-semibold rounded-lg border border-[var(--border)] transition">Figure 1: Full Architecture</button>
        <button id="btn-fig2" onclick="showFig('fig2')" class="tab-btn px-3 py-1 text-xs font-semibold rounded-lg border border-[var(--border)] transition bg-[var(--surface)] text-[var(--foreground)]">Figure 2: Novel Modules Detail</button>
      </div>
    </div>

    <!-- Figure 1 Container -->
    <div id="fig1-container" class="block">
      <div class="overflow-auto max-h-[380px] rounded-lg border border-[var(--border)] bg-white p-2 flex justify-center">
        <img src="data:image/png;base64,{b64_fig1}" alt="ST-CRANet Overall Architecture" class="w-full object-contain rounded shadow-sm">
      </div>
      <p class="text-xs text-[var(--muted-foreground)] mt-1.5 italic"><b>Figure 1:</b> Complete end-to-end ST-CRANet with 2D U-Net backbone, FG-ASA, 3D Conv + Harmonic Gait-Phase Attention, RCAF skip fusion, and TRAP Head.</p>
    </div>

    <!-- Figure 2 Container -->
    <div id="fig2-container" class="hidden">
      <div class="overflow-auto max-h-[380px] rounded-lg border border-[var(--border)] bg-white p-2 flex justify-center">
        <img src="data:image/png;base64,{b64_fig2}" alt="ST-CRANet Novel Modules Detail" class="w-full object-contain rounded shadow-sm">
      </div>
      <p class="text-xs text-[var(--muted-foreground)] mt-1.5 italic"><b>Figure 2:</b> Zoomed mathematical formulation of FG-ASA (Axial strips), HPA-TA (Harmonic phase prior), and TRAP (Dual-perspective pooling).</p>
    </div>
  </div>

  <script>
    function showFig(id) {{
      const f1 = document.getElementById('fig1-container');
      const f2 = document.getElementById('fig2-container');
      const b1 = document.getElementById('btn-fig1');
      const b2 = document.getElementById('btn-fig2');
      if (id === 'fig1') {{
        f1.classList.remove('hidden');
        f2.classList.add('hidden');
        b1.classList.add('active');
        b2.classList.remove('active');
      }} else {{
        f1.classList.add('hidden');
        f2.classList.remove('hidden');
        b2.classList.add('active');
        b1.classList.remove('active');
      }}
    }}
  </script>
</body>
</html>"""

out_path = os.path.join(artifact_dir, 'viewer.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("[SUCCESS] Created viewer.html at:", out_path)
