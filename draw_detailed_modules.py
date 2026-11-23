import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_detailed_diagram():
    # High resolution 300 DPI figure
    fig = plt.figure(figsize=(24, 14), dpi=300, facecolor='#FFFFFF')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 14)
    ax.axis('off')

    c_dark = '#0F172A'
    c_muted = '#475569'

    def draw_box(x, y, w, h, bg, ec, title, subtitle="", radius=0.15, lw=1.5, tc=None, ts=10, ss=8):
        title_color = tc if tc else c_dark
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                           fc=bg, ec=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        if title:
            if subtitle:
                ax.text(x + w/2, y + h/2 + 0.16, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=title_color, zorder=4)
                ax.text(x + w/2, y + h/2 - 0.20, subtitle, ha='center', va='center', fontsize=ss, color=c_muted, zorder=4)
            else:
                ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=title_color, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color='#64748B', style='-|>', lw=2, dashed=False):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    shrinkA=3, shrinkB=3, mutation_scale=15), zorder=5)

    # Header
    p_header = FancyBboxPatch((0.5, 12.6), 23.0, 1.1, boxstyle="round,pad=0.2",
                              fc='#1E293B', ec='#0F172A', lw=2, zorder=2)
    ax.add_patch(p_header)
    ax.text(12.0, 13.3, "Figure 2: Mathematical Architecture of the Three Novel Modules",
            ha='center', va='center', fontsize=18, fontweight='bold', color='#FFFFFF', zorder=4)
    ax.text(12.0, 12.85, "Foreground-Gated Axial Attention (FG-ASA), Harmonic Phase-Aware Temporal Attention (HPA-TA), and Temporal Reverse Attention Pooling (TRAP)",
            ha='center', va='center', fontsize=10.5, color='#94A3B8', zorder=4)

    # =========================================================================
    # MODULE 1: FG-ASA (LEFT PANEL)
    # =========================================================================
    p1 = FancyBboxPatch((0.5, 0.5), 7.2, 11.7, boxstyle="round,pad=0.2",
                        fc='#F5F3FF', ec='#8B5CF6', lw=2, zorder=2)
    ax.add_patch(p1)
    ax.text(4.1, 11.8, "[Novelty 1]: FG-ASA Module", ha='center', va='center', fontsize=13, fontweight='bold', color='#6D28D9', zorder=4)
    ax.text(4.1, 11.45, "Foreground-Gated Axial Spatial Attention", ha='center', va='center', fontsize=8.5, color=c_muted, zorder=4)

    draw_box(1.5, 10.0, 5.2, 1.0, '#FFFFFF', '#8B5CF6', "Input Feature Map", "X ∈ R^(B*T × C × H × W)", ts=9.5, ss=8)
    draw_arrow(4.1, 10.0, 4.1, 9.3, color='#8B5CF6')

    # Split into 3 branches
    draw_box(1.0, 7.8, 2.5, 1.3, '#EDE9FE', '#7C3AED', "Vertical Strip (H)", "Pool(H, 1) -> Conv 1×1\nBody lean, trunk tilt\nKnee lift height", ts=8.5, ss=7, tc='#6D28D9')
    draw_box(4.7, 7.8, 2.5, 1.3, '#EDE9FE', '#7C3AED', "Horizontal Strip (W)", "Pool(1, W) -> Conv 1×1\nStep length, stance base\nScissor gait crossing", ts=8.5, ss=7, tc='#6D28D9')

    draw_arrow(3.0, 9.3, 2.25, 9.1, color='#7C3AED')
    draw_arrow(5.2, 9.3, 5.95, 9.1, color='#7C3AED')

    # Cross sum
    draw_box(2.0, 6.1, 4.2, 1.1, '#FFFFFF', '#7C3AED', "Spatial Additive Fusion", "σ( Feat_H(x) + Feat_W(x) )\nAxial Attention Map: [B*T, C/2, H, W]", ts=8.5, ss=7.5)
    draw_arrow(2.25, 7.8, 3.2, 7.2, color='#7C3AED')
    draw_arrow(5.95, 7.8, 5.0, 7.2, color='#7C3AED')

    # Silhouette Saliency Mask Branch
    draw_box(1.0, 4.3, 2.5, 1.2, '#FEF3C7', '#D97706', "Silhouette Saliency", "Conv 3×3 + Sigmoid\nZeroes out 70% black\nbackground sparsity", ts=8.5, ss=7, tc='#B45309')
    draw_box(4.7, 4.3, 2.5, 1.2, '#E0E7FF', '#4338CA', "Channel Attention Gate", "AdaptiveAvgPool(1×1)\nMLP + Sigmoid\nInter-channel semantics", ts=8.5, ss=7, tc='#3730A3')

    # Multiplicative Cross-Gating
    draw_box(1.2, 2.2, 5.8, 1.4, '#DDD6FE', '#6D28D9', "Multiplicative Cross-Gating", "X_hat = SpatialAttn ⊙ ChannelGate ⊙ Mask_fg\nConv 1×1 + BatchNorm -> [B*T, C, H, W]", ts=9, ss=7.5, tc='#4C1D95')
    draw_arrow(4.1, 6.1, 4.1, 3.6, color='#7C3AED')
    draw_arrow(2.25, 4.3, 3.0, 3.6, color='#D97706')
    draw_arrow(5.95, 4.3, 5.2, 3.6, color='#4338CA')

    # Residual addition
    draw_box(1.5, 0.8, 5.2, 0.9, '#FFFFFF', '#8B5CF6', "Residual Update", "Output: X + X_hat", ts=9, ss=7.5, tc='#6D28D9')
    draw_arrow(4.1, 2.2, 4.1, 1.7, color='#6D28D9')

    # =========================================================================
    # MODULE 2: HPA-TA (CENTER PANEL)
    # =========================================================================
    p2 = FancyBboxPatch((8.4, 0.5), 7.2, 11.7, boxstyle="round,pad=0.2",
                        fc='#FFFBEB', ec='#F59E0B', lw=2, zorder=2)
    ax.add_patch(p2)
    ax.text(12.0, 11.8, "[Novelty 2]: HPA-TA Module", ha='center', va='center', fontsize=13, fontweight='bold', color='#B45309', zorder=4)
    ax.text(12.0, 11.45, "Harmonic Phase-Aware Temporal Attention", ha='center', va='center', fontsize=8.5, color=c_muted, zorder=4)

    draw_box(9.4, 10.0, 5.2, 1.0, '#FFFFFF', '#F59E0B', "Input 5D Spatio-Temporal Tensor", "X ∈ R^(B × C × T=16 × H × W)", ts=9.5, ss=8)
    draw_arrow(12.0, 10.0, 12.0, 9.3, color='#F59E0B')

    # 3D Conv (Guo et al. inspiration)
    draw_box(9.0, 8.0, 6.0, 1.3, '#FEF3C7', '#D97706', "3D Temporal Convolution", "nn.Conv3d(kernel_size=(3, 1, 1), padding=(1, 0, 0))\nFuses consecutive frames: t-1 <-> t <-> t+1\nCaptures instantaneous step velocity & flow", ts=9, ss=7.5, tc='#92400E')
    draw_arrow(12.0, 8.0, 12.0, 7.3, color='#D97706')

    # Frame token pool + Q, K
    draw_box(9.0, 6.0, 6.0, 1.3, '#FFFFFF', '#D97706', "Frame Token Extraction", "AdaptiveAvgPool3d(T, 1, 1) -> Tokens ∈ R^(B × C/2 × T)\nQ = Conv1d(Tokens), K = Conv1d(Tokens)\nDot-Product: (Q × K^T) / √d ∈ R^(B × T × T)", ts=9, ss=7.5)
    draw_arrow(12.0, 6.0, 12.0, 5.3, color='#D97706')

    # Novelty: Harmonic Phase Prior Box
    draw_box(9.0, 3.8, 6.0, 1.5, '#FDE68A', '#B45309', "[NOVELTY]: Fourier Harmonic Gait Prior", "Φ_harmonic(i, j) = Σ [ w_c*cos(2πk·Δt/T) + w_s*sin(2πk·Δt/T) ]\nInjects cyclic bipedal cadence into attention\nIdentifies Parkinsonian festination & hemiplegic asymmetry", ts=9, ss=7.5, tc='#78350F')
    draw_arrow(12.0, 3.8, 12.0, 3.1, color='#B45309')

    # Attention Matrix & Multiply
    draw_box(9.0, 2.0, 6.0, 1.1, '#FFFFFF', '#D97706', "Harmonic Attention Matrix & Values", "M_T = Softmax( (Q K^T)/√d + Φ_harmonic )\nAttended = M_T × Flatten(X_temp)", ts=9, ss=7.5)
    draw_arrow(12.0, 2.0, 12.0, 1.7, color='#D97706')

    draw_box(9.4, 0.8, 5.2, 0.9, '#FFFFFF', '#F59E0B', "Residual Output", "Output: X_seq + γ · Conv1×1(Attended)", ts=9, ss=7.5, tc='#92400E')

    # =========================================================================
    # MODULE 3: TRAP (RIGHT PANEL)
    # =========================================================================
    p3 = FancyBboxPatch((16.3, 0.5), 7.2, 11.7, boxstyle="round,pad=0.2",
                        fc='#FDF2F8', ec='#EC4899', lw=2, zorder=2)
    ax.add_patch(p3)
    ax.text(19.9, 11.8, "[Novelty 3]: TRAP Head", ha='center', va='center', fontsize=13, fontweight='bold', color='#BE185D', zorder=4)
    ax.text(19.9, 11.45, "Temporal Reverse Attention Pooling Head", ha='center', va='center', fontsize=8.5, color=c_muted, zorder=4)

    draw_box(17.3, 10.0, 5.2, 1.0, '#FFFFFF', '#EC4899', "Input Multi-Scale RCAF Tokens", "R_seq ∈ R^(B × T=16 × 192)", ts=9.5, ss=8)
    draw_arrow(19.9, 10.0, 19.9, 9.2, color='#EC4899')

    # Temporal Saliency Scorer
    draw_box(17.0, 7.8, 5.8, 1.4, '#FCE7F3', '#DB2777', "Temporal Saliency Scorer", "Linear(192->64) + LayerNorm + GELU\nLinear(64->1) + Sigmoid\nGenerates Frame Saliency: α_t ∈ [0, 1]", ts=9, ss=7.5, tc='#9D174D')
    draw_arrow(19.9, 7.8, 19.9, 7.1, color='#DB2777')

    # Dual Perspective Branches
    draw_box(16.7, 5.1, 3.1, 1.8, '#FFFFFF', '#DB2777', "Branch 1: Dominant", "w_dom = α_t / Σα\nPools peak stride frames\nHeel-strike, arm swing\nf_dom ∈ R^(B × 192)", ts=8.5, ss=7, tc='#9D174D')
    draw_box(20.1, 5.1, 3.1, 1.8, '#FFF1F2', '#E11D48', "Branch 2: Reverse", "w_rev = (1 - α_t) / Σ(1-α)\nPools suppressed frames\nFreezing, micro-tremors\nf_rev ∈ R^(B × 192)", ts=8.5, ss=7, tc='#9F1239')

    draw_arrow(18.25, 7.1, 18.25, 6.9, color='#DB2777')
    draw_arrow(21.65, 7.1, 21.65, 6.9, color='#E11D48')

    # Dual Concatenation
    draw_box(17.0, 3.1, 5.8, 1.4, '#FBCFE8', '#9D174D', "Dual-Perspective Concatenation", "F_gait = [ f_dom ∥ f_rev ] ∈ R^(B × 384)\nCombines prominent stride with subtle hesitation", ts=9, ss=7.5, tc='#700733')
    draw_arrow(18.25, 5.1, 19.0, 4.5, color='#DB2777')
    draw_arrow(21.65, 5.1, 20.8, 4.5, color='#E11D48')

    # Final Classifier
    draw_box(17.0, 0.8, 5.8, 1.7, '#0F172A', '#1E293B', "Pathology Classification Head", "Linear(384->128) + BatchNorm + GELU + Dropout(0.3)\nLinear(128->5 Classes) + Softmax\n[Normal, Diplegic, Hemiplegic, Neuropathic, Parkinson]", ts=9, ss=7.5, tc='#38BDF8')
    draw_arrow(19.9, 3.1, 19.9, 2.5, color='#0F172A', lw=2.5)

    plt.tight_layout()
    out_file = "st_cranet_detailed_modules.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    print(f"[SUCCESS] Saved detailed diagram to {out_file}")

    artifact_dir = r"C:\Users\Asus\.gemini\antigravity\brain\9d2370a3-794c-4570-b805-58317a7724b3"
    if os.path.exists(artifact_dir):
        shutil.copyfile(out_file, os.path.join(artifact_dir, out_file))
        print(f"[SUCCESS] Copied to artifact dir: {artifact_dir}")

if __name__ == "__main__":
    create_detailed_diagram()
