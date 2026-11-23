import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import numpy as np

def create_diagram():
    # High resolution figure for publication
    fig = plt.figure(figsize=(26, 16), dpi=300, facecolor='#FFFFFF')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Color Palette (Publication Grade)
    c_bg_panel = '#F8FAFC'
    c_border = '#CBD5E1'
    c_encoder = '#3B82F6'       # Vibrant Blue
    c_encoder_bg = '#EFF6FF'
    c_decoder = '#06B6D4'       # Cyan / Teal
    c_decoder_bg = '#ECFEFF'
    c_bottleneck = '#8B5CF6'    # Purple / Violet
    c_bottleneck_bg = '#F5F3FF'
    c_rcaf = '#10B981'          # Emerald Green
    c_rcaf_bg = '#ECFDF5'
    c_temp = '#F59E0B'          # Amber / Orange
    c_temp_bg = '#FFFBEB'
    c_trap = '#EC4899'          # Rose / Pink
    c_trap_bg = '#FDF2F8'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'

    # Helper function for drawing rounded boxes with drop shadow feel
    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", radius=0.2, lw=1.8, title_color=None, title_size=11, sub_size=9):
        tc = title_color if title_color else c_text_dark
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                           fc=bg_color, ec=border_color, lw=lw, zorder=3)
        ax.add_patch(p)
        if title:
            if subtitle:
                ax.text(x + w/2, y + h/2 + 0.18, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=tc, zorder=4)
                ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha='center', va='center', fontsize=sub_size, color=c_text_muted, zorder=4)
            else:
                ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=tc, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color='#64748B', style='-|>', lw=2, dashed=False, zorder=5):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    shrinkA=4, shrinkB=4, mutation_scale=16), zorder=zorder)

    # =========================================================================
    # 1. HEADER BANNER
    # =========================================================================
    p_header = FancyBboxPatch((0.5, 14.3), 25.0, 1.3, boxstyle="round,pad=0.2",
                              fc='#0F172A', ec='#1E293B', lw=2, zorder=2)
    ax.add_patch(p_header)
    ax.text(13.0, 15.15, "ST-CRANet: Spatio-Temporal Cross Reverse Attention Network",
            ha='center', va='center', fontsize=20, fontweight='bold', color='#FFFFFF', zorder=4)
    ax.text(13.0, 14.65, "End-to-End Architecture: Foreground-Gated Axial Attention, 3D Temporal Conv + Harmonic Gait-Phase Attention, RCAF Skip Fusion & TRAP Head",
            ha='center', va='center', fontsize=11, color='#94A3B8', zorder=4)

    # =========================================================================
    # 2. LEFT PANEL: INPUT GAIT SEQUENCE
    # =========================================================================
    p_inp_bg = FancyBboxPatch((0.5, 3.2), 2.7, 10.7, boxstyle="round,pad=0.2",
                              fc='#F1F5F9', ec='#CBD5E1', lw=1.5, zorder=2)
    ax.add_patch(p_inp_bg)
    ax.text(1.85, 13.5, "Input Gait Sequence", ha='center', va='center', fontsize=13, fontweight='bold', color=c_text_dark, zorder=4)
    ax.text(1.85, 13.15, "X ∈ R^(B × T × 1 × 224 × 224)", ha='center', va='center', fontsize=8.5, color=c_text_muted, zorder=4)

    # Draw stacked silhouette frames
    for i in range(4):
        offset = (3 - i) * 0.25
        frame_box = FancyBboxPatch((0.9 + offset, 8.8 + offset*1.8), 1.5, 2.0, boxstyle="round,pad=0.08",
                                   fc='#1E293B', ec='#0284C7', lw=1.5, zorder=4+i)
        ax.add_patch(frame_box)
        # Dummy silhouette outline
        ax.plot([1.65 + offset, 1.65 + offset], [9.8 + offset*1.8, 10.3 + offset*1.8], color='#38BDF8', lw=2.5, zorder=5+i) # torso
        ax.plot([1.65 + offset, 1.45 + offset], [9.8 + offset*1.8, 9.2 + offset*1.8], color='#38BDF8', lw=2, zorder=5+i)   # back leg
        ax.plot([1.65 + offset, 1.85 + offset], [9.8 + offset*1.8, 9.1 + offset*1.8], color='#38BDF8', lw=2, zorder=5+i)   # front leg
        ax.plot([1.65 + offset, 1.45 + offset], [10.2 + offset*1.8, 9.7 + offset*1.8], color='#38BDF8', lw=1.8, zorder=5+i) # arm
        circle = plt.Circle((1.65 + offset, 10.5 + offset*1.8), 0.16, color='#38BDF8', zorder=5+i)
        ax.add_patch(circle)

    ax.text(1.85, 8.3, "T Chronological Frames\nUniform Gait-Cycle Sampling", ha='center', va='center', fontsize=9.5, color=c_text_dark, fontweight='semibold', zorder=6)
    
    # 2D View transform note
    draw_box(0.7, 4.0, 2.3, 3.6, '#FFFFFF', '#94A3B8', "Frame Batching", "Reshape: [B×T, 1, 224, 224]\nExtracts multi-scale 2D\nspatio-anatomical features\nper silhouette frame", radius=0.15, title_size=10, sub_size=8)
    draw_arrow(1.85, 7.8, 1.85, 7.6, color=c_encoder, lw=2.5)

    # Arrow from Input to Encoder 1
    draw_arrow(3.0, 5.8, 3.7, 5.8, color=c_encoder, lw=2.5)

    # =========================================================================
    # 3. U-NET BACKBONE: ENCODERS, BOTTLENECK, DECODERS
    # =========================================================================
    # Panel Background for U-Net
    p_unet_bg = FancyBboxPatch((3.5, 0.5), 11.2, 13.4, boxstyle="round,pad=0.2",
                               fc='#FFFFFF', ec='#E2E8F0', lw=2, zorder=1)
    ax.add_patch(p_unet_bg)
    ax.text(9.1, 13.5, "Hierarchical Spatio-Temporal U-Net Backbone", ha='center', va='center', fontsize=14, fontweight='bold', color=c_text_dark, zorder=3)

    # ENCODERS (Left column of U-Net)
    draw_box(4.0, 11.0, 2.6, 1.5, c_encoder_bg, c_encoder, "Encoder Stage 1", "Conv 3×3 + BN + ReLU\nMaxPool 2×2 -> [64, 112, 112]", title_color=c_encoder)
    draw_box(4.0, 8.0, 2.6, 1.5, c_encoder_bg, c_encoder, "Encoder Stage 2", "Conv 3×3 + BN + ReLU\nMaxPool 2×2 -> [128, 56, 56]", title_color=c_encoder)
    draw_box(4.0, 5.0, 2.6, 1.5, c_encoder_bg, c_encoder, "Encoder Stage 3", "Conv 3×3 + BN + ReLU\nMaxPool 2×2 -> [256, 28, 28]", title_color=c_encoder)

    # Downsampling Arrows
    draw_arrow(5.3, 11.0, 5.3, 9.5, color=c_encoder, lw=2.2)
    draw_arrow(5.3, 8.0, 5.3, 6.5, color=c_encoder, lw=2.2)
    draw_arrow(5.3, 5.0, 5.3, 3.8, color=c_encoder, lw=2.2)

    # BOTTLENECK: PROPOSED GAIT-DSTA MODULE
    p_bn_outer = FancyBboxPatch((4.0, 1.0), 10.2, 2.6, boxstyle="round,pad=0.2",
                                fc=c_bottleneck_bg, ec=c_bottleneck, lw=2.5, zorder=3)
    ax.add_patch(p_bn_outer)
    ax.text(9.1, 3.25, "[NOVELTY 1 & 2] Gait-DSTA Bottleneck: FG-ASA + 3D Conv + Harmonic Gait-Phase Attention",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_bottleneck, zorder=4)

    # Sub-blocks inside Bottleneck
    draw_box(4.3, 1.3, 2.2, 1.6, '#FFFFFF', c_bottleneck, "Bottleneck Conv", "Conv 3×3 + BN + ReLU\n[B×T, 512, 14, 14]", title_size=9.5, sub_size=7.5)
    draw_box(6.8, 1.3, 2.3, 1.6, '#EDE9FE', '#7C3AED', "1. FG-ASA", "Axial Strips (H & W)\nForeground Saliency Gate\n[Posture & Stride]", title_size=9.5, sub_size=7.5, title_color='#7C3AED')
    draw_box(9.4, 1.3, 2.3, 1.6, '#FEF3C7', '#D97706', "2. 3D Temp Conv", "Kernel (3×1×1)\nShort-term stride flow\nt <-> t±1 transitions", title_size=9.5, sub_size=7.5, title_color='#D97706')
    draw_box(11.9, 1.3, 2.0, 1.6, '#FDF2F8', '#BE185D', "3. Harmonic Attn", "Fourier Gait Prior\nCycle Rhythm Matrix\n[B, T, T] Softmax", title_size=9.5, sub_size=7.5, title_color='#BE185D')

    # Bottleneck internal flow arrows
    draw_arrow(6.5, 2.1, 6.8, 2.1, color=c_bottleneck, lw=2)
    draw_arrow(9.1, 2.1, 9.4, 2.1, color=c_temp, lw=2)
    draw_arrow(11.7, 2.1, 11.9, 2.1, color=c_trap, lw=2)

    # Arrow from Bottleneck to Decoder 1
    draw_arrow(12.9, 3.6, 12.9, 5.0, color=c_decoder, lw=2.2)

    # DECODERS (Right column of U-Net)
    draw_box(11.6, 5.0, 2.6, 1.5, c_decoder_bg, c_decoder, "Decoder Stage 1", "UpConv + Conv 3×3\nOutput: [256, 28, 28]", title_color=c_decoder)
    draw_box(11.6, 8.0, 2.6, 1.5, c_decoder_bg, c_decoder, "Decoder Stage 2", "UpConv + Conv 3×3\nOutput: [128, 56, 56]", title_color=c_decoder)
    draw_box(11.6, 11.0, 2.6, 1.5, c_decoder_bg, c_decoder, "Decoder Stage 3", "UpConv + Conv 3×3\nOutput: [64, 112, 112]", title_color=c_decoder)

    # Upsampling Arrows
    draw_arrow(12.9, 6.5, 12.9, 8.0, color=c_decoder, lw=2.2)
    draw_arrow(12.9, 9.5, 12.9, 11.0, color=c_decoder, lw=2.2)

    # U-Net Skip Connections (Dashed horizontal lines)
    draw_arrow(6.6, 5.75, 11.6, 5.75, color='#64748B', style='-|>', lw=1.8, dashed=True)
    ax.text(9.1, 6.0, "skip3: 256ch", ha='center', va='center', fontsize=8.5, color='#475569', zorder=5, backgroundcolor='#FFFFFF')

    draw_arrow(6.6, 8.75, 11.6, 8.75, color='#64748B', style='-|>', lw=1.8, dashed=True)
    ax.text(9.1, 9.0, "skip2: 128ch", ha='center', va='center', fontsize=8.5, color='#475569', zorder=5, backgroundcolor='#FFFFFF')

    draw_arrow(6.6, 11.75, 11.6, 11.75, color='#64748B', style='-|>', lw=1.8, dashed=True)
    ax.text(9.1, 12.0, "skip1: 64ch", ha='center', va='center', fontsize=8.5, color='#475569', zorder=5, backgroundcolor='#FFFFFF')

    # =========================================================================
    # 4. MIDDLE-RIGHT: MULTI-SCALE RCAF FUSION MODULES (PRESERVED 100%)
    # =========================================================================
    p_rcaf_bg = FancyBboxPatch((15.2, 4.3), 4.3, 8.6, boxstyle="round,pad=0.2",
                               fc='#F0FDF4', ec=c_rcaf, lw=2, zorder=2)
    ax.add_patch(p_rcaf_bg)
    ax.text(17.35, 12.5, "RCAF Multi-Scale Fusion", ha='center', va='center', fontsize=13, fontweight='bold', color=c_rcaf, zorder=4)
    ax.text(17.35, 12.15, "(Preserved 100% from CRA-Net)", ha='center', va='center', fontsize=8.5, color=c_text_muted, zorder=4)

    draw_box(15.5, 10.3, 3.7, 1.4, '#FFFFFF', c_rcaf, "RCAF Module 3", "skip1 (64ch) + d3 (64ch)\nr3: 64ch @ 112×112", title_color=c_rcaf, title_size=10, sub_size=8)
    draw_box(15.5, 8.1, 3.7, 1.4, '#FFFFFF', c_rcaf, "RCAF Module 2", "skip2 (128ch) + d2 (128ch)\nr2: 64ch @ 56×56", title_color=c_rcaf, title_size=10, sub_size=8)
    draw_box(15.5, 5.9, 3.7, 1.4, '#FFFFFF', c_rcaf, "RCAF Module 1", "skip3 (256ch) + d1 (256ch)\nr1: 64ch @ 28×28", title_color=c_rcaf, title_size=10, sub_size=8)

    # Connections from Decoders to RCAF
    draw_arrow(14.2, 11.75, 15.5, 11.0, color=c_rcaf, lw=2)
    draw_arrow(14.2, 8.75, 15.5, 8.8, color=c_rcaf, lw=2)
    draw_arrow(14.2, 5.75, 15.5, 6.6, color=c_rcaf, lw=2)

    # RCAF Internal Callout Box (Showing 4 dilated convs + reverse attention)
    draw_box(15.5, 4.6, 3.7, 1.0, '#DCFCE7', '#059669', "RCAF Dilated & Reverse Attn", "4 Convs (d=1,2,3,4) + [fr*(1-σ)]", title_color='#065F46', title_size=8.5, sub_size=7.5)

    # Multiscale Concatenation Block
    draw_box(15.5, 1.0, 3.7, 2.6, '#ECFDF5', '#047857', "Multi-Scale Token Concat",
             "Resize r1, r3 -> 56×56\nConcat: [r2, r1_res, r3_res]\nChannel Dim: 64×3 = 192\nAdaptive Global Pool (1×1)\nOutput: R_seq ∈ R^(B × T × 192)",
             title_color='#047857', title_size=10.5, sub_size=8.5)

    draw_arrow(17.35, 5.9, 17.35, 3.6, color=c_rcaf, lw=2.5)

    # =========================================================================
    # 5. FAR-RIGHT PANEL: [NOVELTY 3] TRAP HEAD & CLASSIFICATION
    # =========================================================================
    p_trap_bg = FancyBboxPatch((20.0, 0.5), 5.5, 13.4, boxstyle="round,pad=0.2",
                               fc=c_trap_bg, ec=c_trap, lw=2, zorder=2)
    ax.add_patch(p_trap_bg)
    ax.text(22.75, 13.5, "[NOVELTY 3] TRAP Head", ha='center', va='center', fontsize=14, fontweight='bold', color='#BE185D', zorder=4)
    ax.text(22.75, 13.15, "Temporal Reverse Attention Pooling", ha='center', va='center', fontsize=9, color=c_text_muted, zorder=4)

    # Input to TRAP arrow from Multiscale concat
    draw_arrow(19.2, 2.3, 20.4, 2.3, color='#BE185D', lw=2.5)
    draw_arrow(20.4, 2.3, 20.4, 11.2, color='#BE185D', lw=2)
    draw_arrow(20.4, 11.2, 21.0, 11.2, color='#BE185D', lw=2)

    # TRAP Saliency Gate
    draw_box(21.0, 10.4, 3.8, 1.5, '#FFFFFF', '#DB2777', "Temporal Scorer", "Linear(192->64) + LN + GELU\nLinear(64->1) + Sigmoid\nOutput: α_t ∈ [B, T, 1]", title_color='#BE185D', title_size=9.5, sub_size=8)

    # Dual Branch Boxes
    draw_box(20.8, 7.8, 4.0, 1.7, '#FDF2F8', '#DB2777', "Branch A: Dominant Motion", "Weight: w_dom = α_t / Σα\nFocus: Stride Peaks & Cadence\nFeature: f_dom ∈ R^(B × 192)", title_color='#9D174D', title_size=9.5, sub_size=8)
    draw_box(20.8, 5.4, 4.0, 1.7, '#FFF1F2', '#E11D48', "Branch B: Reverse Hesitation", "Weight: w_rev = (1 - α_t) / Σ(1-α)\nFocus: Micro-tremors & Hesitations\nFeature: f_rev ∈ R^(B × 192)", title_color='#9F1239', title_size=9.5, sub_size=8)

    draw_arrow(22.9, 10.4, 22.9, 9.5, color='#BE185D', lw=2)
    draw_arrow(22.9, 9.5, 22.9, 7.8, color='#BE185D', lw=2)
    draw_arrow(22.9, 7.8, 22.9, 7.1, color='#BE185D', lw=2)

    # Concatenation of Dual Branches
    draw_box(20.8, 3.4, 4.0, 1.4, '#FFFFFF', '#9D174D', "Dual-Perspective Fusion", "Concat: [f_dom ∥ f_rev]\nUnified Signature: R^(B × 384)", title_color='#831843', title_size=9.5, sub_size=8)
    draw_arrow(22.9, 5.4, 22.9, 4.8, color='#BE185D', lw=2)

    # Classifier & Output
    draw_box(20.8, 1.0, 4.0, 1.8, '#0F172A', '#1E293B', "Pathology Classifier", "Linear(384->128) + BN + GELU + Drop\nLinear(128->5 Classes) + Softmax\n[Normal, Diplegic, Hemiplegic,\nNeuropathic, Parkinsonian]", title_color='#38BDF8', title_size=9.5, sub_size=7.5)
    draw_arrow(22.9, 3.4, 22.9, 2.8, color='#0F172A', lw=2.5)

    # =========================================================================
    # 6. LEGEND AT TOP RIGHT
    # =========================================================================
    p_leg = FancyBboxPatch((15.2, 13.0), 9.5, 1.1, boxstyle="round,pad=0.1",
                           fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=3)
    ax.add_patch(p_leg)
    ax.text(15.5, 13.8, "LEGEND:", fontsize=8.5, fontweight='bold', color=c_text_dark, zorder=4)

    # Legend items
    items = [
        (16.4, 13.8, c_encoder, "2D Encoder"),
        (18.2, 13.8, c_decoder, "2D Decoder"),
        (20.0, 13.8, c_bottleneck, "FG-ASA Spatial"),
        (21.9, 13.8, c_temp, "3D Temporal Conv"),
        (23.7, 13.8, c_trap, "TRAP Reverse Head"),
        (16.4, 13.3, c_rcaf, "RCAF Module (Original)"),
        (19.2, 13.3, '#64748B', "Skip Connection (---)"),
        (22.2, 13.3, '#0F172A', "Sequence Prediction")
    ]
    for lx, ly, col, lab in items:
        circ = plt.Circle((lx, ly), 0.1, color=col, zorder=5)
        ax.add_patch(circ)
        ax.text(lx + 0.18, ly, lab, fontsize=7.5, va='center', color=c_text_dark, zorder=5)

    plt.tight_layout()
    
    # Save in workspace
    workspace_out = "st_cranet_architecture_diagram.png"
    plt.savefig(workspace_out, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    print(f"[SUCCESS] Saved diagram to {workspace_out}")

    # Copy to artifact directory for markdown embedding
    artifact_dir = r"C:\Users\Asus\.gemini\antigravity\brain\9d2370a3-794c-4570-b805-58317a7724b3"
    if os.path.exists(artifact_dir):
        artifact_out = os.path.join(artifact_dir, "st_cranet_architecture_diagram.png")
        shutil.copyfile(workspace_out, artifact_out)
        print(f"[SUCCESS] Copied diagram to artifact directory: {artifact_out}")

if __name__ == "__main__":
    create_diagram()
