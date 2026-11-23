# AnatoPhaseNet v7

Corrected, self-contained version of the final model. The architecture, losses and training settings are those of v6 (geometry branch on, subject adversary off, (2+1)D backbone); the changes are in the data handling.

## Files
| File | Contents |
|---|---|
| `anato_phasenet_v7.py` | Model, training and evaluation in one file; v4/v5 are no longer imported |
| `architecture_v7.png`, `.pdf` | Architecture diagram (`draw_architecture_v7.py`) |
| `methodology_v7.pdf` | Methodology, rendered locally from the `.tex` (`build_methodology_pdf.py`) |
| `methodology_v7.tex` | Methodology source for Overleaf (IEEEtran); upload with the three PNG figures |
| `fig_init_cycles.png`, `fig_geometry.png` | Methodology figures (`make_method_figures.py`) |
| `RESULTS_SUMMARY.md` | v7 results beside v6 (`make_results_summary.py`) |
| `results_gait_it_r2plus1d_full/`, `results_init6_r2plus1d_full/` | Per-fold metrics, predictions, curves, confusion matrices |
| `run_log_gait_it.txt` | Log of the stopped first GAIT-IT run (folds 1-2 and part of 3); it does not belong to the reported results, which were run without a log file |
| `cache/` | Image and geometry caches built by v7 |

## Corrections against v6
1. **GAIT-IT cycles:** the key-frame list is read in blocks of nine, not eight, so every cycle starts at phase 0 (3783 cycles, was 3813).
2. **Walk identity:** the walk id includes the class, so s9's hemiplegic and diplegic recordings stay apart (821 walks, was 813).
3. **Geometry signals:** signals that are constant over a dataset are dropped: height and head bob on GAIT-IT (7 signals; 9 on INIT).
4. **Geometry noise:** its scale comes from each fold's training cycles, not the whole dataset.
5. **Crop:** the box includes its last row and column, and the resized side is rounded.
6. **Guards:** the run stops on a repeated walk id, a walk mixing labels or subjects, an unexpected key-frame layout, or a subject with two roles in a fold.

## Running
From this folder:
```
python anato_phasenet_v7.py --dataset gait_it
python anato_phasenet_v7.py --dataset init --init_classes 6
python make_results_summary.py
```
Use the Python that has PyTorch installed (`Python313` on this machine). The dataset and `resnet18_pretrained.pth` are read from the parent folder.
