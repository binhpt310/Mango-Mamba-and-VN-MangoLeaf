# Mango Mamba Final — Evidence Report

Updated: 2026-09-12. This is the only Markdown report in this workspace.

## Naming and protocol

`V` means Version. `V0` is the original local baseline and `V3` is **Mango Mamba Final**. `V1` and `V2` are not included in the rebuilt tables because their complete local result packages are not present. MangoLeafBD is curriculum Phase 1, Combined Dataset is Phase 2, and VN-MangoLeaf is Phase 3.

All `mean ± SD` values use four seeds: `42`, `123`, `2024`, and `3333`; SD is sample SD. Checkpoints are selected by validation loss, then metrics are read from held-out test results. The remote source was `/home/admincsc/workspace/binh/reproduce_mm` on SSH host `asus`.

| Version | Display name | Input / window | Mixer / ECA / GRN / SwiGLU / inter-window | Test accuracy (%) |
|---|---|---|---|---:|
| V0 | Mango Mamba [Original] | 224px / `[8,8,14,7]` | hybrid / on / on / on / off | 99.4375 ± 0.2394 |
| V3 | **Mango Mamba Final** | 320px / `[8,8,5,5]` | hybrid / on / on / on / stage 0 | **99.6875 ± 0.2394** |

## Rebuilt Table 4 — Ablation without LKA cases

The format follows the paper's Table 4, but every value below is re-aggregated from the completed local `training_results.json` files. Pure LKA and LKA-only ablations are excluded. All rows use MangoLeafBD (`3,200 / 400 / 400`), four seeds, 100-epoch protocol, batch size 64, gradient accumulation 4, AdamW `lr=5e-4`, and held-out test evaluation. `Val loss` is the mean of `final_metrics.val_loss`; it is the reported final-epoch loss, not a reconstructed loss at the selected checkpoint.

| Model configuration | Change from V0 baseline | Params (M) | Val loss | Val accuracy (%) | Val F1 (%) | Test accuracy (%) | Test F1 (%) | Δ test accuracy (pp) | Evidence directory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mango Mamba [Original] V0 — Full Hybrid | baseline | 5.810 | 0.0300 ± 0.0111 | 99.5000 ± 0.3536 | 99.5006 ± 0.3536 | 99.4375 ± 0.2394 | 99.4368 ± 0.2400 | 0.0000 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/original_window/` |
| Mango Mamba [Original] V0 — Pure Mamba | `pure_mamba`; ECA/GRN/SwiGLU on | 8.018 | 0.0323 ± 0.0063 | 99.5000 ± 0.2041 | 99.5005 ± 0.2032 | 99.3125 ± 0.8985 | 99.3118 ± 0.9012 | -0.1250 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/mamba_features/` |
| Mango Mamba [Original] V0 — Hybrid minimal | ECA/GRN/SwiGLU off | 4.411 | 0.0196 ± 0.0026 | 99.8750 ± 0.1443 | 99.8750 ± 0.1444 | 99.2500 ± 0.2041 | 99.2499 ± 0.2042 | -0.1875 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/hybrid_minimal/` |
| Mango Mamba [Original] V0 — No ECA | ECA off | 5.810 | 0.0241 ± 0.0045 | 99.5000 ± 0.3536 | 99.4986 ± 0.3561 | 98.8750 ± 0.5204 | 98.8755 ± 0.5204 | -0.5625 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/no_eca/` |
| Mango Mamba [Original] V0 — No GRN | GRN off | 5.808 | 0.0295 ± 0.0076 | 99.4375 ± 0.3750 | 99.4393 ± 0.3722 | 99.3125 ± 0.3146 | 99.3104 ± 0.3158 | -0.1250 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/no_grn/` |
| Mango Mamba [Original] V0 — No SwiGLU | SwiGLU off | 4.413 | 0.0267 ± 0.0057 | 99.5000 ± 0.3536 | 99.4987 ± 0.3548 | 99.3125 ± 0.3750 | 99.3124 ± 0.3751 | -0.1250 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/no_swiglu/` |
| Mango Mamba Final V3 — no-inter control | 320px, `[8,8,5,5]`, inter-window off | 5.810 | 0.0345 ± 0.0046 | 99.1250 ± 0.2500 | 99.1273 ± 0.2484 | 99.3125 ± 0.4270 | 99.3118 ± 0.4272 | -0.1250 | `01_Ablations/Mango Mamba Final V3 320px [8-8-5-5] No-InterWindow Control/` |
| **Mango Mamba Final V3** | 320px, `[8,8,5,5]`, inter-window stage 0 | 5.824 | 0.0268 ± 0.0067 | 99.4375 ± 0.2394 | 99.4387 ± 0.2381 | **99.6875 ± 0.2394** | **99.6868 ± 0.2400** | **+0.2500** | `02_Mango Mamba Final V3 320px [8-8-5-5]/Phase 1 Evidence/Artifacts/Weights - 4 Seeds/` |

The matched V3 inter-window minus no-inter differences are `-0.2500`, `+0.0000`, `+1.0000`, and `+0.7500` percentage points, with mean `+0.3750 ± 0.5951` percentage points. The inter-window result trends higher, but seed variation exceeds the mean gap.

## Rebuilt Table 6 - Paper baselines and current Mango Mamba versions

This table retains every external comparison row from Table 6 in the reference PDF. The former `MangoMamba` and `Our proposed model` rows are replaced with completed local V0/V3 evidence. Paper rows did not report validation loss, so those cells are `Not reported`. Local runs did not measure latency or mobile inference time, so the old paper timing figures are not reused for V0/V3.

| Model | Test accuracy (%) | Dataset | Images | Val loss | Inference / image | Mobile inference | Params | Model size | Split | Source |
|---|---:|---|---:|---:|---|---|---:|---|---|---|
| DeiT [6] | 99.75 | MangoLeafBD | 4,000 | Not reported | 346.87 ms (CPU) | Not provided | 85.8 M | 1.03 GB | 8:1:1 | PDF Table 6 |
| Optimized DeiT [6] | 99.75 | MangoLeafBD | 4,000 | Not reported | 315.78 ms (CPU) | Not provided | 781 K | 88.47 MB | 8:1:1 | PDF Table 6 |
| SwinTransformer [6] | 99.50 | MangoLeafBD | 4,000 | Not reported | 206.02 ms (CPU) | Not provided | 27.58 M | 110.30 MB | 8:1:1 | PDF Table 6 |
| ShuffleNet [6] | 99.50 | MangoLeafBD | 4,000 | Not reported | 40.76 ms (CPU) | Not provided | 5.36 M | 64.72 MB | 8:1:1 | PDF Table 6 |
| MobileNetV2 [6] | 98.79 | MangoLeafBD | 4,000 | Not reported | 42.28 ms (CPU) | Not provided | 2.23 M | 27.19 MB | 8:1:1 | PDF Table 6 |
| EfficientNet [6] | 98.75 | MangoLeafBD | 4,000 | Not reported | 45.01 ms (CPU) | Not provided | 4.01 M | 48.70 MB | 8:1:1 | PDF Table 6 |
| DenseNet121 [6] | 96.75 | MangoLeafBD | 4,000 | Not reported | 119.34 ms (CPU) | Not provided | 6.96 M | 84.47 MB | 8:1:1 | PDF Table 6 |
| SqueezeNet [6] | 96.50 | MangoLeafBD | 4,000 | Not reported | 20.80 ms (CPU) | Not provided | 726.6 K | 8.78 MB | 8:1:1 | PDF Table 6 |
| ViX-MangoEFormer [9] | 99.43 | MangoLeafBD | 4,000 | Not reported | 120 ms (GPU) | Not provided | Not reported | Not reported | 9:1 | PDF Table 6 |
| ViX-MangoEFormer [9] | 98.02 | Mango Leaf Disease | 12,730 | Not reported | 120 ms (GPU) | Not provided | Not reported | Not reported | 9:1 | PDF Table 6 |
| ViX-MangoEFormer [9] | 98.63 | MLD24 | 6,400 | Not reported | 120 ms (GPU) | Not reported | Not reported | Not reported | 9:1 | PDF Table 6 |
| ViX-MangoEFormer [9] | 99.78 | Combined | 25,530 | Not reported | 120 ms (GPU) | Not reported | Not reported | Not reported | 9:1 | PDF Table 6 |
| Mango Mamba [Original] V0 - Full Hybrid | 99.4375 +/- 0.2394 | MangoLeafBD | 4,000 | 0.0300 +/- 0.0111 | Not measured locally | Not measured locally | 5.810 M | 22.165 MB | 8:1:1 | `01_Ablations/Mango Mamba [Original] V0 224px [8-8-14-7]/original_window/` |
| Mango Mamba Final V3 - no-inter control | 99.3125 +/- 0.4270 | MangoLeafBD | 4,000 | 0.0345 +/- 0.0046 | Not measured locally | Not measured locally | 5.810 M | 22.165 MB | 8:1:1 | `01_Ablations/Mango Mamba Final V3 320px [8-8-5-5] No-InterWindow Control/` |
| **Mango Mamba Final V3 - Phase 1** | **99.6875 +/- 0.2394** | MangoLeafBD | 4,000 | **0.0268 +/- 0.0067** | Not measured locally | Not measured locally | 5.824 M | 22.218 MB | 8:1:1 | `02_Mango Mamba Final V3 320px [8-8-5-5]/Phase 1 Evidence/Artifacts/Weights - 4 Seeds/` |
| Mango Mamba Final V3 - direct-to-VN control | 98.1071 +/- 0.5517 | VN-MangoLeaf | 7,000 | 0.0718 +/- 0.0114 | Not measured locally | Not measured locally | 5.824 M | 22.217 MB | 8:1:1 | `03_Curriculum - Mango Mamba Final V3 320px [8-8-5-5]/Follow-up Results/Direct to VN-MangoLeaf Control/` |
| Mango Mamba Final V3 - Phase 2 | 99.4813 +/- 0.0657 | Combined | 40,000 | 0.0159 +/- 0.0021 | Not measured locally | Not measured locally | 5.824 M | 22.218 MB | 8:1:1 | `03_Curriculum - Mango Mamba Final V3 320px [8-8-5-5]/Follow-up Results/Three-Phase Curriculum/` |
| Mango Mamba Final V3 - Phase 3 | 98.8929 +/- 0.2945 | VN-MangoLeaf | 7,000 | 0.0583 +/- 0.0173 | Not measured locally | Not measured locally | 5.824 M | 22.217 MB | 8:1:1 | `03_Curriculum - Mango Mamba Final V3 320px [8-8-5-5]/Follow-up Results/Three-Phase Curriculum/` |

External rows are paper-reported results, whereas V0/V3 rows are four-seed local aggregates. They are useful context but not a fresh, controlled head-to-head benchmark. On the common VN-MangoLeaf target, V3 three-phase curriculum improves test accuracy by `+0.7857` percentage points over the V3 direct-to-VN control (`98.8929%` versus `98.1071%`).

## Artefact integrity and lineage

The completed `asus` import matches local exactly: 346 files and `7,188,412,717` bytes. It contains 334 result files and 12 logs. Every completed run cited above has `training_results.json`, `args.yaml`, `model_best.pth.tar`, and `last.pth.tar`.

Two interrupted pre-resume artefacts are retained but excluded from all aggregates:

- `01_Ablations/Mango Mamba Final V3 320px [8-8-5-5] No-InterWindow Control/h3_no_inter_mangoleafbd_seed_3333-20260911-075411/`
- `03_Curriculum - Mango Mamba Final V3 320px [8-8-5-5]/Follow-up Results/Three-Phase Curriculum/seed_2024/combined/h3_combined_seed_2024-20260911-075411/`

The V2 and V3 trainer source hashes are:

- `train_mango_mamba.py`: `9cf81d001a866765c99afebd39c58f81b485fb2538ef03ff9dca5b30d4243fdf`
- `models/mango_mamba.py`: `1155fb8058c2a5fde68af7b1fe8915d64c973d9a0555ebb8cf6ba5e417d721af`

## Workspace map

```text
01_Ablations/
02_Mango Mamba Final V3 320px [8-8-5-5]/
03_Curriculum - Mango Mamba Final V3 320px [8-8-5-5]/
04_Code Scripts and Logs/
05_Reference/
MANGO_MAMBA_FINAL_REPORT.md
```

The duplicated prior transfer and superseded documents remain isolated under `99_Archive/` pending local deletion. They are not used as evidence. The report above is the sole Markdown file and contains the retained statistics, evidence paths, and provenance.
