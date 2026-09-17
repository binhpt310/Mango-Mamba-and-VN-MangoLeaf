# Mango Mamba

> **Lightweight Visual State Space Model for Accurate and Explainable Mango Leaf Disease Classification**

---

## 1. Overview & Canonical Architecture

**Mango Mamba** is a lightweight visual state space architecture designed specifically for high-accuracy, low-latency plant disease diagnosis on agricultural edge and mobile devices. 

The canonical model architecture is implemented in:
👉 [`mango_mamba_on_host_asus/04_Code Scripts and Logs/Source Snapshot - Mango Mamba Final V3/mango_mamba.py`](file:///d:/Mango-Mamba-backup/Mango-Mamba-v3/mango_mamba_on_host_asus/04_Code%20Scripts%20and%20Logs/Source%20Snapshot%20-%20Mango%20Mamba%20Final%20V3/mango_mamba.py)  
*(SHA-256: `49a0d363cff65b642ad5323dc638fc6075964de6ca114c78b39915e18a5c5661`)*

### Native Architecture Specifications
- **Input Resolution:** `320 x 320`
- **Window Partition Schedule:** `[8, 8, 5, 5]`
- **Mixer Hierarchy:** Hybrid (Multi-Scale 2D Mamba + Large Kernel Attention)
- **Feature Enhancements:** Efficient Channel Attention (ECA), Global Response Normalization (GRN), and SwiGLU MLP
- **Cross-Window Modeling:** Integrated bidirectional state-space scan for global spatial context
- **Regularization:** Stochastic depth `drop_path_rate = 0.01`
- **Model Capacity:** **5.824 M** parameters (22.218 MB serialized weights)

---

## 2. Window Schedule Design: NVIDIA MambaVision `[8, 8, 14, 7]` vs. Mango Mamba `[8, 8, 5, 5]`

### Why MambaVision's `[8, 8, 14, 7]` Was Replaced
In NVIDIA's original MambaVision architecture (Hatamizadeh et al.), the window schedule `[8, 8, 14, 7]` was designed for standard $224 \times 224$ images, where hierarchical feature maps downsample as:
$$\text{Stage 1: } 56 \times 56 \quad \to \quad \text{Stage 2: } 28 \times 28 \quad \to \quad \text{Stage 3: } 14 \times 14 \quad \to \quad \text{Stage 4: } 7 \times 7$$
On $224 \times 224$, the $14 \times 14$ and $7 \times 7$ windows in the later stages match the full spatial map without padding.

However, fine-grained plant pathology requires higher input resolution ($320 \times 320$) to resolve subtle lesion textures and fungal patches. At $320 \times 320$, the feature map resolutions become:
$$\text{Stage 1: } 80 \times 80 \quad \to \quad \text{Stage 2: } 40 \times 40 \quad \to \quad \text{Stage 3: } 20 \times 20 \quad \to \quad \text{Stage 4: } 10 \times 10$$

Applying `[8, 8, 14, 7]` to these dimensions causes severe division mismatches:
- In Stage 3, $20$ is not divisible by $14$ ($20 \pmod{14} = 6$), forcing $8$ zero-padded border pixels.
- In Stage 4, $10$ is not divisible by $7$ ($10 \pmod{7} = 3$), forcing $4$ zero-padded border pixels.

These synthetic zero-padded tokens corrupt recurrent state transitions in Mamba's Selective Scan module and dilute attention maps with non-informative padding.

### The Exact-Dividing `[8, 8, 5, 5]` Solution
Mango Mamba designs the `[8, 8, 5, 5]` window partition schedule, which divides every feature map cleanly with **zero spatial padding**:
- **Stage 1 ($80 \times 80$):** $80 / 8 = 10 \implies 10 \times 10 = 100$ windows of size $8 \times 8$
- **Stage 2 ($40 \times 40$):** $40 / 8 = 5 \implies 5 \times 5 = 25$ windows of size $8 \times 8$
- **Stage 3 ($20 \times 20$):** $20 / 5 = 4 \implies 4 \times 4 = 16$ windows of size $5 \times 5$
- **Stage 4 ($10 \times 10$):** $10 / 5 = 2 \implies 2 \times 2 = 4$ windows of size $5 \times 5$

### Empirical Proof
- **Accuracy:** Across four independent seeds, Mango Mamba (`[8, 8, 5, 5]`) achieves **99.6875% ± 0.2394%** mean test accuracy, outperforming the MambaVision `[8, 8, 14, 7]` setting (**99.4375% ± 0.2394%**) by **+0.2500 percentage points**.
- **Flawless Evaluation:** On `seed 3333`, the `[8, 8, 5, 5]` configuration reaches **100.0000% test accuracy** (0 errors across all 400 test images, macro-F1 = 1.0).
- **Latency:** Clean memory layouts eliminate padding/unpadding steps, achieving **4.566 ms** forward latency on NVIDIA GB10 GPU.

---

## 3. Benchmark Results

| Model / Configuration | Dataset | Classes | Test Accuracy | Precision / Recall / F1 | Latency |
|---|---|---:|---:|---:|---|
| **Mango Mamba (Phase 1 Baseline)** | MangoLeafBD | 8 | **99.6875 ± 0.2394%** | **99.6868 ± 0.2400%** | **4.566 ms** (NVIDIA GB10 GPU) |
| **Best Seed (Seed 3333)** | MangoLeafBD | 8 | **100.0000%** (0 errors / 400 images) | **1.0000 / 1.0000 / 1.0000** | **4.566 ms** (NVIDIA GB10 GPU) |
| **Three-Phase Curriculum (Phase 3)** | VN-MangoLeaf | 7 | **99.286%** (695 / 700 images) | **99.280% macro-F1** | **612.80 ms** (Redmi Note 11S mobile) |
| Mango Mamba (Ablation: 224px, `[8,8,14,7]`) | MangoLeafBD | 8 | 99.4375 ± 0.2394% | 99.4368 ± 0.2400% | — |

---

## 4. Quickstart: Instantiating & Loading Model

Because native defaults in `mango_mamba.py` match the optimal training configuration, the model can be loaded directly without extra override arguments:

```python
import torch
from models.mango_mamba import mango_mamba_tiny

# 1. Instantiate the canonical model directly:
model = mango_mamba_tiny(num_classes=8)

# 2. Strict loading from the best checkpoint (seed 3333):
checkpoint_path = "mango_mamba_on_host_asus/02_Mango Mamba Final V3 320px [8-8-5-5]/Phase 1 Evidence/Artifacts/Weights - 4 Seeds/phase2_window_8_8_5_5_seed_3333-20260910-013944/model_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=True)
model.eval()

# 3. Forward pass at 320x320:
x = torch.randn(1, 3, 320, 320)
with torch.no_grad():
    output = model(x)
logits = output["logits"]       # shape: [1, 8]
features = output["features"]   # shape: [1, 384]
```

---

## 5. Repository Structure

```text
Mango-Mamba-v3/
├── README.md                                          # Canonical documentation
├── docs/                                              # Architectural diagrams and figures
│   └── figures/mango_mamba_architecture.png
├── for_android_app_only/                              # Android CAM application & deployment evidence
│   ├── ANDROID_MODEL_PROVENANCE.md
│   ├── android/                                       # Gradle Android app module
│   └── artifacts/                                     # On-device benchmarks (Redmi Note 11S)
└── mango_mamba_on_host_asus/                          # Evidence snapshot from training host
    ├── MANGO_MAMBA_FINAL_REPORT.md                    # Detailed experimental report
    ├── GITHUB_EVIDENCE_INDEX.md                       # Publication evidence index
    ├── MANGO_MAMBA_COMPLETION_PLAN.md                 # Protocol and execution specifications
    ├── 01_Ablations/                                  # Component ablations (No-ECA, No-GRN, etc.)
    ├── 02_Mango Mamba Final V3 320px [8-8-5-5]/       # Phase 1 MangoLeafBD weights and metrics
    ├── 03_Curriculum - Mango Mamba Final V3 320px.../ # Phase 2 & Phase 3 curriculum checkpoints
    ├── 04_Code Scripts and Logs/                      # Canonical model source snapshot & runner scripts
    │   └── Source Snapshot - Mango Mamba Final V3/
    │       └── mango_mamba.py                         # <--- CANONICAL MODEL FILE
    └── 06_Host Synced Evidence/                       # Single-checkpoint latency & error analysis
```

---

## 6. Mobile Deployment & Explainability

The model supports Class Activation Mapping (CAM) without retraining. The final normalized $10 \times 10$ feature map is combined with linear classifier weights:
$$\text{CAM}_{c}(y, x) = \text{ReLU}\left(\sum_{k=1}^{384} \mathbf{F}_{k}(y, x) \cdot \mathbf{W}_{c, k}\right)$$

An Android demo application running ONNX Runtime on mobile is packaged under `for_android_app_only/`.
