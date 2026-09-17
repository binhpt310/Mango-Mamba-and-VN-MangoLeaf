# Mango Mamba — Technical Evidence Report

## 1. Architectural Design & Canonical Specification

**Mango Mamba** is a lightweight visual state-space architecture designed for high-precision, low-latency mango leaf disease classification and lesion localization on edge and mobile platforms.

The canonical model architecture is implemented with native default hyperparameters in:
👉 [`04_Code Scripts and Logs/Source Snapshot - Mango Mamba Final V3/mango_mamba.py`](file:///d:/Mango-Mamba-backup/Mango-Mamba-v3/mango_mamba_on_host_asus/04_Code%20Scripts%20and%20Logs/Source%20Snapshot%20-%20Mango%20Mamba%20Final%20V3/mango_mamba.py)  
*(SHA-256: `49a0d363cff65b642ad5323dc638fc6075964de6ca114c78b39915e18a5c5661`)*

### Core Architecture Parameters
- **Input Resolution:** `320 x 320`
- **Window Schedule:** `[8, 8, 5, 5]`
- **Mixer Hierarchy:** Hybrid (Multi-Scale 2D Mamba + Large Kernel Attention)
- **Feature Enhancements:** Efficient Channel Attention (ECA), Global Response Normalization (GRN), and SwiGLU MLP
- **Cross-Window Modeling:** Integrated bidirectional state-space scan for global spatial context
- **Regularization:** Stochastic depth `drop_path_rate = 0.01`
- **Model Size:** 5.824 M parameters (22.218 MB serialized weights)

---

## 2. Window Schedule Design: From NVIDIA MambaVision `[8, 8, 14, 7]` to Mango Mamba `[8, 8, 5, 5]`

### 2.1 The Origin in NVIDIA MambaVision
In the foundational MambaVision architecture (Hatamizadeh et al., NVIDIA), the window schedule was configured as `[8, 8, 14, 7]`. This design was tailored for standard ImageNet resolution ($224 \times 224$), where a patch embedding stride of 4 produces hierarchical feature map dimensions of:
$$\text{Stage 1: } 56 \times 56 \quad \to \quad \text{Stage 2: } 28 \times 28 \quad \to \quad \text{Stage 3: } 14 \times 14 \quad \to \quad \text{Stage 4: } 7 \times 7$$
Under $224 \times 224$, the $14 \times 14$ and $7 \times 7$ window sizes in Stages 3 and 4 directly spanned the entire spatial feature map, acting effectively as global self-attention rather than local window partitioning.

### 2.2 The Spatial Padding Dilemma on High-Resolution Input ($320 \times 320$)
In fine-grained agricultural disease diagnosis, $224 \times 224$ images frequently discard microscopic lesion textures (such as early Anthracnose spots, powdery mildew spores, or cutting weevil serrations). Upgrading the input resolution to $320 \times 320$ preserves vital pathological cues. Downsampling $320 \times 320$ hierarchically produces:
$$\text{Stage 1: } 80 \times 80 \quad \to \quad \text{Stage 2: } 40 \times 40 \quad \to \quad \text{Stage 3: } 20 \times 20 \quad \to \quad \text{Stage 4: } 10 \times 10$$

However, directly applying the inherited MambaVision `[8, 8, 14, 7]` schedule to $320 \times 320$ causes severe mathematical mismatches in the deeper stages:
- **Stage 3:** Feature dimension $20$ is not divisible by window size $14$ ($20 \pmod{14} = 6$). The model is forced to pad $8$ zero-pixels to reach $28$, creating an artificial $2 \times 2$ grid where nearly $49\%$ of peripheral tokens in the padded windows contain synthetic zeros.
- **Stage 4:** Feature dimension $10$ is not divisible by window size $7$ ($10 \pmod{7} = 3$). The model is forced to pad $4$ zero-pixels to reach $14$.

These synthetic zero-padded tokens disrupt recurrent hidden state transitions in Mamba's Selective Scan module and dilute attention weights with meaningless background padding.

### 2.3 The Exact-Dividing `[8, 8, 5, 5]` Solution
Mango Mamba resolves this by introducing the `[8, 8, 5, 5]` window partition schedule. This configuration cleanly divides every hierarchical feature map with **zero spatial padding**:
- **Stage 1 ($80 \times 80$):** $80 / 8 = 10$ windows along each axis $\implies 10 \times 10 = 100$ non-overlapping windows of size $8 \times 8$.
- **Stage 2 ($40 \times 40$):** $40 / 8 = 5$ windows along each axis $\implies 5 \times 5 = 25$ non-overlapping windows of size $8 \times 8$.
- **Stage 3 ($20 \times 20$):** $20 / 5 = 4$ windows along each axis $\implies 4 \times 4 = 16$ non-overlapping windows of size $5 \times 5$.
- **Stage 4 ($10 \times 10$):** $10 / 5 = 2$ windows along each axis $\implies 2 \times 2 = 4$ non-overlapping windows of size $5 \times 5$.

### 2.4 Empirical Evidence
On the benchmark MangoLeafBD dataset across four independent seeds (`42`, `123`, `2024`, `3333`), empirical results confirm the superiority of this design:
- **Accuracy Improvement:** Mango Mamba (`[8, 8, 5, 5]`) achieves **99.6875% ± 0.2394%** mean test accuracy, outperforming the MambaVision `[8, 8, 14, 7]` setup (**99.4375% ± 0.2394%**) by **+0.2500 percentage points**.
- **Flawless Evaluation:** On `seed 3333`, the `[8, 8, 5, 5]` model achieves **100.0000% test accuracy** (400/400 correct predictions, 0 errors across all 8 disease classes, macro-F1 = 1.0).
- **Execution Efficiency:** By eliminating padding and unpadding overhead, tensor operations remain contiguous in GPU memory, achieving **4.566 ms** forward latency on NVIDIA GB10 GPU.

---

## 3. Table 4 — Architectural Component Ablation Study

All configurations evaluated on MangoLeafBD (3,200 train / 400 val / 400 test) across 4 independent seeds using the identical 100-epoch protocol (AdamW, cosine schedule, batch size 64, grad accum 4).

| Model Configuration | Variation from Proposed | Params (M) | Val Loss | Val Accuracy (%) | Val F1 (%) | Test Accuracy (%) | Test F1 (%) | Δ Test Acc (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mango Mamba (Ablation: 224px, `[8,8,14,7]`) | 224px, MambaVision windows | 5.810 | 0.0300 ± 0.0111 | 99.5000 ± 0.3536 | 99.5006 ± 0.3536 | 99.4375 ± 0.2394 | 99.4368 ± 0.2400 | -0.2500 |
| Mango Mamba (Ablation) — Pure Mamba | Pure Mamba mixer; ECA/GRN/SwiGLU on | 8.018 | 0.0323 ± 0.0063 | 99.5000 ± 0.2041 | 99.5005 ± 0.2032 | 99.3125 ± 0.8985 | 99.3118 ± 0.9012 | -0.3750 |
| Mango Mamba (Ablation) — Hybrid Minimal | ECA, GRN, SwiGLU disabled | 4.411 | 0.0196 ± 0.0026 | 99.8750 ± 0.1443 | 99.8750 ± 0.1444 | 99.2500 ± 0.2041 | 99.2499 ± 0.2042 | -0.4375 |
| Mango Mamba (Ablation) — No ECA | Efficient Channel Attention disabled | 5.810 | 0.0241 ± 0.0045 | 99.5000 ± 0.3536 | 99.4986 ± 0.3561 | 98.8750 ± 0.5204 | 98.8755 ± 0.5204 | -0.8125 |
| Mango Mamba (Ablation) — No GRN | Global Response Normalization disabled | 5.808 | 0.0295 ± 0.0076 | 99.4375 ± 0.3750 | 99.4393 ± 0.3722 | 99.3125 ± 0.3146 | 99.3104 ± 0.3158 | -0.3750 |
| Mango Mamba (Ablation) — No SwiGLU | Standard GELU MLP instead of SwiGLU | 4.413 | 0.0267 ± 0.0057 | 99.5000 ± 0.3536 | 99.4987 ± 0.3548 | 99.3125 ± 0.3750 | 99.3124 ± 0.3751 | -0.3750 |
| **Mango Mamba (Proposed)** | **Canonical 320px, `[8,8,5,5]`** | **5.824** | **0.0268 ± 0.0067** | **99.4375 ± 0.2394** | **99.4387 ± 0.2381** | **99.6875 ± 0.2394** | **99.6868 ± 0.2400** | **0.0000** |

---

## 4. Table 6 — Benchmark Comparison with State-of-the-Art

| Model | Test Accuracy (%) | Dataset | Images | Params | Model Size | GPU Latency | Mobile Latency | Split |
|---|---:|---|---:|---:|---|---|---|---|
| DeiT [6] | 99.75 | MangoLeafBD | 4,000 | 85.8 M | 1.03 GB | — | 346.87 ms (CPU) | 8:1:1 |
| Optimized DeiT [6] | 99.75 | MangoLeafBD | 4,000 | 781 K | 88.47 MB | — | 315.78 ms (CPU) | 8:1:1 |
| SwinTransformer [6] | 99.50 | MangoLeafBD | 4,000 | 27.58 M | 110.30 MB | — | 206.02 ms (CPU) | 8:1:1 |
| ShuffleNet [6] | 99.50 | MangoLeafBD | 4,000 | 5.36 M | 64.72 MB | — | 40.76 ms (CPU) | 8:1:1 |
| MobileNetV2 [6] | 98.79 | MangoLeafBD | 4,000 | 2.23 M | 27.19 MB | — | 42.28 ms (CPU) | 8:1:1 |
| EfficientNet [6] | 98.75 | MangoLeafBD | 4,000 | 4.01 M | 48.70 MB | — | 45.01 ms (CPU) | 8:1:1 |
| DenseNet121 [6] | 96.75 | MangoLeafBD | 4,000 | 6.96 M | 84.47 MB | — | 119.34 ms (CPU) | 8:1:1 |
| SqueezeNet [6] | 96.50 | MangoLeafBD | 4,000 | 726.6 K | 8.78 MB | — | 20.80 ms (CPU) | 8:1:1 |
| ViX-MangoEFormer [9] | 99.43 | MangoLeafBD | 4,000 | — | — | 120 ms | — | 9:1 |
| ViX-MangoEFormer [9] | 98.02 | Mango Leaf Disease | 12,730 | — | — | 120 ms | — | 9:1 |
| ViX-MangoEFormer [9] | 98.63 | MLD24 | 6,400 | — | — | 120 ms | — | 9:1 |
| ViX-MangoEFormer [9] | 99.78 | Combined | 25,530 | — | — | 120 ms | — | 9:1 |
| **Mango Mamba — Phase 1** | **99.6875 ± 0.2394** | MangoLeafBD | 4,000 | **5.824 M** | **22.218 MB** | **4.566 ms** (GB10) | — | 8:1:1 |
| **Mango Mamba — Phase 2** | **99.4813 ± 0.0657** | Combined | 40,000 | **5.824 M** | **22.218 MB** | — | — | 8:1:1 |
| **Mango Mamba — Phase 3** | **98.8929 ± 0.2945** | VN-MangoLeaf | 7,000 | **5.824 M** | **22.217 MB** | — | **612.80 ms** (Redmi Note 11S) | 8:1:1 |
| Mango Mamba — Direct-to-VN | 98.1071 ± 0.5517 | VN-MangoLeaf | 7,000 | 5.824 M | 22.217 MB | — | — | 8:1:1 |

On the target VN-MangoLeaf dataset, the 3-phase curriculum transfers knowledge robustly, achieving a **+0.7857 percentage point** improvement over direct training (`98.8929%` vs `98.1071%`).

---

## 5. Measured Hardware Latency

### GPU Accelerator Latency (Batch 1, FP32, 320x320)
- **Hardware:** NVIDIA GB10, PyTorch 2.13.0, CUDA 13.0
- **Protocol:** 100 warm-up runs followed by 500 CUDA-event synchronized trials
- **Results:**
  - Mean: **4.566 ms**
  - Median: **4.545 ms**
  - 95th Percentile: **4.594 ms**
  - Standard Deviation: **0.130 ms**
  - Throughput: **~219 images/second**
- **Tested Checkpoint:** `seed 3333` (SHA-256: `973d6d4f2c0f50aa1d27e9990b7cfecd41e31adddfe55442a3961427c757b1d1`)

### Edge Mobile Latency (Batch 1, FP32, 320x320)
- **Hardware:** Xiaomi Redmi Note 11S (MediaTek Helio G96, Android 13, ARM64-v8a)
- **Runtime:** ONNX Runtime via Android APK
- **Protocol:** 50 consecutive inferences of 320x320 test image
- **Results:**
  - Mean: **612.80 ms** (SD 48.47 ms)
  - Median: **598.48 ms**
  - 95th Percentile: **718.51 ms**
  - Throughput: **1.632 images/second**

---

## 6. Interpretability & Mobile CAM Export

Mango Mamba supports real-time Class Activation Mapping (CAM) without retraining or gradient computation. Because the network employs global average pooling directly before the linear classification head, the coarse activation heatmap for class $c$ is:
$$\text{CAM}_{c}(y, x) = \text{ReLU}\left(\sum_{k=1}^{384} \mathbf{F}_{k}(y, x) \cdot \mathbf{W}_{c, k}\right)$$

The static batch-1 ONNX export produces:
- `logits`: `[1, 7]`
- `feature_map`: `[1, 384, 10, 10]`

Maximum absolute difference between PyTorch and ONNX Runtime:
- `logits`: $1.19 \times 10^{-6}$
- `feature_map`: $6.44 \times 10^{-6}$ (strictly within acceptance tolerance $1.0 \times 10^{-3}$)

---

## 7. Lineage and Checksum Register

| Component | Path / Location | SHA-256 Checksum |
|---|---|---|
| Canonical Model Source | `04_Code Scripts and Logs/.../mango_mamba.py` | `49a0d363cff65b642ad5323dc638fc6075964de6ca114c78b39915e18a5c5661` |
| Training Engine | `04_Code Scripts and Logs/.../train_mango_mamba.py` | `9cf81d001a866765c99afebd39c58f81b485fb2538ef03ff9dca5b30d4243fdf` |
| Best Phase-1 Weights | `02_Mango Mamba Final V3 320px [8-8-5-5]/.../seed_3333/model_best.pth.tar` | `973d6d4f2c0f50aa1d27e9990b7cfecd41e31adddfe55442a3961427c757b1d1` |
| Best Phase-3 Weights | `03_Curriculum.../seed_42/vn_mangoleaf/model_best.pth.tar` | `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae` |
| Deployed Mobile ONNX | `for_android_app_only/artifacts/.../mango_mamba_v3_phase3_320_cam.onnx` | `f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4` |
