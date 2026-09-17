# Mango Mamba Evidence Publication Index

This index identifies the immutable evidence behind Mango Mamba benchmarks, error analysis, and edge deployment artifacts. The complete evidence store is retained locally at `D:/mango_mamba_on_host_asus/` and on the experiment host at `/home/admincsc/workspace/binh/reproduce_mm/results/table6_completion_20260913_051241/`.

The canonical model architecture is defined with native defaults in [`04_Code Scripts and Logs/Source Snapshot - Mango Mamba Final V3/mango_mamba.py`](file:///d:/Mango-Mamba-backup/Mango-Mamba-v3/mango_mamba_on_host_asus/04_Code%20Scripts%20and%20Logs/Source%20Snapshot%20-%20Mango%20Mamba%20Final%20V3/mango_mamba.py) with SHA-256 `49a0d363cff65b642ad5323dc638fc6075964de6ca114c78b39915e18a5c5661`.

## Measured Mango Mamba Phase-1 speed

- Configuration: Mango Mamba, 320px, `[8,8,5,5]`, FP32, batch 1.
- Device/runtime: NVIDIA GB10, PyTorch 2.13.0, CUDA 13.0.
- Result: 4.566 ms mean, 4.545 ms median, 4.594 ms p95 over 500 CUDA-event samples after 100 warm-ups.
- Selected checkpoint SHA-256: `973d6d4f2c0f50aa1d27e9990b7cfecd41e31adddfe55442a3961427c757b1d1` (Held-out MangoLeafBD test accuracy: **100.0%**, 0 errors / 400 test images).

## Phase-3 VN-MangoLeaf held-out error analysis

- Configuration: Mango Mamba, 320px, `[8,8,5,5]`; curriculum seed 42.
- Result: 695/700 correct, 99.286% accuracy, 99.280% macro-F1, five Anthracnose errors.
- Selected checkpoint SHA-256: `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae`.
- Published outputs: summary, per-class metrics, misclassification index, confusion matrix, scripts, source arguments, log, and the five selected source-image/CAM panels. The full raw dataset remains only in the local/host evidence store.

## Android CAM export

- Artifact: `mango_mamba_v3_phase3_320_cam.onnx` (24.12 MiB), static NCHW `[1,3,320,320]`.
- Outputs: `logits [1,7]` and `feature_map [1,384,10,10]`.
- ONNX checker passed. Maximum PyTorch/ONNX Runtime absolute difference: logits `1.1920928955078125e-06`, feature map `6.4373016357421875e-06` (acceptance tolerance `0.001`).
- ONNX SHA-256: `f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4`.
- Manifest SHA-256: `68900e8ba5812d5940e2b8b13a6f2e315992240ff9cb1b13169a373516bc181a`.

## Publication boundary

Do not publish raw datasets, unapproved user/camera images, or multi-gigabyte checkpoint archives without an explicit repository/LFS decision. The report and completion plan link the full local evidence locations and hashes; the Git bundle is a reproducible, reviewable subset rather than a replacement for the evidence store.
