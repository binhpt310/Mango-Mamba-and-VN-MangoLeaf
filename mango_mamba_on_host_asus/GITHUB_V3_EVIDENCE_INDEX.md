# Final V3 evidence publication index

This index identifies the immutable evidence behind the Final V3 updates made on 2026-09-13. The complete evidence store is retained locally at `D:/mango_mamba_on_host_asus/` and on the experiment host at `/home/admincsc/workspace/binh/reproduce_mm/results/table6_completion_20260913_051241/`.

## Measured Final V3 Phase-1 speed

- Configuration: 320px, `[8,8,5,5]`, stage-0 inter-window Mamba, FP32, batch 1.
- Device/runtime: NVIDIA GB10, PyTorch 2.13.0, CUDA 13.0.
- Result: 4.566 ms mean, 4.545 ms median, 4.594 ms p95 over 500 CUDA-event samples after 100 warm-ups.
- Selected checkpoint SHA-256: `973d6d4f2c0f50aa1d27e9990b7cfecd41e31adddfe55442a3961427c757b1d1`.

## Phase-3 VN-MangoLeaf held-out error analysis

- Configuration: Final V3 320px, `[8,8,5,5]`, stage-0 inter-window Mamba; curriculum seed 42.
- Result: 695/700 correct, 99.286% accuracy, 99.280% macro-F1, five Anthracnose errors.
- Selected checkpoint SHA-256: `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae`.
- Git-safe outputs: summary, per-class metrics, misclassification index, confusion matrix, scripts, source arguments, and logs. The five raw source test images and image-containing CAM panels remain in the local/host evidence store to avoid republishing dataset images.

## Android CAM export

- Artifact: `mango_mamba_v3_phase3_320_cam.onnx` (24.12 MiB), static NCHW `[1,3,320,320]`.
- Outputs: `logits [1,7]` and `feature_map [1,384,10,10]`.
- ONNX checker passed. Maximum PyTorch/ONNX Runtime absolute difference: logits `1.1920928955078125e-06`, feature map `6.4373016357421875e-06` (acceptance tolerance `0.001`).
- ONNX SHA-256: `f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4`.
- Manifest SHA-256: `68900e8ba5812d5940e2b8b13a6f2e315992240ff9cb1b13169a373516bc181a`.

## Publication boundary

Do not publish raw datasets, unapproved user/camera images, or multi-gigabyte checkpoint archives without an explicit repository/LFS decision. The report and completion plan link the full local evidence locations and hashes; the Git bundle is a reproducible, reviewable subset rather than a replacement for the evidence store.
