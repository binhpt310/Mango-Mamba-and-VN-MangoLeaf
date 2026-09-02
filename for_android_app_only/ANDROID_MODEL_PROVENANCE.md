# VN-MangoLeaf 7-class Android model provenance

The app source is `android/`. `android/app/src/main/java/com/mangomamba/benchmark/MainActivity.kt` loads exactly one model: `android/app/src/main/assets/mango_mamba_7class.onnx`.

| Artifact | SHA-256 |
|---|---|
| Phase 3 checkpoint | `3bd36d4d85521f4bdd7641be5d79382dffea9ac29148e967a01273a1166f3426` |
| Android ONNX asset | `e2e38ad2449a5d58134de7681985af189746bec1696ff499a350efee618d2166` |
| Rebuilt debug APK | `643aeec654565322fd2a5c26c0c2928fccc0deff37ab712e6ce03d83e07bbe4a` |

## Training origin

The checkpoint and ONNX were copied from `asus`:

`/home/admincsc/workspace/binh/Mango-Mamba/outputs/output_phase3/vnmangoleaf_phase3-20260705-210739/`

`provenance/metadata/args.yaml` records the Phase 3 configuration: `data_dir: vn_mangoleaf`, `num_classes: 7`, seed 42, and initialization from the Phase 2 Combined checkpoint. The Phase 2 parent checkpoint SHA-256 is `1d6bb25c541984d4610d1f55f93f2652c2446dac0f5c37ca66e23aa8dd2f388f`.

`provenance/metadata/training_results.json` records best validation accuracy 97.5714% and `final_metrics.test_acc` 98.7124% at best epoch 73. Its separate `test_metrics.accuracy` field is 97.7110%, so this bundle does not present the paper result as an independently reproduced single metric.

`provenance/paper_code/reproduce_table5.ipynb` is the archived paper curriculum code. It documents the intended MangoLeafBD → Combined → VN-MangoLeaf sequence; the dataset and predecessor checkpoints are deliberately not included here.

## ONNX and Android contract

`provenance/scripts/export_onnx.py` and `provenance/model_source/mango_mamba.py` contain the archived converter and MangoMamba architecture. The ONNX file and source checkpoint are co-located in the Phase 3 output on `asus`; an original export command log was not preserved.

- Input: `image`, `float32[1,3,224,224]`
- Output: `logits`, `float32[1,7]`
- Labels: Anthracnose, Dieback, Gall Midge, Healthy, Powdery Mildew, Red Rust, Sooty Mold
- Android preprocessing: short side 256, center crop 224, ImageNet normalization

The debug APK contains only `assets/mango_mamba_7class.onnx`; its embedded ONNX SHA-256 matches the asset above.

## Redmi Note 11S smoke test

The final APK was installed on 2026-09-02. One VN-MangoLeaf sample from six of seven classes was classified correctly; the initial Anthracnose sample was predicted as Sooty Mold. A five-image Anthracnose follow-up classified four correctly. Latency across these 12 image inferences was 268–856 ms. This is a device smoke test, not a full test-set evaluation.

This bundle contains no images, datasets, intermediate checkpoints, Grad-CAM outputs, or 9-class assets.
