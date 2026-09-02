# VN-MangoLeaf 7-class Android model provenance

## Current deployed model

The Android app loads exactly one canonical asset: `android/app/src/main/assets/mango_mamba_7class.onnx`. Its label file is `mango_mamba_7class_labels.json`; `MainActivity.kt` rejects any label list that is not the exact order below.

| Artifact | SHA-256 | Status |
|---|---|---|
| Android ONNX asset | `e2e38ad2449a5d58134de7681985af189746bec1696ff499a350efee618d2166` | Runtime model |
| Labels asset | `8dc2823c4db6fe1769650990b07a613e30a1e62a8e7472cba65ecf02d5c3cf46` | Runtime label mapping |
| Phase 3 checkpoint | `3bd36d4d85521f4bdd7641be5d79382dffea9ac29148e967a01273a1166f3426` | Retained source-weight evidence |
| Training arguments | `e6569cc01ceb511e7ce7ed7288e55c96521e79153fa80101617e279e29f495b5` | Retained configuration evidence |
| Training results | `a542c69ab51d584cfdd2fca863065cea992e2eefa8e58ffdf875f6a784d73235` | Retained metric evidence |

Supported labels, in required order: `Anthracnose`, `Dieback`, `Gall Midge`, `Healthy`, `Powdery Mildew`, `Red Rust`, `Sooty Mold`.

The app contract is input `image`, `float32[1,3,224,224]`; output `logits`, `float32[1,7]`; resize short side to 256, center crop to 224, then apply ImageNet normalization. The app uses one shared ONNX Runtime session with graph optimization `ALL`.

## Training evidence

`provenance/metadata/args.yaml` records Phase 3 training on `vn_mangoleaf`, 7 classes, seed 42, initialized from the Phase 2 Combined checkpoint. `training_results.json` records best validation accuracy `97.5714%` at epoch 73; its `final_metrics.test_acc` is `98.7124%` while its separate `test_metrics.accuracy` is `97.7110%`. These are archived training results, not a new reproduction claim.

The retained checkpoint, architecture source, and ONNX export script are sufficient to inspect the current model lineage. The paper notebook was removed because it is not needed for Android runtime or current-model validation.

## Redmi Note 11S throughput benchmark

Measured on 2026-09-03 on Xiaomi Redmi Note 11S (`2201117SG`), Android 13 / API 33, package `com.mangomamba.benchmark` version `0.2-mixed`. The installed debug APK SHA-256 was `dda3b386a7ebc7be6d1031ff5214abe3162c3f4f8025569c1ca604bcc4731178`.

| Metric | Result |
|---|---:|
| Images | 498 (83 per each of six disease classes) |
| Correct | 497 / 498 (99.7992%) |
| Average `session.run` latency | 491.88 ms |
| P50 / P95 / P99 latency | 504.80 / 580.68 / 630.10 ms |
| Min / max latency | 307.42 / 651.46 ms |
| Total measured `session.run` time | 244.955 s |
| Average throughput | 2.033 images/s |

The deterministic manifest SHA-256 was `fabcb5c63840797d4065130d4b6b9464c68ed4e07bfa32714f414629270e5cba`; the result CSV SHA-256 was `2ae1e7cfc5178dea7332c6f4f6a5280d8dbe9c383ac6a5a75c216064ad1c6163`. The one error was `Anthracnose_138.jpg`, predicted as Red Rust at 71.6792% confidence.

Latency starts immediately before `OrtSession.run()` and ends when that call returns. It includes neither JPEG decoding, EXIF rotation, resize/crop/tensor construction, activity launch, ADB transfer, nor camera capture. Therefore 2.033 images/s is ONNX Runtime inference throughput on this device, not end-to-end camera throughput. No separate warm-up was excluded.

The raw VN MangoLeaf folder currently contains the six disease directories above. Healthy is not present there; the two available archived VN Healthy probes were separately tested on the same device and both were predicted Healthy (2/2, 383.73 ms and 389.15 ms).

## Replacing the ONNX safely

1. Export a new ONNX that preserves the runtime contract and the exact seven-label order above.
2. Run `scripts/replace_android_7class_model.ps1 -ModelPath <new.onnx> -LabelsPath <labels.json>`.
3. Build the app, install it, and run `scripts/measure_android_device_throughput.ps1`.
4. Record the new ONNX/label hashes, manifest hash, result hash, device details, and benchmark metrics in this file.

The replacement script checks the label order before overwriting the canonical assets. It cannot prove an arbitrary ONNX has the required tensor names/shapes or adequate accuracy; the build and device benchmark are mandatory checks.

## Retained and excluded files

Tracked: the canonical ONNX and labels, source checkpoint, training arguments/results, architecture source, and ONNX export script. They are the evidence needed to identify the current deployed model and inspect its lineage.

Excluded through `.gitignore`: Gradle caches, APK build outputs, copied test images, manifests, screenshots, and device result CSVs. They are reproducible local test outputs; their necessary hashes and summary metrics are recorded above instead.
