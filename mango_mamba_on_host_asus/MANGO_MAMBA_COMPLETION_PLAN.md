# Mango Mamba: Table-Completion, Error-Analysis, and Android Deployment Plan

## 1. Scope and baseline policy

This plan completes the missing local evidence in Table 6 of `MANGO_MAMBA_FINAL_REPORT.md` without overwriting historical results or reusing latency values from the old manuscript.

- **Only active proposed baseline:** **Mango Mamba** — `320x320`, window sizes `[8,8,5,5]`, hybrid mixer, ECA/GRN/SwiGLU enabled (canonically defined in [`04_Code Scripts and Logs/Source Snapshot - Mango Mamba Final V3/mango_mamba.py`](file:///d:/Mango-Mamba-backup/Mango-Mamba-v3/mango_mamba_on_host_asus/04_Code%20Scripts%20and%20Logs/Source%20Snapshot%20-%20Mango%20Mamba%20Final%20V3/mango_mamba.py)).
- **Ablation / comparator rows:** Mango Mamba Ablation (`224x224`, `[8,8,14,7]`) inherited from NVIDIA MambaVision default schedule. Measured as a historical comparator, but not the proposed baseline.
- **Window Schedule Optimization:** The transition from 224px `[8,8,14,7]` (MambaVision reference schedule) to 320px `[8,8,5,5]` cleanly divides the $80 \to 40 \to 20 \to 10$ feature maps with zero spatial padding, outperforming the legacy schedule by +0.25 pp (99.69% vs 99.44%).
- **Do not copy legacy timing.** Table 6 currently says local Mango Mamba timing is not measured. Replace those cells only with timings produced by this protocol, with hardware and runtime stated beside each value.

### Experimental purpose and paper form

The goal is a **new V3 experimental and analysis section**, following the evidence structure of the older `Mango-Mamba and VN-MangoLeaf.pdf`, rather than an attempt to reproduce its unsupported latency claims. The old paper provides a useful form: four-seed ablation statistics (Table 4), training protocol (Table 5), accuracy/efficiency comparison (Table 6), error analysis with explanation maps (Section 3.7/Figure 13), and an offline Android ONNX demonstration (Figure 12).

The new claim to test is an efficiency trade-off, not an accuracy superlative: Final V3 may be comparable rather than best in accuracy while being substantially smaller, faster, or more deployable. It may be written only when the new measurements support it under disclosed conditions. The valid conclusion is therefore conditional:

- report accuracy/F1 with four-seed mean ± standard deviation on the native protocol;
- report parameters, serialized model size, batch-1 latency, memory, and Android latency from the same immutable artifacts;
- use a predeclared non-inferiority margin for accuracy (for example, no worse than 0.5 percentage points on the matched target) before describing an efficiency advantage;
- never call a cross-paper CPU/GPU comparison “faster” without qualifying the hardware/runtime difference. The direct local efficiency comparison is between locally measured Mango Mamba variants; paper rows remain cited context.

## 1A. Reuse-first asset registry

Inspect and reuse these assets before writing a replacement. Make a copy into `results/table6_completion_YYYYMMDD/tools_snapshot/`, record its source path and SHA-256, and patch only that copied version. Do not silently alter a historical runner.

| Asset | Reuse decision | Required V3 adaptation / limitation |
|---|---|---|
| `reproduce_mm/scripts/run_h3_followups.py` on `asus-ascent-gx10` | **Reuse directly as the training run/configuration reference.** It already uses `320x320`, `[8,8,5,5]`, the four project seeds, idempotent completion checks, and the current Phase-1/Phase-2/Phase-3 follow-up flow. | Add only a new, isolated fine-tune/control entry point after the retrain decision; preserve its checkpoint-discovery and manifest conventions. |
| `reproduce_mm/scripts/benchmark_batch_size.py` and existing `logs/batch_benchmark*.json` | Reuse its GPU metadata, JSON output, cache cleanup, warm-up, and measurement logging patterns. | It measures **training-step throughput** at legacy `224x224`, not inference latency. Its old numbers must not enter Table 6. Build `benchmark_inference_v3.py` from it with checkpoint loading, native settings, CUDA events, per-image samples, and the Section 4 protocol. |
| `reproduce_mm/scripts/run_reproduction.py` | Reuse only generic command/manifest patterns if useful. | It encodes legacy `224x224` exploratory experiments. It is not the current trainer and cannot supply baseline settings/results. |
| `../Mango-Mamba/for_android_app/deploy/export_onnx.py` on host | **Reuse as the ONNX export and parity-check base.** It already handles checkpoint formats, `EXPORT_MODE`, ONNX checker, ONNX Runtime parity, labels, and model-size logging. | Copy into the tools snapshot as `export_v3_cam_onnx.py`; force Mango Mamba architecture arguments (`320x320`, `[8,8,5,5]`), emit logits + final feature map, and write preprocessing/model manifests. Its current default is 224 and logits-only. |
| `../Mango-Mamba/for_android_app/eval/benchmark_leaf_annotate.py` | Do not use for Mango Mamba classification timing. | It benchmarks a separate Leaf-Annotate segmentation workflow; it may be considered only later for optional leaf-mask preprocessing, never as a classifier or CAM benchmark. |
| `../Mango-Mamba` branch `main` historic Grad-CAM image assets | Use as a figure-layout/reference example only. | They are historic visual assets, not V3 evidence. Regenerate all explanation maps from selected V3 checkpoint SHA-256 and keep a source manifest. |

The host repository is currently dirty outside `reproduce_mm`. All new work stays under `/home/admincsc/workspace/binh/reproduce_mm/results/table6_completion_YYYYMMDD/` and its local `tools_snapshot/`; do not reset, clean, or modify neighbouring repositories as part of this plan.

## 2. Deliverables before editing the report

Save all new artifacts under a timestamped directory, for example `results/table6_completion_YYYYMMDD/`, and do not modify existing checkpoint/result folders.

1. `environment.json`: hostname, GPU/CPU/RAM, OS, CUDA, PyTorch, ONNX Runtime, Python, commit hashes, exact command, and timestamp.
2. `checkpoint_manifest.json`: one immutable record per measured checkpoint: model row, dataset/phase, seed, checkpoint path, SHA-256, input size, window schedule, classes, and selected epoch.
3. `latency_raw.csv` and `latency_summary.csv`: one row per seed/model/runtime; include warm-up count, measured count, mean, median/p50, p95, standard deviation, throughput, batch size, precision, and whether preprocessing is included.
4. `prediction_records.parquet` or `prediction_records.csv`: sample ID/path, true class, predicted class, per-class probabilities/logits, confidence, margin, entropy, checkpoint SHA-256, and split name.
5. `error_analysis/`: confusion matrices, per-class metrics, false-positive/false-negative image panels, error taxonomy, and CAM overlays.
6. `android_export/`: immutable selected checkpoint, ONNX files, label map, preprocessing manifest, parity-test results, model hash, and Android integration notes.
7. `table6_proposed_rows.md`: ready-to-paste replacement rows plus footnotes. Do not edit `MANGO_MAMBA_FINAL_REPORT.md` until these artifacts pass acceptance checks.

## 3. Checkpoint inventory and selection

### 3.1 Rows that need local latency

Measure every completed local Mango Mamba row already present in Table 6, at its trained/evaluated resolution:

1. Mango Mamba (Ablation: 224px, `[8,8,14,7]`), MangoLeafBD.
2. **Mango Mamba — Phase 1**, MangoLeafBD, `320x320`, `[8,8,5,5]` — the active baseline.
3. Mango Mamba — Direct-to-VN control, VN-MangoLeaf, `320x320`, `[8,8,5,5]`.
4. Mango Mamba — Phase 2, Combined, `320x320`, `[8,8,5,5]`.
5. Mango Mamba — Phase 3, VN-MangoLeaf, `320x320`, `[8,8,5,5]`.

Use `model_best.pth.tar` for every seed. For the active Phase-1 baseline, first reproduce held-out test metrics from all four saved best checkpoints. Select one deployment source checkpoint by: highest **validation-selected** checkpoint test result only after the run is complete, then lowest validation loss, then lowest seed. Record the selection rule and SHA-256. Do not select a checkpoint by manually inspecting test images or errors.

The baseline is the canonical Mango Mamba configuration (320px, `[8,8,5,5]`), not the historical exploratory variants.

## 4. Benchmark protocol: inference speed per Mango Mamba model

### 4.1 Separate metrics; never mix them

Produce these independently:

- **GPU model-forward latency:** PyTorch CUDA, batch 1, model-forward only; primary local latency reported in Table 6.
- **GPU end-to-end latency:** decode + resize/normalize + host-to-device transfer + model + postprocessing; useful deployment metric, reported separately.
- **CPU model-forward latency:** optional, batch 1, pinned CPU-thread settings. It is not directly comparable to the published CPU rows unless device/runtime are identical.
- **Android end-to-end latency:** final exported V3 only, measured on each named phone/chipset. Do not put it in the generic GPU column.

### 4.2 Reproducible GPU procedure

For each seed checkpoint and required Table-6 row:

1. Load the checkpoint with exactly its saved model arguments and its native input resolution. Set `model.eval()` and inference mode; do not apply augmentation.
2. Log the resolved model shape, device, precision, software versions, checkpoint SHA-256, and GPU clocks/power mode when available.
3. Use fixed, already resident input tensors for model-forward timing. Warm up **200** forwards, then record **1,000** individual batch-1 forwards using `torch.cuda.Event(enable_timing=True)` with synchronization before/after the benchmark. Use `torch.inference_mode()` and AMP only if explicitly reporting an FP16/BF16 result.
4. Repeat the entire process **five independent times** after reload or process restart. Report seed-level mean/p50/p95 and then model-row mean ± standard deviation across all measured seed/checkpoint runs.
5. Benchmark end-to-end latency separately using at least 200 real held-out images with a fixed preprocessing pipeline. Report image I/O separately from GPU execution where possible.
6. Do not compare V0 224px and V3 320px as an equal-workload speed contest. Put both native resolutions in the table footnote. Optionally add a clearly labelled normalized experiment at one common resolution, but never substitute it for the native-protocol result.

### 4.4 New efficiency experiment: measurements that support the paper claim

Use the old paper's Table-6 form, but fill it with new V3 evidence. For every local model/dataset row, collect the following measurement card under one manifest: accuracy/F1, parameter count, inference-only checkpoint size, ONNX size, GPU batch-1 median/p95 latency, peak GPU memory, and (for the selected V3 release) Android warm/cold end-to-end latency and peak app memory.

The primary baseline is **Mango Mamba Phase 1** at `320x320` / `[8,8,5,5]`. Legacy 224px `[8,8,14,7]` serves as an exploratory reference, not an equal-workload speed baseline. The paper-comparison narrative must state that any benefit over DeiT/Swin/ViX is a model-size/deployment comparison unless the competing model is re-measured locally on the same device/runtime.

### 4.3 Acceptance criteria for latency

- No `cuda.synchronize()` is omitted around event timing boundaries.
- Batch size is 1 for “per image”; throughput uses its own stated batch size.
- Each reported timing links to raw samples and a checkpoint SHA-256.
- Timing cells name device, precision, and resolution, for example: `2.34 ms/image (RTX ..., PyTorch CUDA, FP32, 320x320, batch 1; median)`.
- Published external numbers remain labelled as paper-reported and hardware-incomparable.

## 5. Error analysis from trained best weights

Run inference without augmentation on the official held-out test split for each required row. Keep test data read-only and do not tune model/training choices against the resulting error analysis.

### 5.1 Required analyses

1. Confusion matrix, class support, precision, recall, F1, balanced accuracy, and macro/weighted averages.
2. Per-class false-positive and false-negative rates, with confidence intervals where feasible.
3. Ranked error sets: high-confidence wrong predictions, smallest logit/probability margins, and highest entropy predictions.
4. A review panel for each class containing representative true positives, false positives, false negatives, and ambiguous/low-confidence cases. Preserve original image paths and never silently discard difficult images.
5. Error taxonomy decided by two reviewers where possible: label ambiguity, lesion/healthy overlap, multiple symptoms, occlusion/background, blur/illumination, crop/scale, and likely annotation issue. Store reviewer and rationale; mark disagreements as unresolved.
6. Compare the active Phase-1 baseline on MangoLeafBD, and Phase-3 curriculum against direct-to-VN on VN-MangoLeaf. This is the valid analysis pair for the domain transfer question.

### 5.2 Model-selection safeguard

Error analysis is descriptive. It must not choose a new epoch, change thresholds, relabel samples, or trigger retraining from held-out-test performance. Any proposed data-cleaning/relabeling action becomes a new, versioned dataset and requires a fresh split/evaluation.

### 5.3 New error-analysis figure set, matching the old-paper form

Regenerate the old paper's Figure-13-style analysis exclusively from V3 weights: one confusion-matrix figure, one per-class metric figure, and one panel of representative error cases with predicted class, true class, confidence, and CAM overlay. Include at least the reported Anthracnose/Red-Rust-type confusions when they actually occur, but do not preselect errors to reproduce the old narrative. The selection query and random seed for image-panel sampling must be stored with the prediction records.

## 6. Retraining from the current Mango Mamba SOTA checkpoint

### 6.1 Intended source and decision gate

The source is the selected **Final V3 Phase-1 `model_best.pth.tar`** checkpoint from Section 3. This is the current Mango Mamba baseline checkpoint, not the older original model.

**Do not retrain merely to obtain Grad-CAM.** The final architecture has global average pooling plus a linear head, so it can produce a class-activation map from the last normalized feature map without changing learned weights. Retraining begins only after the error analysis identifies a documented, training-split/validation-supported improvement target.

Before launching retraining, write `retrain_decision.md` with:

- error-analysis hypothesis and affected classes;
- training/validation-only intervention (for example, class-balanced sampling or a bounded augmentation change);
- unchanged test split and evaluation script;
- source checkpoint SHA-256 and selected epoch;
- expected risk, stop condition, and success metric fixed before training.

### 6.2 Controlled retraining design

1. Clone the resolved Mango Mamba Phase-1 configuration exactly: `320x320`, `[8,8,5,5]`, optimizer/schedule, splits, and preprocessing.
2. Start from the selected `model_best.pth.tar`; preserve it immutable. Use an explicit fine-tune run directory and a new optimizer state unless a true interrupted-run resume is intended and documented.
3. Run the proposed change and a no-change continuation control with the same seeds. With only two processes allowed, run one treatment and one control in parallel; do not run more than two training jobs total.
4. Select the retrained checkpoint by validation only. Evaluate once on held-out test after selection. Report all seeds, including failed/no-improvement seeds.
5. Promote a retrained checkpoint to the Android candidate only if it meets the predeclared validation criterion, preserves or improves held-out test performance without regression in safety-critical classes, passes export parity, and has a recorded SHA-256.

### 6.3 Minimum new experimental matrix

Run no new training until the reuse inventory, latency inventory, and error analysis are complete. If the decision gate approves retraining, the minimum publishable matrix is:

1. Existing Mango Mamba Phase-1 four-seed baseline (immutable reference).
2. New checkpoint-initialized **control continuation** using no intervention.
3. New checkpoint-initialized **single documented treatment** motivated by training/validation evidence, with the same seeds and budget as the control.

This supports a clear improvement claim even if raw accuracy does not increase: quantify whether the treatment maintains validation/test accuracy while improving a predeclared practical criterion such as calibration, per-class recall, model size, or latency. Do not alter architecture, data, resolution, and training schedule simultaneously; otherwise the result cannot explain the source of any efficiency or robustness change.

## 7. Android model and CAM/Grad-CAM-equivalent deployment

The existing app branch is `https://github.com/binhpt310/Mango-Mamba/tree/v3/for_android_app_only`. Keep the current release model until the candidate passes every gate below.

### 7.1 Export design

1. Export the selected V3 checkpoint to a **classification ONNX model** returning logits.
2. Export a second **explanation ONNX wrapper** returning both logits and the final normalized spatial feature map *before* adaptive average pooling. Do not alter classifier weights.
3. Save `labels.json`, `preprocess.json` (RGB order, `320x320`, resize/crop policy, normalization mean/std, interpolation), model input/output names, ONNX opset, source checkpoint SHA-256, and ONNX SHA-256.
4. Verify PyTorch versus ONNX logits on a fixed test corpus. Record maximum absolute error, top-1 agreement, and probabilities. Block release if top-1 disagrees or numerical drift exceeds a predeclared tolerance.

The V3 exporter must start from the existing Android `export_onnx.py` parity logic rather than a fresh unverified exporter. Its default `224` input and logits-only wrapper are invalid for this release until the copied V3 wrapper is changed and parity-tested at `320x320`.

### 7.2 Correct explanation terminology

Because the classifier is `feature map -> global average pooling -> linear head`, compute a **Class Activation Map (CAM)** on device:

`CAM_k(h,w) = sum_c classifier_weight[k,c] * feature_map[c,h,w]`.

This has the same spatial ranking as Grad-CAM for this GAP-plus-linear head (up to a positive constant). Label it `CAM (Grad-CAM-equivalent for the GAP-linear classifier)` in the app and paper; do not call an arbitrary heat map “Grad-CAM.”

On Android:

1. Capture from camera or accept a user-selected image only after user action and runtime permission.
2. Apply the exact exported preprocessing pipeline, run the classification and explanation models, select the predicted class (or a user-selected displayed class), calculate CAM, ReLU/normalize it, resize to the original displayed image, and alpha-overlay it.
3. Display predicted class, confidence, input resolution, and a short notice: “Highlight regions are model evidence, not a diagnosis.”
4. Process images locally by default. Do not upload/copy user images or retain them unless the user explicitly requests save/share; delete temporary camera files after use.
5. Provide a toggle to show/hide the overlay and an accessibility-safe non-color-only confidence display.

### 7.3 Android acceptance tests

- At least 50 fixed reference images, including all classes and difficult cases: Android top-1 matches ONNX desktop results exactly.
- CAM output dimension, normalization, and overlay alignment are checked on portrait/landscape image orientations and camera-captured images.
- Cold-start and warm end-to-end latency, peak memory, failure handling, offline operation, and model hash are recorded for every tested device.
- Manual checks confirm no camera/gallery image leaves the device in the default configuration.
- Rollback remains possible by retaining the current model asset and a manifest-selected active model version.

## 8. Host execution and two-process schedule

Host: `asus-ascent-gx10`, work directory: `/home/admincsc/workspace/binh/reproduce_mm`.

Use a dedicated tmux session and a lock/manifest that prevents more than two active GPU jobs. Benchmarking should be serialized per GPU unless a resource check proves safe; training slots have priority only after the benchmark inventory and error analysis are complete.

Suggested tmux layout:

```bash
cd /home/admincsc/workspace/binh/reproduce_mm
tmux new-session -d -s mango-table6
tmux new-window -t mango-table6 -n inventory
tmux new-window -t mango-table6 -n benchmark-a
tmux new-window -t mango-table6 -n benchmark-b
tmux new-window -t mango-table6 -n analysis-export
tmux attach -t mango-table6
```

Execution order:

1. Snapshot/review reusable scripts; create only copied V3-specific wrappers under the timestamped results tool directory.
2. Inventory checkpoints, validate args/weights, make hashes, and reproduce baseline test metrics.
3. Run GPU latency benchmarks. Use one GPU benchmark process at a time unless separate GPUs are confirmed; a second slot may run CPU-side manifest/error processing. Monitor `nvidia-smi` and save logs.
4. Generate prediction records and error analysis from immutable best weights.
5. Decide whether retraining is justified. If yes, run **one treatment + one matched control** only; if not, proceed to export.
6. Export the selected V3 model and explanation wrapper using the copied existing exporter; perform parity tests.
7. Update the Android app in a separate branch/commit, run device tests, then prepare report rows/footnotes.

Never place the sudo password in a shell command, tmux command, log, manifest, source file, or Git history. Use `sudo` interactively only when a privileged operation is genuinely necessary; normal work inside the user-owned workspace should not require it.

## 9. Report update criteria

Only after all artifacts exist, update Table 6 with local Mango Mamba timing values and a table note such as:

> Local Mango Mamba latencies are batch-1 median model-forward measurements on the stated device, runtime, precision, and native input resolution; they are not directly comparable with paper-reported values measured on different hardware or runtimes.

Keep the V3 Phase-1 row visually identified as the baseline. The V0 and controls stay in the table as experimental comparison rows. Add Android latency only when it has been measured on identified devices, and link the explanation/export manifest rather than claiming a mobile result from desktop timing.

The results section should mirror the old paper's structure while making the evidence stronger: (1) four-seed V3/control ablation table, (2) transparent protocol/environment table, (3) Table-6 efficiency comparison with local timing provenance, (4) error/CAM analysis from current weights, and (5) Android ONNX/CAM parity plus on-device latency. This structure lets the paper demonstrate a lightweight deployment advantage even if V3 is not the highest-accuracy model, without overstating cross-hardware comparisons.

## 10. Persistent agent handoff prompt and host-to-local artifact return

The following text is intentionally stored in this plan on both machines. Any AI agent asked to continue the work must read it before running a command.

> **Mango Mamba continuation prompt**
>
> You are continuing the Mango Mamba experimental program. First read `MANGO_MAMBA_COMPLETION_PLAN.md`, `MANGO_MAMBA_FINAL_REPORT.md`, and the current result manifests. The only active proposed baseline is **Mango Mamba: 320x320, window sizes [8,8,5,5], hybrid + ECA + GRN + SwiGLU** (canonically defined in [`04_Code Scripts and Logs/Source Snapshot - Mango Mamba Final V3/mango_mamba.py`](file:///d:/Mango-Mamba-backup/Mango-Mamba-v3/mango_mamba_on_host_asus/04_Code%20Scripts%20and%20Logs/Source%20Snapshot%20-%20Mango%20Mamba%20Final%20V3/mango_mamba.py)). Older exploratory runs with `[8,8,14,7]` are historical/ablation comparator rows only.
>
> Work on the host in `/home/admincsc/workspace/binh/reproduce_mm`. Reuse existing tools before making new ones: `scripts/run_h3_followups.py` is the training/config reference; `scripts/benchmark_batch_size.py` is a training-throughput template only, not a source of inference values; and `../Mango-Mamba/for_android_app/deploy/export_onnx.py` is the ONNX/parity base that must be copied and adapted for 320px Mango Mamba plus CAM output. Do not run legacy `224x224` runners and label their output as the proposed Mango Mamba baseline.
>
> Always invoke the project interpreter explicitly: `/home/admincsc/workspace/binh/reproduce_mm/.venv-mango/bin/python`. Do not use the system `python`/`python3`, which does not contain the required Mango Mamba dependencies. Record the interpreter path and package versions in every environment manifest.
>
> Put every new artifact under a unique `results/table6_completion_YYYYMMDD_HHMMSS/` directory. Before a run, save a manifest containing command, Git commit, source-tool SHA-256, checkpoint SHA-256, seed, native resolution, windows, device/runtime, and dataset split. Preserve the existing artifacts and never delete/reset/clean the host repository or neighbouring repositories.
>
> Use `tmux` and never exceed two active work processes. Before launching anything, inspect the existing `mango-table6` (or named continuation) session and current GPU usage. With one GPU, serialize GPU training/benchmarking; the second slot may be CPU-only manifest, prediction, or report processing. Capture stdout/stderr with `tee` into the timestamped results directory. Never put credentials in commands, scripts, logs, manifests, prompts, or Git.
>
> Do not invent a retraining treatment. First complete/reuse the V3 checkpoint inventory, native-resolution latency measurement, and error analysis. A new training job requires a written `retrain_decision.md` containing a training/validation-supported hypothesis, predeclared metric, source checkpoint, matched no-change control, seeds, and stop condition. Select checkpoints on validation only; use held-out test exactly once after selection.
>
> When any task finishes or reaches a stable checkpoint, return its complete evidence package to the local workspace `D:/mango_mamba_on_host_asus/` (or the current local Mango Mamba workspace) while preserving relative paths. Copy: manifests, commands/logs, args/configs, `training_results.json`, metric/prediction/error-analysis outputs, model/export hashes, and the selected lightweight inference checkpoint/ONNX files when their acceptance checks pass. Verify host and local SHA-256 after transfer, then add a `SYNCED_TO_LOCAL.md` file in the artifact directory with source path, destination path, transfer time, and hashes. Do not copy raw datasets or unapproved user camera/gallery images.
>
> Only after the local copy is verified may the agent propose the exact Table-6 cell values and report wording. All local timing values must state hardware, runtime, precision, batch size, and native resolution. Do not claim faster/better than a literature model measured on different hardware; describe the measured accuracy-efficiency trade-off precisely.

### 10.1 Immediate tmux preflight (safe to launch now)

The first session establishes provenance and reveals what is already complete; it is not a substitute for a justified new treatment. From the host workspace:

```bash
cd /home/admincsc/workspace/binh/reproduce_mm
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/table6_completion_${STAMP}"
mkdir -p "$RUN_DIR"/{logs,manifests,tools_snapshot}
tmux has-session -t mango-table6 2>/dev/null || tmux new-session -d -s mango-table6 -n preflight
tmux send-keys -t mango-table6:preflight \
  "cd /home/admincsc/workspace/binh/reproduce_mm && .venv-mango/bin/python scripts/check_environment.py 2>&1 | tee '$RUN_DIR/logs/environment.log'" C-m
tmux new-window -t mango-table6 -n inventory \
  "cd /home/admincsc/workspace/binh/reproduce_mm && find results -type f \\( -name model_best.pth.tar -o -name model_best_inference.pth -o -name training_results.json -o -name args.yaml \\) -print | sort | tee '$RUN_DIR/logs/artifact_inventory.log'"
tmux attach -t mango-table6
```

When a `retrain_decision.md` has been approved, open at most two named training windows for the matched control and treatment. The command must be derived from `run_h3_followups.py` and must retain `--input-size 3 320 320` and `--window-size 8 8 5 5` unless the approved decision explicitly tests one of them.

## 11. Completed current execution and next gate

The simplified single-checkpoint baseline measurement has completed, and the subsequently requested VN-MangoLeaf curriculum error analysis and Android CAM export are also complete. No retraining was launched.

1. **Mango Mamba Phase-1 seed 3333** (post-hoc strongest existing MangoLeafBD test result: 100.0% accuracy, 0 errors) was measured at 320px, FP32, CUDA, batch 1: 100 warm-up + 500 samples on NVIDIA GB10 yielded 4.566 ms mean, 4.545 ms median, 4.594 ms p95, and 0.130 ms standard deviation. The 400-image MangoLeafBD test re-evaluation had zero errors. Evidence: `results/table6_completion_20260913_051241/v3_phase1_seed3333_simple/`; checkpoint SHA-256 `973d6d4f2c0f50aa1d27e9990b7cfecd41e31adddfe55442a3961427c757b1d1`.
2. **Mango Mamba Phase-3 curriculum seed 42** (post-hoc strongest existing VN-MangoLeaf test result: 99.286% accuracy) was analyzed on the 700-image held-out split, without training/tuning: 99.286% accuracy, 99.280% macro-F1, and five saved misclassifications. All five are Anthracnose predicted as Gall_Midge (1), Red_Rust (2), or Sooty_Mold (2); exported artifacts include original images, CAM panels, CSV, metrics, and confusion matrix. Evidence: `results/table6_completion_20260913_051241/vn_phase3_seed42_error_analysis/`; checkpoint SHA-256 `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae`.
3. **Android CAM export** for that immutable Phase-3 checkpoint passed ONNX structural and ONNX Runtime parity checks. `mango_mamba_v3_phase3_320_cam.onnx` is a static batch-1 320px graph with outputs `logits [1,7]` and `feature_map [1,384,10,10]`; `mango_mamba_v3_phase3_320_cam_android_cam_manifest.json` has the labels, preprocessing, classifier weights, CAM contract, and parity values. Evidence: `results/table6_completion_20260913_051241/android_v3_phase3_seed42/`. Export does not constitute Android-device latency or app integration.
4. Completed packages are copied to `D:/mango_mamba_on_host_asus/06_Host Synced Evidence/table6_completion_20260913_051241/`. `MANGO_MAMBA_FINAL_REPORT.md` now contains the single measured Table-6 Mango Mamba inference cell, speed caveat, VN error analysis, and export status.

The next permitted work is either (a) Android app integration/device validation using the frozen ONNX/manifest, or (b) a new retraining treatment only after writing the Section 6 `retrain_decision.md` based on a training/validation-supported hypothesis. Do not use the held-out errors themselves to choose or tune a treatment.

### 11.1 Exact Android v3 handoff: user-image/camera CAM toggle

An Android agent must use a **clean worktree of GitHub branch `v3`** and the `for_android_app_only/` module present on that branch. Do not use or modify the dirty host clone currently on `v2`; do not infer paths from that clone if `v3` differs. First locate the Android app module, its ONNX Runtime session wrapper, and the screen which accepts gallery/camera images.

Use these immutable, already parity-validated inputs from the synced evidence package:

```text
06_Host Synced Evidence/table6_completion_20260913_051241/android_v3_phase3_seed42/
  mango_mamba_v3_phase3_320_cam.onnx
  mango_mamba_v3_phase3_320_cam_android_cam_manifest.json
  export.log
```

Copy only the ONNX model and JSON manifest into the app's versioned assets/model directory. Do **not** copy training checkpoints, raw datasets, test photos, or any user-captured image into Git. The app must read the manifest rather than hard-code a stale 224px contract:

1. Input is static `NCHW float32 [1,3,320,320]`, output names are `logits` and `feature_map`, and class order is the manifest `class_names` list.
2. Reproduce preprocessing exactly: decode in RGB order; resize the shorter side to 366; center-crop to 320×320; normalize RGB with mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`; pack channels in NCHW. Keep the original/cropped bitmap to align the overlay.
3. In one ONNX Runtime call, request both outputs. `logits` has `[1,7]`; `feature_map` has `[1,384,10,10]`. Use the predicted argmax by default, but allow a user-selected predicted-class explanation only—never present a CAM as diagnosis certainty.
4. Load `classifier.weight[targetClass][channel]` from the manifest and compute, for every `y,x` in the 10×10 map: `cam[y,x] = max(0, sum(channel=0..383, featureMap[0][channel][y][x] * weight[targetClass][channel]))`. Divide by `max(cam)` only if it is positive; bilinearly resize to the displayed cropped image and alpha-overlay it. The UI label must say **“CAM (Grad-CAM-equivalent)”** and **“attention map, not lesion segmentation.”**
5. Add an accessible Show CAM / Hide CAM toggle. The default state is hidden. Both gallery import and camera capture must use the exact same preprocessing and output path. Handle denied permissions, malformed images, zero CAM, and ONNX errors without crashing; do not retain camera/gallery images after the inference session unless an explicit user action saves them.
6. Preserve normal prediction display (class + confidence) when CAM is hidden. Show a non-colour-only text explanation and confidence so the overlay is not the sole output.

Required v3 app checks before merging:

- The app loads the model SHA-256 `f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4` and manifest checkpoint SHA-256 `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae`.
- A fixed 50-image, non-user test corpus has Android top-1 agreement with desktop ONNX; record device, Android version, ONNX Runtime version, cold/warm end-to-end latency, and memory. Do not call the GB10 4.566ms measurement Android performance.
- Add a unit test for the 10×10 CAM formula and a UI/instrumentation test verifying Show/Hide does not change logits or predicted class.
- Run the Android build/tests and commit the app changes, model, manifest, and a concise `ANDROID_CAM_INTEGRATION.md` containing the above results. Keep a provenance link to this plan and the export log.

### 11.2 Publication boundary for GitHub v3

The local workspace is the complete evidence store for the table sources and synchronized new package. GitHub should receive the report/plan, V3 measurement and analysis scripts, command logs/args/JSON/CSV/PNG outputs, the **five selected VN-MangoLeaf error source images and five CAM panels**, the 24.12 MiB CAM ONNX and manifest, and hashes. Do not publish the full raw datasets, user/camera images, or multi-gigabyte training/checkpoint archives unless the repository explicitly enables and the maintainer approves Git LFS. Before pushing, use a clean `v3` worktree, inspect `git status`, `git lfs`, file sizes, and branch upstream. The current host `Mango-Mamba` clone is on dirty `v2`; it is not an authorized publication worktree.
