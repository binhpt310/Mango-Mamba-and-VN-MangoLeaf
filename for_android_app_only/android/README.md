# MangoMamba VN-MangoLeaf Android classifier

This Android app runs the Phase 3 VN-MangoLeaf model with exactly seven classes:

`Anthracnose`, `Dieback`, `Gall Midge`, `Healthy`, `Powdery Mildew`, `Red Rust`, and `Sooty Mold`.

`MainActivity.kt` loads only `app/src/main/assets/mango_mamba_7class.onnx`. Model lineage, hashes, and training metadata are in `../ANDROID_MODEL_PROVENANCE.md` and `../provenance/`. Build outputs and device-test data stay local under `app/build/` and `../artifacts/`.

## Build

```powershell
$env:ANDROID_HOME = 'C:\Users\Admin\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
.\gradlew.bat clean :app:assembleDebug
```

The built APK is `app/build/outputs/apk/debug/app-debug.apk`.

## Model contract

- Input: `image`, `float32[1,3,224,224]`
- Output: `logits`, `float32[1,7]`
- Preprocessing: resize short side to 256, center crop 224, ImageNet normalization

For an ADB smoke test, put an image in the app external-files directory and run:

```powershell
adb shell am start -n com.mangomamba.benchmark/.MainActivity --es test_image_path /sdcard/Android/data/com.mangomamba.benchmark/files/test.jpg
```

## Replacing the model

Use `../scripts/replace_android_7class_model.ps1` with a new `.onnx` and labels JSON. The script only accepts the current seven labels in the exact order. It does not validate a new model's input/output contract or accuracy. Build, install, run the device batch benchmark, then update `../ANDROID_MODEL_PROVENANCE.md` with the new hashes and results.
