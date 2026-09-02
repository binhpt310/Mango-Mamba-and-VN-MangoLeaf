# MangoMamba VN-MangoLeaf Android classifier

This Android app runs the Phase 3 VN-MangoLeaf model with exactly seven classes:

`Anthracnose`, `Dieback`, `Gall Midge`, `Healthy`, `Powdery Mildew`, `Red Rust`, and `Sooty Mold`.

`MainActivity.kt` loads only `app/src/main/assets/mango_mamba_7class.onnx`. The model lineage, hashes, training metadata, source code, and APK are one directory above this project in `../ANDROID_MODEL_PROVENANCE.md`, `../provenance/`, and `../artifacts/`.

## Build

```powershell
$env:ANDROID_HOME = 'C:\Users\Admin\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
.\gradlew.bat clean :app:assembleDebug
```

The built APK is `../artifacts/MangoMamba-VNMangoLeaf-7class-debug.apk`.

## Model contract

- Input: `image`, `float32[1,3,224,224]`
- Output: `logits`, `float32[1,7]`
- Preprocessing: resize short side to 256, center crop 224, ImageNet normalization

For an ADB smoke test, put an image in the app external-files directory and run:

```powershell
adb shell am start -n com.mangomamba.benchmark/.MainActivity --es test_image_path /sdcard/Android/data/com.mangomamba.benchmark/files/test.jpg
```
