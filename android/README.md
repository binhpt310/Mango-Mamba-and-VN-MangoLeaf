# Android CAM — VN-MangoLeaf 320px

Ứng dụng Kotlin chạy ngoại tuyến. `MainActivity.kt` nạp duy nhất cặp asset:

- `app/src/main/assets/mango_mamba_v3_phase3_320_cam.onnx`
- `app/src/main/assets/mango_mamba_v3_phase3_320_cam_manifest.json`

Model lấy từ phase 3, seed 42. Input `image: float32[1,3,320,320]`; output `logits: [1,7]` và `feature_map: [1,384,10,10]`. Resize cạnh ngắn 366, center crop 320, chuẩn hóa ImageNet. Thứ tự nhãn và trọng số classifier/CAM nằm trong manifest; giữ nguyên thứ tự mà `MainActivity.kt` kiểm tra.

## Nguồn model và checksum

- [Checkpoint phase 3 seed 42](../experiments/04_phase3_vn_mangoleaf_320_seed_42/model_best.pth.tar): SHA-256 `208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae`.
- ONNX đang dùng: SHA-256 `f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4`.
- Manifest runtime: SHA-256 byte CRLF `c5cc1da8a463d6b749ae82a559d512823f0326b2cab545e05e42deff43d89b3a`; chuẩn hóa LF cho hash `68900e8ba5812d5940e2b8b13a6f2e315992240ff9cb1b13169a373516bc181a`, cùng nội dung JSON.

Thứ tự nhãn bắt buộc: `Anthracnose`, `Dieback`, `Gall_Midge`, `Healthy`, `Powdery_Mildew`, `Red_Rust`, `Sooty_Mold`. CAM dùng `ReLU(sum_k F[k,y,x] * classifier.weight[class,k])`, chia cho giá trị lớn nhất dương rồi nội suy song tuyến; đây là coarse CAM, không phải segmentation.

[Manifest export gốc](../experiments/04_phase3_vn_mangoleaf_320_seed_42/android_export_manifest.json) và [log export](../experiments/04_phase3_vn_mangoleaf_320_seed_42/android_export.log) ghi ONNX checker đã pass, sai số logits `1.1920928955078125e-06` và feature map `6.4373016357421875e-06`, tolerance `0.001`. Checkpoint seed 42 đạt 695/700 ảnh test đúng, khác với kết quả trung bình bốn seed.

Luận văn báo cáo Redmi Note 11S, Android 13, FP32, 50 lần cùng một ảnh: mean 612,80 ms, SD 48,47 ms, median 598,48 ms, p95 718,51 ms. Chưa có raw CSV/log và APK tương ứng trong workspace. APK cũ có SHA-256 `dda3b386a7ebc7be6d1031ff5214abe3162c3f4f8025569c1ca604bcc4731178`, chỉ chứa ONNX 224px; đã lưu cùng bộ provenance cũ ngoài workspace tại `E:/Luan van thac si/Mango-Mamba-archive/2026-10-15-reorganization/for_android_app_only/`.

## Build

Cần JDK 17 và Android SDK có platform 34. Thiết lập `JAVA_HOME`, `ANDROID_HOME` theo máy hiện tại, hoặc tạo `local.properties` với `sdk.dir`; không commit đường dẫn máy cá nhân.

```powershell
cd android
.\gradlew.bat :app:assembleDebug
```

Linux: `bash android/gradlew -p android :app:assembleDebug`. APK build ra `android/app/build/outputs/apk/debug/app-debug.apk`. Chỉ đưa vào `releases/` sau khi kiểm tra cặp ONNX/manifest và ghi checksum. APK 224px trước đây được lưu ngoài workspace, không dùng làm bản cài của app CAM 320px.

## Smoke test thiết bị

```powershell
adb shell am start -n com.mangomamba.benchmark/.MainActivity --es test_image_path /sdcard/Android/data/com.mangomamba.benchmark/files/test.jpg
```

Ứng dụng cũng nhận `test_batch_manifest_path` để đánh giá một danh sách ảnh. Đó là công cụ kiểm tra, chưa thay cho dữ liệu thô của protocol 50 lần cùng một ảnh được luận văn báo cáo.

## Cập nhật model

1. Dùng [exporter](../code/export_android_cam.py) với checkpoint và output mới theo [protocol](../docs/EXPERIMENT_PROTOCOL.md). Exporter tạo `<stem>.onnx` và `<stem>_manifest.json`, kiểm tra ONNX và parity.
2. Kiểm tra thứ tự lớp, tensor contract, preprocessing; chép cả hai file vào assets dưới đúng tên MainActivity đang nạp. Trọng số CAM trong manifest phải đi cùng ONNX.
3. Build, kiểm tra trên thiết bị rồi ghi hash APK/ONNX/manifest và raw benchmark mới. Giữ đường dẫn lịch sử trong chứng cứ gốc; tra vị trí hiện tại qua [manifest sắp xếp](../docs/REORGANIZATION_MANIFEST.json).
