# MangoMamba

Điểm vào: [chỉ mục bằng chứng theo Bảng/Hình](docs/EVIDENCE_INDEX.md). Mỗi lượt chạy nằm trực tiếp trong `experiments/<nhóm>_<case>_seed_<seed>/`; code chung nằm trong `code/`, hình dùng trong luận văn nằm trong `docs/figures/`.

## Kết quả được báo cáo

- MangoLeafBD, giai đoạn 1: **99,69 ± 0,24%** test accuracy, 4 seed.
- Bộ dữ liệu kết hợp, giai đoạn 2: **99,48 ± 0,07%**, 4 seed.
- VN-MangoLeaf, giai đoạn 3: **98,89 ± 0,29%**, 4 seed.
- Tinh chỉnh trực tiếp từ giai đoạn 1: **98,11 ± 0,55%**, 4 seed.
- Khảo sát thành phần 224px: Full, không ECA, không GRN, không SwiGLU, Hybrid tối giản; mỗi cấu hình 4 seed.

Kết quả lấy từ `training_results.json → test_metrics` của checkpoint có validation loss nhỏ nhất. `final_metrics` là số liệu epoch cuối, có thể khác. Mô hình chính: ảnh 320×320, cửa sổ `[8,8,5,5]`, hybrid Mamba/LKA, ECA/GRN/SwiGLU và trộn liên cửa sổ ở stage 0. Khảo sát thành phần: ảnh 224×224, cửa sổ `[8,8,14,7]`, không trộn liên cửa sổ.

## Mở nhanh

- [Model dùng chung](code/mango_mamba.py) · [trainer](code/train_mango_mamba.py) · [runner](code/run_curriculum.py).
- [Checkpoint MangoLeafBD seed 3333](experiments/01_phase1_mangoleafbd_320_seed_3333/model_best.pth.tar).
- [Checkpoint VN-MangoLeaf seed 42](experiments/04_phase3_vn_mangoleaf_320_seed_42/model_best.pth.tar).
- [Hình luận văn](docs/figures/) · [Android CAM 320px](android/README.md).
- [Giao thức tái lập](docs/EXPERIMENT_PROTOCOL.md) · [nguồn model Android](android/README.md#nguồn-model-và-checksum).
- [Manifest SCP](docs/CHECKPOINT_SYNC_MANIFEST.json) · [mapping/checksum khi sắp xếp](docs/REORGANIZATION_MANIFEST.json).

## Chạy lại

Môi trường thực nghiệm gốc là Linux, CUDA 13.0, PyTorch 2.13.0; cần các thư viện mô hình trong [giao thức](docs/EXPERIMENT_PROTOCOL.md). Dữ liệu đặt tại `data/{mangoleafBD,combined_dataset,vn_mangoleaf}/{train,val,test}` hoặc truyền `--data-root`.

```bash
python code/run_curriculum.py --task phase1 --seed 42
python code/run_curriculum.py --task curriculum --seed 42
python code/run_curriculum.py --task direct-vn --seed 42
python code/run_curriculum.py --task ablation --case no_eca --seed 42
```

Runner giữ nguyên kết quả đã hoàn tất; lượt có JSON nhưng thiếu checkpoint sẽ báo lỗi. Dùng `--output runs` để chạy mới mà không ghi vào bộ chứng cứ `experiments/`. Muốn chạy curriculum ở output mới, cần tạo hoặc chép checkpoint phase 1 cùng seed vào output đó trước. Các YAML/log/JSON gốc vẫn giữ đường dẫn host lịch sử; dùng runner cho đường dẫn đang hoạt động.

## Phạm vi hiện có

Giữ đủ **36 lượt kết quả**, trong đó **16 checkpoint** của ba giai đoạn và đối chứng trực tiếp đã tải từ `asus-ascent`, kiểm tra SHA-256. Chưa tìm thấy 20 checkpoint khảo sát 224px trên host được chỉ định. Dữ liệu ảnh đầy đủ, manifest chia tập, raw benchmark điện thoại 50 lần và ảnh UI Hình 4.1 chưa có trong workspace.

APK 224px cũ, thử nghiệm ngoài phạm vi và toàn bộ cây thư mục gốc được lưu ngoài workspace tại `E:/Luan van thac si/Mango-Mamba-archive/2026-10-15-reorganization`. Workspace không phát hành APK cũ dưới tên app CAM 320px.
