# Kế hoạch hoàn thiện ứng dụng nhận diện bệnh lá xoài Android

## 1. Quyết định đã chốt

- Nhánh đích: `v3` trên remote `origin`.
- Chỉ dùng model 7 lớp trong `android/app/src/main/assets/mango_mamba_7class.onnx`.
- Nhãn theo đúng `mango_mamba_7class_labels.json`: `Anthracnose`, `Dieback`, `Gall Midge`, `Healthy`, `Powdery Mildew`, `Red Rust`, `Sooty Mold`.
- Ứng dụng chạy inference hoàn toàn trên thiết bị. Không upload ảnh và không cần backend.
- Chế độ liên tục chụp ảnh tĩnh mỗi 0,8 giây. Đây không phải quay video và không dùng API quay video.
- Nếu ảnh hoặc inference trước còn đang xử lý, nhịp 0,8 giây đó bị bỏ qua. Không tạo hàng đợi và không chạy song song nhiều inference.

## 2. Mục tiêu trải nghiệm

**Design read:** Ứng dụng Android hỗ trợ người lớn chụp và nhận diện bệnh lá xoài, với ngôn ngữ tin cậy, sáng, tối giản và dùng Material 3.

Người dùng mở app, thấy ngay camera, chụp một ảnh hoặc bật chụp liên tục, đọc tên lớp và phần trăm lớn trên màn hình, sau đó mở hướng dẫn về triệu chứng và phòng chống. Không để log kỹ thuật, benchmark, tên tensor, top-k debug hoặc đường dẫn file xuất hiện trong luồng người dùng.

### Phong cách A: Lá Xanh Dễ Dùng

| Hạng mục | Quy định triển khai |
|---|---|
| Chủ đề | Chỉ giao diện sáng. Nền `#F7FBF6`, bề mặt `#FFFFFF`, xanh chính `#247A3D`, xanh nhạt `#E4F2E7`, chữ `#172217`, lỗi `#B3261E`. |
| Màu sắc | Xanh chính là accent duy nhất. Màu vàng và đỏ chỉ dùng cho trạng thái cảnh báo, không dùng trang trí. Không gradient tím, phát sáng, glassmorphism hoặc nền ảnh mờ. |
| Chữ | Dùng sans-serif hệ thống Android. Tiêu đề 24sp, tên kết quả 28sp, phần trăm 40sp, nội dung 18sp, chú thích không nhỏ hơn 16sp. Dùng `sp`, không chèn text trực tiếp trong layout. |
| Khoảng cách | Lưới 8dp. Padding trang 20dp. Thành phần tương tác cao ít nhất 48dp. Card 16dp; nút bo tròn 16dp. |
| Icon | Không emoji. Chỉ dùng icon Material có nghĩa thao tác: camera, dừng, quay lại, mở liên kết. Mỗi nút tối đa một icon ở bên trái và icon không thay thế nhãn chữ. Không dùng icon lá, khiên hoặc trang trí lặp lại. |
| Chuyển động | Chỉ phản hồi chạm, chuyển trạng thái và loading. Không auto-animation, marquee, confetti hoặc hiệu ứng thu hút chú ý. |

Mọi nút phải có nhãn tiếng Việt ngắn, tương phản WCAG AA, và không chỉ truyền đạt trạng thái bằng màu.

## 3. Phạm vi và ngoài phạm vi

| Trong phạm vi | Ngoài phạm vi |
|---|---|
| Camera preview, chụp đơn, chụp liên tục 0,8 giây | Video recording, streaming, cloud inference |
| ONNX Runtime inference 7 lớp trên máy | Train lại model hoặc thay đổi kiến trúc model |
| Kết quả top-1 lớn và top-3 tùy chọn | Màn hình benchmark cho người dùng cuối |
| Trang triệu chứng, phòng chống, nguồn Việt Nam | Chẩn đoán cây hoặc kê thuốc tự động |
| Build, cài và test thật qua ADB | Thu thập telemetry hoặc lưu ảnh hàng loạt |

## 4. Luồng màn hình

### 4.1 Camera

Màn hình mở app là camera. Phía trên chỉ có tiêu đề `Chẩn đoán lá xoài` và hướng dẫn ngắn `Đưa lá vào vùng khung, đủ sáng và lấy nét rõ.`

Camera preview chiếm phần lớn màn hình. Dưới preview có vùng kết quả hiện tại. Hai nút hành động ở cuối:

1. `Chụp ảnh`: chụp một ảnh, chuyển sang trạng thái đang phân tích, rồi cập nhật kết quả.
2. `Chụp liên tục`: bật vòng lặp chụp. Khi đang bật, nhãn nút đổi thành `Dừng chụp` và đổi sang kiểu cảnh báo rõ ràng.

Không đặt nhiều nút phụ trên camera. Nút chọn ảnh từ máy chỉ giữ trong build phát triển hoặc đặt dưới mục `Thử ảnh có sẵn`, không cạnh tranh với camera.

### 4.2 Kết quả

Sau một lần chụp, hiển thị ảnh vừa xử lý, tên bệnh tiếng Việt, tên lớp kỹ thuật nhỏ hơn, phần trăm top-1 lớn và thời gian xử lý nhỏ hơn. Hiển thị tối đa hai lớp kế tiếp trong mục `Kết quả khác` khi người dùng mở rộng.

Ngưỡng UX cần được cấu hình một chỗ:

- Từ 80%: `Độ tin cậy cao`.
- Từ 50% đến dưới 80%: `Nên chụp lại để kiểm tra`.
- Dưới 50%: `Kết quả chưa đủ chắc chắn. Hãy chụp gần hơn và đủ sáng.`

Đây là cách diễn đạt mức tin cậy của model, không phải xác nhận bệnh từ chuyên gia.

Nút `Xem triệu chứng và cách phòng` chỉ xuất hiện sau khi có kết quả. `Healthy` hiển thị `Chưa phát hiện dấu hiệu bệnh trong 7 lớp hiện có` và vẫn có hướng dẫn theo dõi lá.

### 4.3 Chụp liên tục

Khi bật, UI hiển thị trạng thái chữ `Đang chụp liên tục, cập nhật mỗi 0,8 giây` và nút `Dừng chụp`. Mỗi lần inference xong, cập nhật duy nhất các vùng: ảnh mới nhất, tên lớp, phần trăm và thời điểm cập nhật.

Quy tắc xử lý:

1. Bộ đếm 800 ms gọi yêu cầu chụp nếu `isCaptureOrInferenceRunning == false`.
2. Lưu ảnh tạm vào cache riêng của app, sửa xoay EXIF, tạo tensor và inference trên dispatcher nền.
3. Cập nhật UI trên main thread rồi giải phóng bitmap và file tạm không cần thiết.
4. Nếu bận hoặc camera báo lỗi, bỏ nhịp đó, hiển thị lỗi có thể hành động và đợi nhịp kế tiếp.
5. Bấm `Dừng chụp`, rời màn hình hoặc Activity bị dừng sẽ hủy bộ đếm và không bắt đầu ảnh mới.

Không lưu mọi khung đã chụp. Chỉ giữ ảnh mới nhất trong cache cho màn hình kết quả, trừ khi người dùng chủ động chọn lưu sau này.

### 4.4 Thông tin bệnh

Màn hình gồm tên lớp, tên tiếng Việt và bốn phần theo thứ tự: `Dấu hiệu thường gặp`, `Cách phòng chống`, `Khi nào cần hỏi cán bộ nông nghiệp`, `Nguồn tham khảo tại Việt Nam`.

Không dùng ảnh minh họa do AI cho thông tin bệnh. Chỉ dùng ảnh người dùng vừa chụp hoặc ảnh có nguồn và quyền sử dụng rõ ràng. Link mở Custom Tab hoặc trình duyệt mặc định, có tên tổ chức và ngày kiểm tra link.

## 5. Kiến trúc tối thiểu

Giữ một module Android và nền XML Views hiện có. Không chuyển sang Compose, Navigation Component, DI framework hoặc thư viện tải ảnh chỉ để phục vụ luồng này.

| Thành phần | Trách nhiệm |
|---|---|
| `MainActivity` | Host luồng camera, kết quả và thông tin; chỉ điều phối UI. |
| `CameraController` | Cấu hình CameraX, preview, `ImageCapture`, quyền camera và vòng lặp 800 ms. |
| `InferenceEngine` | Tải một ONNX session dùng lại, tiền xử lý, inference, softmax và trả về ranking. |
| `DiseaseContentRepository` | Đọc nội dung bệnh có version từ JSON trong assets hoặc `res/raw`. |
| `CaptureResult` | Dữ liệu nhỏ gồm URI/file tạm, ranking, latency và timestamp. |

Chỉ thêm dependency CameraX cần thiết: `camera-core`, `camera-camera2`, `camera-lifecycle` và `camera-view`. Tái sử dụng Material, coroutines, lifecycle và ONNX Runtime đã có. Không thêm thư viện để làm những việc Android SDK đã hỗ trợ.

## 6. Hợp đồng model và inference

Không thay đổi các quy tắc đã tạo model:

- Input `image`, `float32[1,3,224,224]`.
- Output logits `float32[1,7]`.
- Resize cạnh ngắn 256, center crop 224, ImageNet normalization.
- NCHW, softmax sau logits.
- `OrtSession` được tạo một lần, tái sử dụng xuyên suốt vòng đời app và đóng khi app thực sự kết thúc.

Trước khi tích hợp camera, tạo test ánh xạ nhãn từ JSON và test tiền xử lý với ảnh cố định. Không hard-code danh sách nhãn thứ hai trong Kotlin. Dùng JSON asset làm nguồn duy nhất để giảm nguy cơ lệch model và UI.

## 7. Nội dung bảy lớp

| Label model | Tên hiển thị dự kiến | Nội dung cần biên soạn |
|---|---|---|
| `Anthracnose` | Thán thư | Dấu vết, điều kiện phát triển, vệ sinh vườn, nguồn Việt Nam. |
| `Dieback` | Khô cành | Triệu chứng trên lá/cành, cắt tỉa an toàn, nguồn Việt Nam. |
| `Gall Midge` | Muỗi đục lá | Dấu hiệu, thời điểm theo dõi, biện pháp canh tác, nguồn Việt Nam. |
| `Healthy` | Lá khỏe | Hạn chế của 7 lớp, cách theo dõi và chụp lại khi có thay đổi. |
| `Powdery Mildew` | Phấn trắng | Dấu hiệu, điều kiện ẩm, biện pháp phòng ngừa, nguồn Việt Nam. |
| `Red Rust` | Rỉ sắt đỏ | Dấu hiệu, vệ sinh vườn, biện pháp phòng ngừa, nguồn Việt Nam. |
| `Sooty Mold` | Nấm bồ hóng | Dấu hiệu, kiểm soát côn trùng tiết mật, nguồn Việt Nam. |

Mỗi mục nội dung có cấu trúc: `modelLabel`, `displayNameVi`, `summary`, `symptoms`, `prevention`, `whenToSeekHelp`, `references`. `references` phải lưu `title`, `organization`, `url`, `checkedAt`.

Chỉ đưa thông tin và đường dẫn sau khi kiểm tra nguồn cơ quan nông nghiệp, viện/trường hoặc tài liệu khuyến nông Việt Nam. Không tự suy diễn thuốc, liều lượng hoặc cảnh báo không có nguồn.

## 8. Quyền riêng tư, lỗi và khả năng tiếp cận

- Chỉ xin `CAMERA` khi người dùng bắt đầu dùng camera. Không xin quyền lưu trữ rộng.
- Ảnh tạm nằm trong cache/app-specific storage, bị xóa theo vòng đời cache hoặc khi thay ảnh mới.
- Nếu người dùng từ chối quyền, giải thích ngắn và có nút `Mở cài đặt`.
- Nếu camera không mở được, model lỗi, ảnh mờ hoặc hết bộ nhớ, dùng thông báo tiếng Việt kèm một hành động rõ ràng: `Thử lại`, `Chụp lại` hoặc `Mở cài đặt`.
- Loading phải là nội dung mô tả, ví dụ `Đang phân tích ảnh`, không chỉ là spinner.
- Tất cả nút, preview và ảnh kết quả có content description phù hợp. Hỗ trợ TalkBack, cỡ chữ hệ thống và xoay màn hình.
- Không dùng thuật ngữ khiến người dùng hiểu kết quả model là chẩn đoán chắc chắn.

## 9. Kế hoạch thực hiện

| Mốc | Công việc | Điều kiện hoàn thành |
|---|---|---|
| M1 | Đổi thương hiệu từ benchmark sang app người dùng; dọn text mã hóa lỗi và bỏ UI benchmark khỏi luồng chính. | App mở bằng tiếng Việt rõ ràng, không còn log kỹ thuật trong UI. |
| M2 | Thêm design tokens, typography, layout Camera theo hướng A và trạng thái rỗng/loading/lỗi. | Review ảnh chụp màn hình trên thiết bị, nút lớn và không có emoji/icon trang trí. |
| M3 | Tích hợp CameraX chụp một ảnh, quyền camera, EXIF, ảnh tạm và `InferenceEngine`. | Chụp một lá qua camera, hiển thị top-1 và top-3 đúng nhãn. |
| M4 | Thêm bộ đếm 800 ms, khóa một tác vụ, dừng an toàn và cập nhật kết quả trực tiếp. | Không có hơn một capture/inference đồng thời; dừng ngay khi bấm. |
| M5 | Thêm trang thông tin và JSON nội dung 7 lớp đã kiểm chứng nguồn. | Mỗi label có nội dung, link mở được và không có khuyến cáo vô căn cứ. |
| M6 | Test, tối ưu nhẹ, cài qua ADB và ghi evidence thiết bị. | Build debug thành công, APK cài được, camera và continuous mode chạy trên thiết bị thật. |

## 10. Kiểm thử và bằng chứng nghiệm thu

### Tự động

- Test label JSON có đúng 7 nhãn và thứ tự model.
- Test softmax tổng gần 1, ranking giảm dần và model output sai kích thước bị báo lỗi.
- Test scheduler: khi bận thì bỏ tick; khi dừng thì không có capture tiếp theo.
- Test `DiseaseContentRepository`: đủ 7 mục, mỗi mục có các trường bắt buộc, URL là HTTPS.

### Trên thiết bị Android qua ADB

1. Xác nhận thiết bị ở trạng thái `device`.
2. Build debug với `ANDROID_HOME` và `ANDROID_SDK_ROOT` trỏ đến Android SDK cục bộ.
3. Cài APK, mở app, cấp quyền camera và chụp từng ảnh.
4. Bật chụp liên tục ít nhất 30 giây, quay lại một lần, rời app và mở lại.
5. Ghi lại model label, confidence, latency, số tick bỏ qua, lỗi nếu có và ảnh chụp màn hình của Camera, Kết quả, Thông tin bệnh.
6. Kiểm tra file tạm không tăng vô hạn và app không crash.

Chỉ công bố inference đã hoạt động sau khi có bằng chứng từ thiết bị. Nếu chỉ build hoặc chỉ kiểm tra code, ghi rõ phần device test còn thiếu.

## 11. Tiêu chí hoàn thành MVP

- App 7 lớp chạy offline và nhãn UI khớp nhãn model.
- Người dùng chụp được một ảnh hoặc bật/tắt chụp liên tục 0,8 giây.
- Camera không tạo hàng đợi inference, UI không bị khóa và không lưu ảnh liên tục vô hạn.
- Kết quả top-1, phần trăm và cảnh báo độ tin cậy đọc được bằng tiếng Việt.
- Người dùng mở được trang triệu chứng, phòng chống và nguồn Việt Nam cho từng lớp.
- Giao diện bám hướng A, không emoji, không icon trang trí, không hiệu ứng hoặc bố cục rập khuôn kiểu AI.
- APK được cài và kiểm thử qua ADB trên thiết bị Android thật.
