# Feedback của thầy và định hướng

Tổng hợp mọi nhận xét của thầy theo thời gian, kèm trạng thái xử lý. Mục nào làm xong thì đổi `[ ]` thành `[x]`
và ghi file / commit đã giải quyết nó.

---

## Đợt 1 — Lỗi tái hiện (file `feedback_ThầyDu.docx`, đầu tháng 10/2026)

Thầy: chưa nên chuyển sang Transformer hay kiến trúc mới, vì pipeline tái hiện còn lỗi.

| # | Nhận xét | Cách xử lý trong repo | Trạng thái |
|---|---|---|---|
| 1.1 | **L2 = 0.01 bị áp cho mọi Conv1D**, paper chỉ áp cho Dense → regularize quá mạnh, có thể là lý do cả 3 ResNet thấp hơn paper | `models.py`: L2 chỉ ở Dense ẩn; YAML ghi rõ `conv_l2: 0.0` | [ ] |
| 1.2 | Không dùng `validation_split=0.1`; tách train/val **có phân tầng**; MinMax **chỉ fit trên train** rồi áp sang val/test | `data.py`: `make_splits` + `load_split` | [ ] |
| 1.3 | Code tính P/R/F1 **macro**, paper nhiều khả năng dùng **weighted** → báo cáo cả hai, dùng weighted để so paper | `train.py`: ghi cả macro và weighted vào log | [ ] |
| 1.4 | Lưu số epoch thực chạy, epoch có val_loss nhỏ nhất, epoch có val_acc lớn nhất, learning curves. Thử trên cùng split 3 trường hợp: **chạy đủ 500 epoch**, **ES theo val_loss**, **ES theo val_accuracy** | log có `epochs_ran`, `best_epoch_val_loss`, `best_epoch_val_acc`; lưu `history.csv` + `curve.png`; 3 trường hợp = baseline + `ablation_noES` + `ablation_ES_valacc` | [ ] |
| 1.5 | Projection Conv1D 1×1 hợp lý, giữ trước mắt. Sửa xong lỗi chung mà vẫn lệch thì mới so có/không BN ở projection | `ablation_noBN.yaml` | [ ] |
| 1.6 | Cố định chính xác **file dữ liệu, thứ tự 16 đặc trưng, các split, seed**. Gửi thầy **notebook đã chạy có output + bảng kết quả từng lần**, không gửi code chưa chạy | md5 file dữ liệu + `feature_order` trong YAML; `splits/*.npz` commit lên repo; notebook có output | [ ] |

## Đợt 2 — Dọn baseline (file "Làm sạch lại baseline", 13/10/2026)

Lỗi thầy bắt được ở bản notebook cũ: `model.summary()` của một model nhưng train model khác, cờ `noBN`
hard-code trong tên file, mỗi notebook một hàm log khác nhau. Nguyên tắc mới: **mỗi kết quả sinh hoàn toàn
từ một config, và config được lưu cạnh kết quả.**

| # | Yêu cầu | Cách xử lý | Trạng thái |
|---|---|---|---|
| 2.1 | Một pipeline dùng chung cho 3 model, đọc từ config, không sửa tay giữa các lần chạy | `configs/*.yaml` + `src/` | [ ] |
| 2.2 | `run_id` sinh tự động; run đã `completed` thì bỏ qua (an toàn khi Colab ngắt) | `config.make_run_id`, `run.py` | [ ] |
| 2.3 | **Một hàm duy nhất** ghi log; log có đường dẫn checkpoint | `train.append_log` | [ ] |
| 2.4 | Đánh giá test bằng **checkpoint tốt nhất** (load `best.keras`), không dùng weights cuối | `train.py` | [ ] |
| 2.5 | Bảng baseline chính chỉ **một cấu hình**: Conv1×1+BN, L2 chỉ Dense, ≤500 epoch, ES val_loss patience 100. noBN / noES để ở bảng ablation riêng | cột `group` trong log | [ ] |
| 2.6 | **5 split phân tầng**, báo mean ± std, CI 95% (phân phối t, df = 4), accuracy + macro-F1 + weighted-F1 + recall từng lớp + confusion matrix cộng dồn | `aggregate.py` | [ ] |
| 2.7 | Kiểm tra log trước khi tổng hợp: đủ 3×5 run, không trùng, tất cả completed, run noES có `epochs_ran == 500` | `aggregate.check_log` | [ ] |
| 2.8 | Sau baseline: các baseline mạnh và thí nghiệm hoán vị cột | Giai đoạn 3 trong checklist | [ ] |

## Đợt 3 — Hướng phát triển (khoảng 15/10/2026)

Theo những gì đã trao đổi: mô hình học trên **hypergraph các đặc trưng**, trong đó mỗi công thức hình học
(xem `paper_reference.md` §1) là một hyperedge; nhiệm vụ phụ **che cả nhóm đặc trưng rồi tái tạo**; loss tập
trung vào **cặp dễ nhầm SIRA–DERMASON**. Thầy vẫn giữ thứ tự ưu tiên: **làm sạch baseline trước**.

---

## Định hướng nghiên cứu (thầy, đợt 1)

- Tập trung **phân loại dữ liệu bảng trên Dry Bean**. Không chuyển sang hồi quy (không có biến đáp ứng liên tục
  phù hợp) hay computer vision (không có ảnh gốc).
- Giữ Dry Bean để so trực tiếp với các bài cũ. Nhưng để lên tạp chí tốt, **một bộ dữ liệu có thể chưa đủ**: nên
  kiểm tra thêm vài bộ nông sản khác, hoặc tự tạo một bộ.

### Các hướng kiểm tra sau khi baseline đã sạch (theo thứ tự thầy nêu)

1. **Độ nhạy của ResNet khi hoán vị thứ tự 16 đặc trưng.** Conv1D giả định cột kề nhau thì liên quan,
   trong khi thứ tự cột là tùy ý.
2. **Global classifier không phụ thuộc thứ tự**: residual MLP hoặc TabM (thầy đánh dấu ** – ưu tiên cao).
3. **Specialist cho cặp SIRA–DERMASON**, kích hoạt khi mô hình toàn cục không chắc chắn (thầy đánh dấu *).
4. **So sánh công bằng** với RBF-SVM, XGBoost/CatBoost, MLP, TabM, và TabPFN nếu phù hợp.
5. **Ablation** cho: loại biến phụ thuộc tất định, specialist, calibration.

### Ghi chú thiết kế thí nghiệm (đã thống nhất trong các buổi trước)

**Hoán vị cột**
- Cần nhóm đối chứng: nếu accuracy qua các hoán vị dao động ±0.4% mà chỉ đổi seed khởi tạo (giữ thứ tự gốc)
  cũng dao động ±0.4% thì chưa kết luận được gì.
- Thiết kế: cùng split; nhóm A = nhiều hoán vị (gồm `original`, `reversed`, `corr_clustered`, `corr_scattered`,
  ngẫu nhiên), nhóm B = thứ tự gốc × nhiều seed. Kết luận "phụ thuộc thứ tự" khi phương sai giữa các hoán vị
  lớn hơn phương sai do seed (ANOVA / Kruskal–Wallis / Levene).
- Đối chứng âm: residual MLP / XGBoost (về lý thuyết bất biến với thứ tự). Đối chứng cơ chế: ResNet1 với
  `kernel_size = 1` → nếu hiệu ứng biến mất thì nguyên nhân là tính cục bộ của kernel.
- Cơ sở lý thuyết: receptive field của ResNet1 là 15 (stem 3 + 6 conv × 2). Cặp cột cách nhau ≥ 15 vị trí
  (vd cột 1 và 16) không bao giờ cùng nằm trong một receptive field trước Flatten.
- Ngoài accuracy, đo **tỉ lệ đồng thuận dự đoán** (cùng hoán vị khác seed vs khác hoán vị cùng seed) — nhạy hơn.

**Baseline mạnh**
- Cùng 5 file split; mỗi model **cùng ngân sách tuning** (vd Optuna 50 trial), chọn siêu tham số chỉ trên
  validation, test chỉ chạm một lần; ghi qua cùng hàm log.

**Hướng hypergraph (nếu theo)**
- Ablation bắt buộc: hypergraph hình học vs hypergraph **ngẫu nhiên** cùng số cạnh/kích thước vs đồ thị đầy đủ
  (≈ FT-Transformer) vs không cạnh. Nếu ngẫu nhiên ≈ hình học thì luận điểm chính sụp → làm sớm.
- Che nhóm + tái tạo có thể quá dễ (đặc trưng là hàm tất định của nhau): theo dõi đường cong loss tái tạo,
  nếu về ~0 sau vài epoch thì nhiệm vụ phụ vô nghĩa.
- Cải thiện SIRA–DERMASON có thể có trần (phân phối chồng lấn thật) → cần nhiều split và kiểm định;
  với split chồng lấp, dùng corrected resampled t-test (Nadeau–Bengio).
- Đóng góp không chỉ là accuracy: bất biến hoán vị, khả năng giải thích, độ bền khi thiếu đặc trưng.

**Thêm bộ dữ liệu** (cần kiểm tra lại trên UCI/Kaggle): nhóm Koklu công bố một số bộ nông sản khác với cùng
họ đặc trưng hình thái (Rice Cammeo/Osmancik, Pumpkin Seeds, Raisin, Date Fruit).

---

## Kết quả đã có (pipeline cũ, chỉ để tham chiếu)

| Giai đoạn | ResNet1 | ResNet2 | ResNet3 | Ghi chú |
|---|---|---|---|---|
| Notebook đầu (L2 trên Conv, `validation_split`) | ~91–92 | ~91–92 | – | lỗi đợt 1 |
| Sau sửa đợt 1, 1 split, 6 cấu hình | 91.9–92.5 | ≈ 92.3 | ≈ 92.3 | `runs_log.csv` cũ → đổi tên `runs_log_legacy.csv` |
| Pipeline config (13/10), 5 split, baseline | **93.21 ± 0.62** | **93.25 ± 0.61** | **92.91 ± 0.61** | test acc; ResNet3 < ResNet1 có ý nghĩa (p ≈ 0.009) |
| Paper [P] | 95.70 ± 0.51 | 93.79 ± 0.57 | 93.71 ± 0.48 | 10 lần hold-out |

Quan sát chưa giải thích (cần báo thầy): ResNet1 (batch 1024) có val_acc gần ngẫu nhiên đến khoảng epoch 90–100
rồi mới nhảy lên ~92% — giả thuyết: ít bước cập nhật mỗi epoch nên thống kê BatchNorm hội tụ chậm. **Chưa kiểm chứng.**
Fig. 6 của paper cũng có hiện tượng tương tự (val_acc thấp đến ~epoch 60).
