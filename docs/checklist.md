# Checklist

Làm **theo thứ tự**. Mục nào qua được phần "Tự kiểm tra" thì mới sang mục sau.
Mỗi mục xong là 1 commit, message ghi mã mục, ví dụ `C3: data.py + 5 split`.

Phạm vi 3 ngày: **C0 → C10** (pipeline sạch + 60 run + báo cáo cho thầy). C11 là phần thêm nếu còn thời gian.
Giai đoạn 3 (hoán vị, global classifier, specialist…) là nghiên cứu, làm sau khi thầy duyệt baseline.

Cổng cuối ngày: chưa qua cổng thì chưa sang việc của ngày hôm sau.

---

## Tối Chủ Nhật 18/10 — khởi động (~1 giờ)

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **C0. Khung repo**: repo GitHub, cây thư mục, `.gitignore`, `requirements.txt`, `docs/` | 30 phút | Chép file xlsx vào `data/` mà `git status` không thấy nó |
| [ ] | **C1. Config YAML**: `baseline.yaml` (đủ 16 tên đặc trưng, đúng Table 2) + 3 file ablation chỉ ghi phần khác | 45 phút | `yaml.safe_load` đọc được cả 4 file, `lr` ra kiểu `float` |

Tối nay chưa xong C1 thì chuyển C1 lên đầu ngày 1.

## Ngày 1 — Thứ Hai 19/10: viết phần lõi (không cần GPU)

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **C2. `src/config.py`**: `load_config`, `make_run_id`, `save_config` | 1 giờ | merge noBN chỉ đổi `shortcut_bn`; cùng cfg → cùng run_id; đổi `lr` → run_id khác |
| [ ] | **C3. `src/data.py`**: `make_splits` (lưu `splits/*.npz`) + `load_split` | 1,5 giờ | 3 tập index rời nhau, hợp lại đủ 13 611; tỉ lệ lớp đều; `X_tr` đúng [0,1] |
| [ ] | **C4. `src/models.py`**: `build_model(cfg)` cho resnet1/2/3 | 1,5 giờ | params 2 044 679 / 3 833 479 / 3 887 367; chỉ Dense ẩn có L2; noBN ít params hơn |
| [ ] | **C5. `src/train.py`**: seed → data → model → callbacks → fit → load `best.keras` → metrics → lưu artefact → `append_log` | 2 giờ | 1 run 3 epoch sinh đủ file trong `results/runs/<run_id>/` và 1 dòng log đủ cột |

**Cổng ngày 1:** trên GitHub có commit C0–C5; 1 run 3 epoch chạy trọn vẹn từ đầu đến dòng log.
Nếu C5 chưa xong thì để sáng ngày 2, nhưng baseline vẫn phải bắt đầu chạy trước 11h.

## Ngày 2 — Thứ Ba 20/10: chạy baseline + noBN + ES_valacc

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **C6. `run.py`**: lặp model × split, bỏ qua run completed, có `--ablation`, `--epochs` (làm đầu tiên) | 45 phút | smoke test chạy 2 lần liên tiếp → lần 2 bỏ qua hết |
| [ ] | **C8. Baseline 15 run** trên Colab GPU | ~1 giờ GPU | 15 dòng `completed` |
| [ ] | **C7. `aggregate.py`**: `check_log` + mean ± std, CI95, recall theo lớp, CM cộng dồn, đặt cạnh số paper (**viết trong lúc C8 chạy**) | 1,5 giờ | xoá thử 1 dòng log → script phải báo lỗi |
| [ ] | **C9a. Ablation noBN**, chạy ngay sau baseline | ~1 giờ GPU | 15 dòng, `run_id` có `nobn` |
| [ ] | **C9b. ES_valacc**, chạy tối | ~1 giờ GPU | 15 dòng |

**Cổng ngày 2:** `aggregate.py` in được bảng baseline; kết quả gần pipeline cũ (93.21 / 93.25 / 92.91).
Nếu lệch > 1 điểm → dừng lại tìm bug trước khi chạy tiếp.

## Ngày 3 — Thứ Tư 21/10: noES + báo cáo

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **C9c. noES** (lâu nhất, chạy từ sáng) | ~1,5–2 giờ GPU | `epochs_ran == 500` ở cả 15 dòng |
| [ ] | **C10a. Notebook 02** (làm trong lúc GPU chạy): bảng 4 group, learning curve, CM | 1 giờ | chạy từ đầu đến cuối không lỗi, có output |
| [ ] | **C10b. README**: mục tiêu, cách chạy, md5 dữ liệu, bảng kết quả, khác biệt với paper | 1 giờ | người khác đọc README là chạy lại được |
| [ ] | **C10c. Push + gửi thầy**: tick các ô trong `advisor_feedback.md`, gửi link repo + notebook có output | 30 phút | |
| [ ] | *(thêm)* **C11-lite**: RBF-SVM + XGBoost trên cùng 5 split, ghi qua cùng hàm log | 1 giờ, chỉ cần CPU | |

**Cổng ngày 3:** đủ 60 dòng log, `check_log` qua, notebook 02 có output, repo đã push.

---

## Rủi ro và cách xử lý

- **Colab ngắt giữa chừng:** `run.py` bỏ qua run đã completed → chỉ cần chạy lại đúng lệnh. Giữ tab mở trong lúc chạy.
- **Hết quota GPU:** chạy phần còn lại bằng tài khoản Google khác, hoặc chuyển noES sang ngày 4. noES là ablation, không chặn báo cáo.
- **Ngày 1 bị trễ:** C5 dời sang sáng ngày 2. Khi đó baseline vẫn phải bắt đầu chạy trước 11h ngày 2.

## Giai đoạn 3 — sau khi thầy duyệt baseline

- [ ] **C11. Baseline mạnh đầy đủ**: RBF-SVM, XGBoost, CatBoost, MLP / residual MLP, cùng ngân sách tuning
- [ ] **C12. Hoán vị đặc trưng**: ResNet1, so nhóm hoán vị với nhóm seed; đối chứng bằng MLP và kernel_size = 1
- [ ] **C13. Global classifier** không phụ thuộc thứ tự (residual MLP / TabM; hướng hypergraph)
- [ ] **C14. Specialist SIRA–DERMASON** + calibration + ablation
- [ ] **C15. Thêm một bộ dữ liệu nông sản thứ hai**
