# Checklist

Làm **theo thứ tự**. Mục nào qua được phần "Tự kiểm tra" thì mới sang mục sau.
Mỗi mục xong là 1 commit, message bắt đầu bằng mã mục, ví dụ `C8: baseline 15 run`.

Phạm vi: **C0 → C10** (pipeline sạch + 60 run + báo cáo cho thầy). C11-lite là phần thêm nếu còn thời gian.
Giai đoạn 3 (hoán vị, global classifier...) là nghiên cứu, chỉ làm sau khi thầy duyệt baseline.

---

## Đã xong — tối 18/10: code pipeline (đã test trên dữ liệu giả lập)

| | Mục | Đã kiểm tra |
|---|---|---|
| [x] | **C0. Khung repo**: `.gitignore`, `requirements.txt`, `docs/` | git status sạch, dữ liệu bị chặn |
| [x] | **C1. Config YAML**: `baseline.yaml` + 3 ablation | đọc được, `lr` là `float`; noBN chỉ khác `shortcut_bn` + `group` |
| [x] | **C2. `src/config.py`** | cùng cfg → cùng run_id; đổi `lr` → run_id khác |
| [x] | **C3. `src/data.py`** | 3 tập rời nhau, đủ 13 611; lệch tỉ lệ lớp ≤ 0.04%; X_tr ∈ [0,1], test ∈ [−0.07, 1.02] |
| [x] | **C4. `src/models.py`** | params 2 044 679 / 3 833 479 / 3 887 367 (khớp paper); L2 chỉ ở `dense` |
| [x] | **C5. `src/train.py`** | đủ artefact + 1 dòng log; chạy 2 lần giống hệt nhau (tất định); 20 epoch học được |
| [x] | **C6. `run.py`** | chạy lại → `[skip ]`; 15 run liền không lỗi |
| [x] | **C7. `aggregate.py`** | thiếu run → báo lỗi + exit 1; đủ run → in bảng và ghi `results/summary/` |

## Ngày 1 — Thứ Hai 19/10: GitHub + baseline + noBN

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **G. Thiết lập GitHub + Colab** (`docs/huong_dan.md` §3): tạo repo, token, secret, upload zip, push lần đầu | 20 phút | trên GitHub thấy 10 commit |
| [ ] | **Smoke test trên dữ liệu thật** (notebook 01, ô 0–3) | 10 phút | `train=9799 val=1089 test=2723`; `params=2,044,679`; ghi md5 vào README |
| [ ] | **C8. Baseline 15 run** (ô 4) | ~1 giờ GPU | 15 dòng `[done ]` |
| [ ] | *Trong lúc chạy:* **đọc hiểu code** `data.py` → `models.py` → `train.py` cùng mình | | trả lời được câu hỏi kiểm tra |
| [ ] | **Kiểm tra baseline**: `aggregate.py --groups baseline` | 10 phút | gần 93.21 / 93.25 / 92.91 (pipeline cũ); lệch > 1 điểm → dừng tìm nguyên nhân |
| [ ] | **C9a. noBN** (ô 5, lệnh đầu) | ~1 giờ GPU | 15 dòng, run_id có `nobn` |
| [ ] | Push kết quả (ô 6): `C8: baseline` / `C9a: noBN` | 5 phút | |

## Ngày 2 — Thứ Ba 20/10: ablation còn lại + báo cáo

| | Mục | Ước lượng | Tự kiểm tra |
|---|---|---|---|
| [ ] | **C9b. ES_valacc** (chạy từ sáng) | ~1 giờ GPU | 15 dòng |
| [ ] | **C9c. noES** (chạy ngay sau C9b) | ~1,5–2 giờ GPU | `epochs_ran == 500` ở cả 15 dòng |
| [ ] | **C10a. Notebook 02** có output | 30 phút | chạy hết các ô, `check_log` đạt với 4 group |
| [ ] | **C10b. README**: điền bảng kết quả, md5 | 30 phút | |
| [ ] | **C10c. Báo cáo cho thầy**: phân tích kết quả, tick `advisor_feedback.md`, push, gửi link | 2 giờ | |

## Ngày 3 — Thứ Tư 21/10: dự phòng + baseline cổ điển

| | Mục | Ước lượng |
|---|---|---|
| [ ] | Phần nào của ngày 2 chưa xong | |
| [ ] | *(thêm)* **C11-lite**: RBF-SVM + XGBoost trên cùng 5 split, ghi qua cùng `append_log` | 1–2 giờ, chỉ cần CPU |

---

## Rủi ro và cách xử lý

- **Colab ngắt giữa chừng:** chạy lại ô 0 rồi chạy lại đúng ô đang dở. Run đã xong sẽ được bỏ qua.
- **Hết quota GPU:** chuyển noES sang ngày 3. noES là ablation, không chặn báo cáo.
- **Kết quả baseline lệch nhiều so với pipeline cũ:** kiến trúc mới có thêm ~6k–10k tham số (khớp paper hơn), nên lệch vài phần mười điểm là bình thường. Lệch > 1 điểm thì gửi mình log để tìm nguyên nhân.

## Giai đoạn 3 — sau khi thầy duyệt baseline

- [ ] **C11. Baseline mạnh đầy đủ**: RBF-SVM, XGBoost, CatBoost, MLP / residual MLP, cùng ngân sách tuning
- [ ] **C12. Hoán vị đặc trưng**: ResNet1, so nhóm hoán vị với nhóm seed; đối chứng bằng MLP và kernel_size = 1
- [ ] **C13. Global classifier** không phụ thuộc thứ tự (residual MLP / TabM; hướng hypergraph)
- [ ] **C14. Specialist SIRA–DERMASON** + calibration + ablation
- [ ] **C15. Thêm một bộ dữ liệu nông sản thứ hai**
