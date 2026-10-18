# Kế hoạch thí nghiệm — baseline 3 ResNet (bản chốt)

## 1. Giao thức chung (áp dụng cho MỌI run, kể cả các model so sánh sau này)

| Mục | Giá trị chốt | So với paper |
|---|---|---|
| Dữ liệu | `Dry_Bean_Dataset.xlsx` (UCI id 602), 13 611 × 16, 7 lớp. Ghi **md5** của file vào README và log | giống |
| Thứ tự đặc trưng | đúng thứ tự cột trong file (liệt kê đủ 16 tên trong `baseline.yaml`) | paper không nêu |
| Số split | **5** (`split_id` 0–4) | paper: 10 lần hold-out → khác, theo yêu cầu thầy |
| Chia train/test | 80/20, phân tầng, `random_state = split_id` | giống |
| Validation | **10% của phần train**, phân tầng | paper không nêu → tự chọn |
| Lưu split | `splits/split_<k>.npz` chứa `train_idx, val_idx, test_idx`; mọi model đọc lại file này | – |
| Chuẩn hoá | MinMaxScaler **fit trên train**, transform val/test | giống |
| Seed khởi tạo | `seed = 0` cố định cho mọi run; bật `enable_op_determinism()` | paper không nêu |
| Loss | categorical cross-entropy (+ L2 của Dense) | paper không nêu → giả định |
| Chọn model | `ModelCheckpoint` lưu `best.keras` theo monitor; **load lại best.keras rồi mới evaluate test** | – |

## 2. Kiến trúc (cố định trong code, không đưa vào YAML)

- Stem: Conv1D(64, 3, same, glorot_uniform) → BN → ReLU
- Block(f): Conv1D(f,3) → BN → ReLU → Conv1D(f,3) → BN; shortcut = Conv1D(f,1) [→ BN nếu `shortcut_bn`]; Add → ReLU
- Head: Flatten → Dense(units, ReLU, **L2 = 0.01**) → Dropout(0.5) → Dense(7, softmax)

| | Blocks | Dropout sau block | Dense | Optimizer | LR | Batch | Params mong đợi |
|---|---|---|---|---|---|---|---|
| resnet1 | 128-256-384 | – | 128 | Nadam | 0.0002 | 1024 | 2 044 679 (paper 2.04M) |
| resnet2 | 128-256-512 | 0.5 sau block 3 | 256 | Nadam | 0.0001 | 512 | 3 833 479 (paper 3.83M) |
| resnet3 | 128-256-384-512 | 0.3 sau block 3 và 4 | 128 | Adam | 0.0001 | 512 | 3 887 367 (paper 3.89M) |

(Số params của bản `shortcut_bn = true`, đã kiểm tra bằng `model.count_params()`. Bản noBN: 2 041 607 / 3 829 895 / 3 882 247.
Pipeline ngày 13/10 cho 2 038 407 / 3 826 183 / 3 876 999, tức ResNet3 = 3.88M, lệch so với paper 3.89M. Bản mới khớp cả 3 số
của paper sau khi làm tròn. Vì kiến trúc có khác chút ít, kết quả cũ chỉ dùng để đối chiếu.)

## 3. Các cấu hình sẽ chạy

| Group (file YAML) | Khác baseline ở đâu | Câu hỏi nó trả lời | Số run |
|---|---|---|---:|
| **baseline** (`baseline.yaml`) | – : shortcut Conv1×1 + BN, ES val_loss, patience 100, max 500 epoch, checkpoint val_loss | Kết quả chính để so paper | 3 × 5 = 15 |
| **ablation_noBN** | `shortcut_bn: false` | BN ở projection có ảnh hưởng không? (feedback 1.5) | 15 |
| **ablation_noES** | ES tắt → chạy đủ 500 epoch; checkpoint vẫn theo val_loss; log thêm `test_acc_last` | Early stopping có làm mất kết quả không? (feedback 1.4) | 15 |
| **ablation_ES_valacc** | ES + checkpoint theo `val_accuracy` | Chọn theo val_acc có tốt hơn val_loss? (feedback 1.4) | 15 |
| | | **Tổng** | **60** |

Thứ tự chạy: baseline trước (phải đạt kiểm tra) → noBN → ES_valacc → noES (lâu nhất).
Kiểm tra chéo: baseline mới phải cho kết quả gần với pipeline cũ (93.21 / 93.25 / 92.91). Lệch nhiều → có bug.

## 4. Những gì mỗi run phải lưu

`results/runs/<run_id>/`: `config.yaml` (config đầy đủ sau khi merge), `history.csv`, `curve.png`,
`cm.npy` (confusion matrix test), `best.keras` (không commit).

Một dòng trong `results/runs_log.csv` (chỉ ghi bằng một hàm duy nhất):

| Nhóm cột | Cột |
|---|---|
| Định danh | `run_id, group, model, split_id, seed, shortcut_bn, es_monitor, status, timestamp, data_md5` |
| Mô hình | `n_params` |
| Huấn luyện | `epochs_ran, best_epoch_val_loss, best_epoch_val_acc, train_time_s, ckpt_path` |
| Test (model tốt nhất) | `test_acc, test_f1_macro, test_f1_weighted, test_prec_macro, test_prec_weighted, test_rec_macro, test_rec_weighted` |
| Theo lớp | `recall_<Lớp>` × 7 |
| Chỉ noES | `test_acc_last` (weights ở epoch cuối) |

## 5. Bảng tổng hợp (`aggregate.py`)

1. Kiểm tra log: mỗi group đủ 3 model × 5 split, không trùng `run_id`, tất cả `completed`,
   group noES có `epochs_ran == 500`. Sai → báo lỗi, không tính.
2. Mỗi group × model: mean ± std (ddof = 1) và CI 95% = mean ± 2.776 · s/√5 cho
   accuracy, macro-F1, weighted-F1, weighted-P, weighted-R.
3. Recall từng lớp (mean qua 5 split) và confusion matrix cộng dồn 5 split.
4. Bảng đặt cạnh số paper (`paper_reference.md`) — so bằng **weighted**.
5. Ghi ra `results/summary/*.csv`.

## 6. Cấu trúc repo

```
drybean-1d-resnet/
├── README.md               mục tiêu, cách chạy, bảng kết quả, khác biệt so với paper
├── requirements.txt
├── .gitignore              data/, *.keras, __pycache__/, .ipynb_checkpoints/
├── configs/
│   ├── baseline.yaml       cấu hình đầy đủ
│   ├── ablation_noBN.yaml  chỉ ghi phần khác baseline
│   ├── ablation_noES.yaml
│   └── ablation_ES_valacc.yaml
├── data/                   (không commit) Dry_Bean_Dataset.xlsx
├── splits/                 split_0.npz … split_4.npz   (commit)
├── src/
│   ├── __init__.py
│   ├── config.py           load_config (merge baseline + ablation), make_run_id
│   ├── data.py             make_splits, load_split
│   ├── models.py           residual_block, build_model(cfg)
│   └── train.py            train_one_run(cfg) + metrics + append_log (hàm log duy nhất)
├── run.py                  chạy lưới model × split, bỏ qua run đã completed
├── aggregate.py            check_log + bảng tổng hợp
├── notebooks/
│   ├── 01_run_colab.ipynb  clone repo, mount Drive, gọi run.py   (commit KÈM output)
│   └── 02_results.ipynb    gọi aggregate.py, vẽ bảng/biểu đồ     (commit KÈM output)
├── results/
│   ├── runs_log.csv        (commit)
│   ├── summary/            (commit)
│   └── runs/<run_id>/      (commit trừ best.keras)
└── docs/
    ├── paper_reference.md
    ├── advisor_feedback.md
    ├── experiment_plan.md
    ├── checklist.md
    └── huong_dan.md           vai trò từng file, lệnh git/python, thiết lập GitHub + Colab
```

So với pipeline 13/10: gộp `metrics.py` + `run_log.py` vào `train.py`, đổi `run_experiments.py` → `run.py`.
Còn 5 file code chính + 2 script + 2 notebook.
