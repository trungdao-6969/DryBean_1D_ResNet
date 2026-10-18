# Tái lập 1D-ResNet trên Dry Bean Dataset

Tái lập và kiểm chứng ba kiến trúc trong bài báo

> Phan, T.-T.-H. (2026). *Lightweight 1D-ResNet architecture for improving dry bean variety classification.* Optimization Letters. doi:10.1007/s11590-026-02324-7

trên bộ dữ liệu [Dry Bean (UCI id 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset): 13 611 mẫu, 16 đặc trưng hình thái, 7 lớp.

Tác giả: Đào Việt Trung — Khoa Toán–Tin, Đại học Bách khoa Hà Nội. Đồ án II.

---

## Mục tiêu

1. Dựng một pipeline **tái lập được**: mọi kết quả sinh từ một file config, dùng cùng 5 split cố định, và được ghi vào một log duy nhất.
2. So kết quả của ResNet1/2/3 với số liệu trong paper bằng giao thức đánh giá sạch: validation tách từ train, scaler chỉ fit trên train, đánh giá test bằng checkpoint tốt nhất.
3. Làm ablation: BN ở projection shortcut và ba cách chọn epoch (early stopping theo val_loss, theo val_accuracy, hoặc không early stopping).

## Kết quả

> Sẽ cập nhật sau khi chạy xong (`python aggregate.py`). Accuracy tính bằng %, dạng mean ± std [CI 95%] qua 5 split.

| Model | Params | Test accuracy | Weighted F1 | Paper (accuracy) |
|---|---:|---|---|---|
| 1D-ResNet1 | 2 044 679 | | | 95.70 ± 0.51 |
| 1D-ResNet2 | 3 833 479 | | | 93.79 ± 0.57 |
| 1D-ResNet3 | 3 887 367 | | | 93.71 ± 0.48 |

Bảng đầy đủ nằm trong `results/summary/`, phân tích nằm trong `notebooks/02_results.ipynb`.

## Giao thức

| | Repo này | Paper |
|---|---|---|
| Split | 5 lần hold-out 80/20, phân tầng (`random_state = 0..4`) | 10 lần hold-out 80/20 |
| Validation | 10% của train, phân tầng | không nêu |
| Chuẩn hoá | Min-Max fit trên train | Min-Max |
| L2 = 0.01 | chỉ ở Dense ẩn | chỉ ở Dense ẩn |
| Shortcut | Conv1D 1×1 + BN (ablation: không BN) | không nêu (suy ra từ số tham số) |
| Chọn model | checkpoint có val_loss nhỏ nhất | không nêu |
| Early stopping | val_loss, patience 100, tối đa 500 epoch | patience 100, 500 epoch |
| Seed khởi tạo | 0, bật op determinism | không nêu |
| Báo cáo | accuracy, P/R/F1 macro + weighted, recall từng lớp | weighted (suy ra) |

Siêu tham số theo Table 2 của paper: ResNet1 dùng Nadam, lr 2e-4, batch 1024; ResNet2 dùng Nadam, lr 1e-4, batch 512; ResNet3 dùng Adam, lr 1e-4, batch 512.

Dữ liệu: `Dry_Bean_Dataset.xlsx` tải từ UCI, md5 = `<điền sau khi chạy notebook 01, ô 1>`.

## Cấu trúc

```
configs/      baseline.yaml (đầy đủ) + ablation_*.yaml (chỉ ghi phần khác)
src/
  config.py   ghép config, sinh run_id
  data.py     tạo/đọc split cố định, Min-Max fit trên train
  models.py   ResNet1/2/3
  train.py    huấn luyện 1 run, tính metric, ghi log
run.py        chạy lưới model × split, bỏ qua run đã xong
aggregate.py  kiểm tra log, lập bảng mean ± std, CI 95%, paired t-test
splits/       split_0..4.npz (chỉ số train/val/test + md5 dữ liệu)
results/      runs_log.csv, runs/<run_id>/ (config, history, curve, cm), summary/
notebooks/    01_run_colab (chạy), 02_results (bảng và hình)
docs/         số liệu paper, feedback, kế hoạch, checklist, hướng dẫn
```

## Cách chạy

```bash
pip install -r requirements.txt
# đặt file dữ liệu vào data/Dry_Bean_Dataset.xlsx
python run.py --smoke --models resnet1 --splits 0       # chạy thử 3 epoch
python run.py                                           # baseline: 3 model × 5 split
python run.py --ablation configs/ablation_noBN.yaml     # ablation
python aggregate.py                                     # bảng kết quả
```

Trên Colab: mở `notebooks/01_run_colab.ipynb` và chạy theo thứ tự các ô.

Tài liệu tham khảo nhanh (kiến trúc, bảng số liệu của paper, Koklu 2020, Khan 2023) nằm trong [`docs/paper_reference.md`](docs/paper_reference.md).
