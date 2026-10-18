# Paper reference — số liệu để đối chiếu

File này gom mọi con số cần so sánh, để không phải mở lại paper. Số trong ngoặc vuông `[...]` là
**ghi chú của mình khi kiểm tra lại**, không phải số trong paper.

Nguồn:
- **[P]** Phan, T.-T.-H. (2026). *Lightweight 1D-ResNet architecture for improving dry bean variety classification.* Optimization Letters. doi:10.1007/s11590-026-02324-7
- **[K]** Koklu, M. & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture 174, 105507. doi:10.1016/j.compag.2020.105507
- **[KH]** Khan, M. S. et al. (2023). *Comparison of multiclass classification techniques using dry bean dataset.* Int. J. Cognitive Computing in Engineering 4, 6–20. doi:10.1016/j.ijcce.2023.01.002

---

## 1. Dữ liệu (Dry Bean, UCI id 602) — [K] Table 1

| Lớp | Số mẫu | Train 80% [P] | Test 20% [P] |
|---|---:|---:|---:|
| Seker | 2027 | 1621 | 406 |
| Barbunya | 1322 | 1057 | 265 |
| Bombay | 522 | 417 | 105 |
| Cali | 1630 | 1304 | 326 |
| Horoz | 1928 | 1542 | 386 |
| Sira | 2636 | 2108 | 528 |
| Dermason | 3546 | 2836 | 710 |
| **Tổng** | **13 611** | **10 885** | **2 726** |

16 đặc trưng = 12 kích thước + 4 hình dạng (đơn vị pixel). Tên cột trong file UCI:
`Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea,
EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4`
(chú ý `AspectRation` và `roundness` viết đúng như vậy trong file).

Công thức [K] (L = Major, l = Minor, A = Area, P = Perimeter, C = ConvexArea):
AspectRatio = L/l · Ed = √(4A/π) · Extent = A/A_bbox · Solidity = A/C · Roundness = 4πA/P² ·
Compactness = Ed/L · SF1 = L/A · SF2 = l/A · SF3 = A/(π·L²/4) · SF4 = A/(π·L·l/4).
[Phần lớn đặc trưng là hàm tất định của vài đại lượng gốc → liên quan đến ablation "loại biến phụ thuộc tất định".]

---

## 2. Kiến trúc 3 mô hình [P] §2.2

Chung cho cả 3: input `(batch, 16, 1)`; **stem** Conv1D(64, k=3, same, GlorotUniform) → BN → ReLU.

**Residual block** (Fig. 3, ResNet v1):
`x → Conv1D(f,3) → BN → ReLU → Conv1D(f,3) → BN → (+ shortcut) → ReLU`.
[Paper không nói shortcut làm gì khi số kênh đổi. Để khớp số tham số phải có projection Conv1D 1×1
(+ BN) ở mọi block, vì block nào cũng đổi số kênh.]

| | 1D-ResNet1 | 1D-ResNet2 | 1D-ResNet3 |
|---|---|---|---|
| Residual blocks | 3: 128-256-384 | 3: 128-256-**512** | **4**: 128-256-384-512 |
| Dropout sau block | — | 0.5 sau block 3 | 0.3 sau block 3 và block 4 |
| Flatten | 16×384 = 6144 | 16×512 = 8192 | 16×512 = 8192 |
| Dense (ReLU, L2 = 0.01) | 128 | **256** | 128 |
| Dropout sau Dense | 0.5 | 0.5 | 0.5 |
| Output | Dense(7, softmax) | Dense(7, softmax) | Dense(7, softmax) |

L2 = 0.01 **chỉ ở lớp Dense ẩn** (mô tả chi tiết của ResNet1). Conv1D không có L2.

### Siêu tham số — [P] Table 2

| | ResNet1 | ResNet2 | ResNet3 |
|---|---|---|---|
| Initialization | GlorotUniform | GlorotUniform | GlorotUniform |
| Activation | ReLU | ReLU | ReLU |
| L2 | 0.01 | 0.01 | 0.01 |
| Learning rate | 2×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |
| Optimizer | Nadam | Nadam | Adam |
| Epochs | 500 | 500 | 500 |
| Patience (early stopping) | 100 | 100 | 100 |
| Batch size | 1024 | 512 | 512 |
| # blocks | 3 | 3 | 4 |

Không ghi trong paper: loss function, monitor của early stopping, cách tạo tập validation, seed,
shortcut có BN hay không.

### Độ phức tạp — [P] Table 7

| | Params | FLOPs (M) | Training time (s) | Epoch có val tốt nhất (Fig. 6–8) |
|---|---|---:|---:|---:|
| ResNet1 | 2.04M | 41.57 | 116.07 | ≈ 132 |
| ResNet2 | 3.83M | 59.42 | 250.46 | ≈ 318 |
| ResNet3 | 3.89M | 92.51 | 304.23 | ≈ 281 |

[Số tham số pipeline cũ của mình (shortcut Conv1×1+BN): 2 038 407 / 3 826 183 / 3 876 999 → khớp.]

---

## 3. Kết quả [P] — 10 lần hold-out 80/20 phân tầng (%)

### Tóm tắt (mean ± std)

| | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **ResNet1** (Table 3) | 95.54 ± 0.45 | 95.55 ± 0.47 | 95.55 ± 0.46 | **95.70 ± 0.51** |
| **ResNet2** (Table 5) | 93.92 ± 0.54 | 93.79 ± 0.57 | 93.81 ± 0.58 | **93.79 ± 0.57** |
| **ResNet3** (Table 6) | 93.74 ± 0.48 | 93.71 ± 0.48 | 93.70 ± 0.47 | **93.71 ± 0.48** |

Wilcoxon ResNet1 vs 1DCNN: p = 0.00195.

### Từng lần chạy

ResNet1 — Table 3

| Iter | P | R | F1 | Acc |
|---|---|---|---|---|
| 1 | 95.04 | 94.97 | 94.97 | 95.97 |
| 2 | 96.04 | 95.99 | 96.00 | 95.99 |
| 3 | 95.62 | 95.56 | 95.53 | 96.46 |
| 4 | 95.19 | 95.15 | 95.15 | 95.15 |
| 5 | 95.45 | 95.45 | 95.44 | 95.45 |
| 6 | 95.30 | 95.23 | 95.23 | 95.23 |
| 7 | 95.80 | 95.78 | 95.78 | 95.78 |
| 8 | 95.71 | 95.70 | 95.71 | 95.70 |
| 9 | 94.82 | 94.77 | 94.78 | 94.79 |
| 10 | 96.39 | 96.40 | 96.38 | 96.40 |

ResNet2 — Table 5

| Iter | P | R | F1 | Acc |
|---|---|---|---|---|
| 1 | 92.74 | 92.73 | 92.73 | 92.73 |
| 2 | 94.38 | 94.34 | 94.34 | 94.34 |
| 3 | 93.76 | 93.76 | 93.74 | 93.76 |
| 4 | 93.96 | 93.90 | 93.92 | 93.90 |
| 5 | 94.61 | 94.42 | 94.46 | 94.42 |
| 6 | 93.07 | 92.95 | 92.91 | 92.95 |
| 7 | 94.05 | 94.00 | 94.00 | 94.00 |
| 8 | 94.24 | 94.20 | 94.20 | 94.20 |
| 9 | 93.45 | 93.02 | 93.08 | 93.02 |
| 10 | 94.09 | 94.09 | 94.09 | 94.09 |

ResNet3 — Table 6

| Iter | P | R | F1 | Acc |
|---|---|---|---|---|
| 1 | 92.77 | 92.69 | 92.71 | 92.69 |
| 2 | 93.33 | 93.28 | 93.28 | 93.28 |
| 3 | 93.75 | 93.72 | 93.71 | 93.72 |
| 4 | 94.07 | 94.05 | 94.02 | 94.05 |
| 5 | 93.43 | 93.39 | 93.40 | 93.39 |
| 6 | 93.42 | 93.43 | 93.41 | 93.43 |
| 7 | 94.36 | 94.31 | 94.29 | 94.31 |
| 8 | 94.13 | 94.09 | 94.05 | 94.09 |
| 9 | 93.99 | 93.97 | 93.98 | 93.97 |
| 10 | 94.20 | 94.18 | 94.17 | 94.18 |

### Theo lớp, ResNet1 lần 2 — Table 4

| Lớp | P | R | F1 | "Accuracy" |
|---|---|---|---|---|
| Barbunya | 97.86 | 96.48 | 97.16 | 97.86 |
| Bombay | 100 | 100 | 100 | 100 |
| Cali | 97.59 | 97.59 | 97.59 | 97.60 |
| Dermason | 95.85 | 92.96 | 94.38 | 95.74 |
| Horoz | 98.43 | 97.91 | 98.17 | 98.42 |
| Seker | 96.48 | 97.46 | 96.97 | 96.47 |
| Sira | 91.58 | 95.51 | 93.50 | 91.59 |

### Các điểm không nhất quán trong [P] (đã kiểm tra lại bằng code)

1. Với weighted average, **recall luôn bằng accuracy**. Table 3 lần 1 (94.97 vs 95.97) và lần 3
   (95.56 vs 96.46) lệch nhau đúng ~1 điểm. Nếu lấy cột recall làm accuracy thì trung bình ResNet1
   là **95.50**, không phải 95.70. Ở ResNet2/3 recall = accuracy ở mọi dòng → paper nhiều khả năng dùng weighted.
2. Std trong bảng là std **quần thể** (ddof = 0): ResNet1 acc tính lại 0.51 (ddof=0) / 0.54 (ddof=1).
3. ResNet3: văn bản ghi ±0.53/0.55/0.54/0.55, Table 6 ghi ±0.48/0.48/0.47/0.48 (bảng đúng với ddof=0).
4. Mean recall ResNet2 tính lại là 93.74 (bảng ghi 93.79); mean precision 93.84 (bảng 93.92).
5. Table 4: cột "Accuracy" gần như bằng cột Precision, không phải accuracy theo lớp.
6. Paper nói có tập validation để chọn model (Fig. 1, §2.1) nhưng §3.1 chỉ mô tả chia train/test 80/20.
   Không rõ validation lấy từ đâu → giả thuyết: có thể đã dùng test làm validation.

---

## 4. So sánh trong [P] — Fig. 9 (accuracy %)

| Phương pháp | Nguồn | Accuracy |
|---|---|---:|
| SVM | Koklu & Ozkan 2020 | 93.13 |
| DBANN2 | Hasan et al. 2021 | 93.44 (F1 94.57) |
| XGBoost | Khan et al. 2023 | 93.00 |
| CatBoost (random undersampling + Box–Cox) | Krishnan et al. (chỉ nhắc trong phần related work) | 93.8 |
| 1DCNN (4 Conv1D + 2 Dense) | Phan et al. 2024 | 94.79 |
| **1D-ResNet1** | [P] | **95.70** |
| 1D-ResNet2 | [P] | 93.79 |
| 1D-ResNet3 | [P] | 93.71 |

[Các con số này đến từ giao thức đánh giá khác nhau (10-fold CV, hold-out 80/20, có/không ADASYN…)
nên **không so trực tiếp được** với nhau. Muốn so công bằng phải chạy lại trên cùng 5 split của mình.]

---

## 5. Koklu & Ozkan 2020 [K]

**Giao thức:** 10-fold cross-validation trên toàn bộ 13 611 mẫu, MATLAB. Không có tập test riêng.

**Cấu hình mô hình:**
- MLP: 2 lớp ẩn 12-3 (paper ghi kiến trúc 17-12-3-7), sigmoid ở lớp ẩn và lớp ra, lr 0.3, tối đa 500 epoch, min gradient 1e-5, goal 1e-3
- SVM: kernel bậc 3 (cubic polynomial), one-vs-one, box constraint = 1
- DT: tối đa 16 split, tiêu chí Gini
- kNN: k = 10, khoảng cách Euclid, trọng số bằng nhau

**Table 12 (%):**

| | MLP | SVM | DT | kNN |
|---|---|---|---|---|
| Accuracy | 91.73 | **93.13** | 87.92 | 92.52 |
| Error rate | 8.27 | 6.87 | 12.08 | 7.48 |
| Precision | 93.11 | 94.45 | 89.19 | 93.93 |
| Recall | 92.68 | 94.03 | 87.93 | 93.59 |
| Specificity | 98.53 | 98.77 | 97.91 | 98.67 |
| F1 | 92.88 | 94.23 | 88.29 | 93.75 |

[Precision/Recall/F1 ở đây là **macro** (Table 6 của [K] định nghĩa trung bình theo lớp), khác cách
[P] báo cáo. Abstract của [K] liệt kê nhầm thứ tự DT/kNN; bảng 12 mới đúng.]

**Accuracy theo lớp (= recall lớp), Fig. 10 (%):**

| Lớp | MLP | SVM | DT | kNN |
|---|---|---|---|---|
| Barbunya | 89.56 | 92.36 | 84.04 | 90.70 |
| Bombay | 99.23 | 100.00 | 99.43 | 100.00 |
| Cali | 93.87 | 95.03 | 91.17 | 94.85 |
| Dermason | 92.87 | 94.36 | 92.19 | 92.05 |
| Horoz | 94.87 | 94.92 | 94.35 | 95.64 |
| Seker | 93.69 | 94.67 | 93.19 | 94.72 |
| Sira | 84.67 | 86.84 | 84.26 | 87.18 |

**Confusion matrix SVM (Table 9, cộng dồn 10 fold; hàng = thật, cột = dự đoán):**

| | Barbunya | Bombay | Cali | Dermason | Horoz | Seker | Sira |
|---|---|---|---|---|---|---|---|
| Barbunya | 1221 | 1 | 60 | 0 | 6 | 11 | 23 |
| Bombay | 0 | 522 | 0 | 0 | 0 | 0 | 0 |
| Cali | 42 | 0 | 1549 | 0 | 24 | 4 | 11 |
| Dermason | 1 | 0 | 0 | 3346 | 4 | 38 | 157 |
| Horoz | 6 | 0 | 27 | 18 | 1830 | 0 | 47 |
| Seker | 9 | 0 | 1 | 49 | 1 | 1919 | 48 |
| Sira | 9 | 0 | 5 | 278 | 33 | 22 | 2289 |

[Cặp nhầm lớn nhất: Sira→Dermason 278, Dermason→Sira 157 → đây là lý do thầy gợi ý specialist SIRA–DERMASON.
Tính lại từ ma trận: accuracy 93.13 khớp.]

---

## 6. Khan et al. 2023 [KH]

**Giao thức (theo bài):** lọc outlier bằng IQR → (tuỳ chọn) ADASYN cân bằng lớp → chia train/test 80/20
→ 8 classifier, Python. Không nêu rõ seed, số lần lặp hay tuning chi tiết.

**Table 6 (%):**

| | Không ADASYN | | | | Có ADASYN | | | |
|---|---|---|---|---|---|---|---|---|
| Model | ACC | AUC | F1 | Kappa | ACC | AUC | F1 | Kappa |
| LR | 91.0 | 99.35 | 91.0 | 89.14 | 83.0 | 98.03 | 83.0 | 80.60 |
| KNN | 89.0 | 96.76 | 89.0 | 86.92 | 95.0 | 98.52 | 95.0 | 92.77 |
| DT | 90.0 | 94.85 | 90.0 | 87.33 | 90.0 | 94.38 | 90.0 | 88.83 |
| RF | 92.0 | 99.40 | 92.0 | 90.18 | 94.0 | 99.60 | 94.0 | 92.62 |
| SVM | 92.0 | 99.58 | 92.0 | 90.67 | 86.0 | 98.67 | 86.0 | 84.03 |
| NB | 89.0 | 99.18 | 89.0 | 86.94 | 79.0 | 97.16 | 79.0 | 75.87 |
| **XGB** | **93.0** | 99.60 | 93.0 | 90.92 | **95.4** | 99.64 | 94.0 | 93.10 |
| MLP | 91.0 | 99.43 | 91.0 | 89.47 | 84.0 | 98.38 | 84.0 | 81.74 |

Recall theo lớp của XGB (Fig. 8, %): Seker 95, Barbunya 97, Bombay 100, Cali 97, Horoz 96, Sira 85, Dermason 99.

Đặc trưng quan trọng nhất (XGB): ShapeFactor2, MinorAxisLength, ShapeFactor1; sau đó EquivDiameter,
roundness, ConvexArea. Ít quan trọng: Extent, Solidity, Eccentricity, AspectRation.

### Cần cẩn trọng khi trích dẫn [KH] (đã tính lại từ Table 5)

1. **ADASYN áp dụng trước khi chia** (Fig. 1): test có ~710 mẫu Bombay dù Bombay chỉ có 522 mẫu thật
   → test chứa mẫu tổng hợp, kết quả "có ADASYN" bị thổi phồng. [P] cũng chỉ ra điểm này.
2. Confusion matrix XGB không ADASYN tổng 3 403 mẫu ≈ 25% của 13 611, không phải 20%. Accuracy tính
   lại từ ma trận là **92.51%**, không phải 93.0%.
3. Confusion matrix XGB có ADASYN: accuracy tính lại **94.0%** (bảng ghi 95.4%); recall Dermason chỉ 88.2%
   (Fig. 8 ghi 99%).
4. Cột FPR ≈ 34% và SE ≈ 99–100% không khớp với ACC ≈ 93% → các chỉ số phụ trong Table 6 không đáng tin.
5. Fig. 4 cho thấy sau lọc IQR, Area ≤ ~100k, trong khi Bombay có Area ~115k–254k (Fig. 5A) → lẽ ra
   gần hết Bombay bị loại, nhưng confusion matrix vẫn có 117 mẫu Bombay. Không rõ lọc IQR được áp dụng thế nào.

→ Khi báo cáo, chỉ nên dùng **XGB 93.0% (không ADASYN)** làm số tham khảo, kèm ghi chú giao thức khác.
