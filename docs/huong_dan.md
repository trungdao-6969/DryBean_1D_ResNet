# Hướng dẫn: vai trò từng file và các lệnh

## 1. Luồng chạy của một run

```
configs/baseline.yaml ─┐
configs/ablation_*.yaml ┼─► config.load_config ─► cfg (dict) ─► config.make_run_id ─► run_id
run.py: model, split ───┘                          │
                                                   ▼
            data.load_split(cfg) ◄── splits/split_k.npz + data/Dry_Bean_Dataset.xlsx
                   │  X_tr, X_va, X_te (đã Min-Max theo train), y, classes
                   ▼
            models.build_model(cfg) ─► Keras model
                   ▼
            train.train_one_run(cfg): fit ─► best.keras ─► predict test ─► metrics
                   ▼
            results/runs/<run_id>/{config.yaml, history.csv, curve.png, cm.npy, best.keras}
            results/runs_log.csv  (+1 dòng, ghi qua append_log)
                   ▼
            aggregate.py: check_log ─► results/summary/*.csv
```

## 2. Từng file làm gì

| File | Vai trò | Giải quyết feedback nào |
|---|---|---|
| `configs/baseline.yaml` | Mọi tham số của thí nghiệm: dữ liệu, thứ tự 16 cột, split, kiến trúc (BN, L2), huấn luyện, callbacks, siêu tham số riêng theo Table 2 | 1.6, 2.1 |
| `configs/ablation_*.yaml` | Chỉ ghi **phần khác** baseline. Sửa baseline thì ablation tự cập nhật theo | 1.4, 1.5, 2.5 |
| `src/config.py` | `load_config`: ghép baseline ← ablation ← overrides (đệ quy), chép `per_model[model]` vào `train`. `make_run_id`: tên run + 6 ký tự md5 của toàn bộ cfg, nên hai config khác nhau không thể trùng tên | 2.1, 2.2 |
| `src/data.py` | `make_splits`: 5 split phân tầng (train/test rồi train/val), lưu chỉ số + md5 dữ liệu. `load_split`: đọc split, kiểm tra md5, fit MinMax **chỉ trên train** | 1.2, 1.6 |
| `src/models.py` | ResNet1/2/3. Phần cố định (filters, Dropout, Dense) nằm trong `ARCH`; phần có thể đổi (`shortcut_bn`, L2, kernel) lấy từ cfg. L2 chỉ ở Dense | 1.1, 1.5 |
| `src/train.py` | `train_one_run`: seed → data → model → fit → **load best.keras** → test → lưu artefact → `append_log`. `append_log` là hàm duy nhất ghi log | 1.3, 1.4, 2.3, 2.4 |
| `run.py` | Lặp 3 model × 5 split, bỏ qua run đã `completed`, ghi `failed` nếu lỗi. `--smoke` để chạy thử | 2.2 |
| `aggregate.py` | `check_log` (đủ 15 run/group, không trùng, cùng md5, noES đủ epoch) rồi mới tính mean ± std, CI95 (phân phối t), recall theo lớp, CM cộng dồn, paired t-test | 2.6, 2.7 |
| `splits/*.npz` | Chỉ số train/val/test của 5 split. **Commit lên GitHub** để mọi model sau này (SVM, XGBoost, TabM...) dùng đúng các split này | 1.6 |
| `results/runs_log.csv` | Mỗi run một dòng: cấu hình, số epoch, epoch tốt nhất, metric macro + weighted, recall theo lớp, md5, phiên bản TF, đường dẫn checkpoint | 1.4, 2.3 |
| `notebooks/01_run_colab.ipynb` | Chỉ gọi `run.py` và `aggregate.py`, không chứa code mô hình | 2.1 |
| `notebooks/02_results.ipynb` | Bảng, biểu đồ, learning curve, confusion matrix cho báo cáo | 1.6 |
| `.gitignore` | Chặn dữ liệu, `*.keras`, file rác, `results_smoke/` | |
| `requirements.txt` | Thư viện cần cài | |

### Những điểm toán / thống kê nên nắm

- **Phân tầng 2 lần**: `train_test_split(..., stratify=y)` giữ tỉ lệ 7 lớp ở cả 3 tập (lệch tối đa ~0.04%).
  Kích thước: test = ⌈0.2·13611⌉ = 2723; val = ⌈0.1·10888⌉ = 1089; train = 9799 → ResNet1 có ⌈9799/1024⌉ = 10 bước/epoch.
- **Không rò rỉ khi chuẩn hoá**: x' = (x − min_tr)/(max_tr − min_tr). Vì thế X_tr nằm đúng trong [0,1], còn val/test có thể ra ngoài một chút. Nếu val/test cũng nằm gọn trong [0,1] thì scaler đã bị fit trên cả tập.
- **Loss**: `sparse_categorical_crossentropy` với nhãn nguyên tương đương categorical CE với one-hot:
  L = −(1/B)·Σ log p_{y_i} + 0.01·‖W_dense‖²  (L2 chỉ ở Dense).
- **Chọn model**: `ModelCheckpoint(save_best_only)` lưu epoch có val_loss nhỏ nhất, rồi load lại file đó trước khi đánh giá test. Không dùng `restore_best_weights`.
- **Weighted recall = accuracy**: Σ_c (n_c/N)·(TP_c/n_c) = Σ TP_c / N. Dùng đẳng thức này để kiểm tra bảng của paper.
- **CI 95% với n = 5**: mean ± t_{0.975, 4}·s/√5, với t ≈ 2.776 (không dùng 1.96 vì n quá nhỏ). s tính với ddof = 1.
  Paper báo std với ddof = 0, nên khi so phải nhớ s_paper = s·√(9/10).
- **Paired t-test**: mọi group dùng cùng 5 split, nên so ablation với baseline theo từng split (d_k = B_k − A_k), t = mean(d)/(s_d/√5).
  Ghép cặp như vậy loại bỏ được phần biến thiên do split.

## 3. Thiết lập GitHub + Colab (làm một lần, ~15 phút)

**Bước 1: tạo repo trống.** Vào github.com → nút **New** → Repository name `drybean-1d-resnet` → **không** tick "Add a README"
(repo đã có lịch sử commit sẵn, nếu tạo thêm README trên GitHub thì lần push đầu sẽ bị xung đột) → Create.

**Bước 2: tạo token để Colab được push.** github.com → ảnh đại diện → **Settings** → **Developer settings** →
**Personal access tokens** → **Fine-grained tokens** → **Generate new token**:
- Token name: `colab-drybean`, Expiration: 90 days
- Repository access: **Only select repositories** → chọn `drybean-1d-resnet`
- Permissions → Repository permissions → **Contents: Read and write**
- Generate → **copy token ngay** (chỉ hiện một lần)

**Bước 3: lưu token vào Colab.** Mở Colab bất kỳ → biểu tượng 🔑 (Secrets) ở thanh trái → Add new secret:
Name `GH_TOKEN`, Value = token vừa copy → bật **Notebook access**. Secret gắn với tài khoản Google, dùng được cho mọi notebook.

**Bước 4: đưa repo lên Drive.** Tải `drybean-1d-resnet.zip` lên **MyDrive** (thư mục gốc). Mở một notebook Colab mới, chạy:

```python
from google.colab import drive; drive.mount('/content/drive')
!cd /content/drive/MyDrive && unzip -q drybean-1d-resnet.zip && ls drybean-1d-resnet
```

**Bước 5: push lần đầu.** Vẫn trong notebook đó (sửa `YOUR_GITHUB_USERNAME`):

```python
import subprocess
from google.colab import userdata
token = userdata.get('GH_TOKEN')
r = subprocess.run(f'cd /content/drive/MyDrive/drybean-1d-resnet && '
                   f'git push https://{token}@github.com/YOUR_GITHUB_USERNAME/drybean-1d-resnet.git main',
                   shell=True, capture_output=True, text=True)
print((r.stdout + r.stderr).replace(token, '***'))
```

Tải lại trang GitHub, bạn sẽ thấy code và lịch sử commit.

**Bước 6: chạy thí nghiệm.** Trên Drive, vào `MyDrive/drybean-1d-resnet/notebooks/` → chuột phải `01_run_colab.ipynb` →
Open with → Google Colaboratory. Từ giờ luôn mở notebook từ đây, để output được lưu vào repo.

## 4. Các lệnh dùng trong dự án

### Python

| Lệnh | Ý nghĩa |
|---|---|
| `python -u run.py` | Chạy baseline 15 run. `-u`: in ra ngay (không đệm), để Colab hiện tiến độ theo thời gian thực |
| `python -u run.py --smoke --models resnet1 --splits 0` | Chạy thử 1 run, 3 epoch, ghi vào `results_smoke/` |
| `python -u run.py --ablation configs/ablation_noBN.yaml` | Ghép baseline với ablation rồi chạy 15 run |
| `python run.py --models resnet2 --splits 3 4` | Chỉ chạy một phần lưới |
| `python aggregate.py --groups baseline` | Kiểm tra log và lập bảng cho một group |
| `pip install -r requirements.txt` | Cài thư viện (Colab đã có sẵn, không cần chạy) |

Trong notebook, dấu `!` ở đầu dòng nghĩa là chạy lệnh shell; `%cd` đổi thư mục làm việc và giữ nguyên cho các ô sau.

### Git

| Lệnh | Ý nghĩa |
|---|---|
| `git status` | Xem file nào đã thay đổi / chưa được theo dõi |
| `git add <file/thư mục>` | Đưa thay đổi vào "vùng chuẩn bị" cho commit tiếp theo |
| `git commit -m "msg"` | Chụp một bản lưu (snapshot) các thay đổi đã add, kèm mô tả |
| `git log --oneline` | Xem lịch sử commit, mỗi commit một dòng |
| `git push <url> main` | Đẩy các commit ở máy (ở đây là Drive) lên GitHub, nhánh `main` |
| `git pull` | Kéo commit mới từ GitHub về (khi sửa code trên GitHub hoặc ở nơi khác) |
| `git diff` | Xem chi tiết dòng nào đã đổi trong các file chưa add |

Quy ước commit: một mục checklist một commit, message bắt đầu bằng mã mục, ví dụ `C8: baseline 15 run`.

## 5. Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `md5 không khớp` | File dữ liệu khác file đã dùng để tạo split. Dùng đúng file cũ, hoặc xoá `splits/*.npz` để tạo lại (khi đó mọi run cũ coi như bỏ) |
| `File dữ liệu thiếu cột` | File CSV/XLSX khác bản UCI, hoặc tên cột khác. So với `feature_order` trong YAML |
| `[FAIL ]` | Xem traceback ngay phía trên. Run lỗi được ghi `status=failed`; sửa xong chạy lại lệnh cũ, run đó sẽ được chạy lại |
| Push báo `403` / `Authentication failed` | Token sai/hết hạn, hoặc chưa cấp quyền Contents: Read and write cho đúng repo |
| Push báo `rejected (fetch first)` | Trên GitHub có commit mà máy chưa có (vd đã sửa file trên web). Chạy `git pull` trước rồi push lại |
| Val accuracy ~12% suốt vài chục epoch đầu (ResNet1) | Hiện tượng đã thấy ở pipeline cũ: chỉ có 10 bước/epoch nên BatchNorm hội tụ chậm. Thường nhảy lên sau epoch ~90. Cần ghi vào báo cáo, không phải lỗi code |
