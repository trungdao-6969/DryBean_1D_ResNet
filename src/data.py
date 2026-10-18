"""
src/data.py — đọc dữ liệu, tạo / đọc split cố định, chuẩn hoá Min-Max.

Theo feedback của thầy:
  * Chia train/test VÀ train/val đều có phân tầng (stratify).
  * Validation tách TỪ phần train; test không tham gia chọn bất cứ thứ gì.
  * MinMaxScaler chỉ fit trên train, rồi transform val và test.
  * Split lưu ra splits/split_<k>.npz -> mọi model (ResNet, SVM, XGBoost, ...)
    đọc lại đúng cùng các chỉ số này.
  * md5 của file dữ liệu được lưu trong split -> đổi file dữ liệu là báo lỗi ngay.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_table(cfg):
    """Đọc file dữ liệu -> X (n, 16) theo ĐÚNG feature_order, y (n,) số nguyên, tên lớp."""
    path = cfg["data"]["path"]
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)

    features = cfg["data"]["feature_order"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"File dữ liệu thiếu cột: {missing}")

    X = df[features].to_numpy(dtype=np.float64)
    # Mã hoá nhãn theo thứ tự chữ cái: BARBUNYA=0, BOMBAY=1, ..., SIRA=6
    classes, y = np.unique(df[cfg["data"]["target"]].astype(str).to_numpy(), return_inverse=True)
    return X, y, list(classes)


def split_path(cfg, k):
    return os.path.join(cfg["split"]["dir"], f"split_{k}.npz")


def make_splits(cfg):
    """Tạo n_splits split phân tầng và lưu chỉ số. Split k dùng random_state = k."""
    X, y, classes = load_table(cfg)
    md5 = file_md5(cfg["data"]["path"])
    os.makedirs(cfg["split"]["dir"], exist_ok=True)
    idx = np.arange(len(y))

    for k in range(cfg["split"]["n_splits"]):
        # Bước 1: train_full / test (80/20), giữ tỉ lệ lớp
        tr_full, te = train_test_split(idx, test_size=cfg["split"]["test_size"],
                                       stratify=y, random_state=k)
        # Bước 2: tách validation TỪ train_full, cũng giữ tỉ lệ lớp
        tr, va = train_test_split(tr_full, test_size=cfg["split"]["val_size"],
                                  stratify=y[tr_full], random_state=k)
        np.savez(split_path(cfg, k),
                 train_idx=np.sort(tr).astype(np.int32),
                 val_idx=np.sort(va).astype(np.int32),
                 test_idx=np.sort(te).astype(np.int32),
                 data_md5=md5, classes=np.array(classes))
        print(f"split_{k}: train={len(tr)} val={len(va)} test={len(te)}")


def load_split(cfg):
    """Trả về dict gồm X_tr, y_tr, X_va, y_va, X_te, y_te (đã scale), classes, data_md5."""
    k = cfg["split"]["split_id"]
    s = np.load(split_path(cfg, k))
    X, y, classes = load_table(cfg)

    md5 = file_md5(cfg["data"]["path"])
    if str(s["data_md5"]) != md5:
        raise ValueError(f"File dữ liệu khác với file đã dùng để tạo split_{k} (md5 không khớp)")
    if list(s["classes"]) != classes:
        raise ValueError("Danh sách lớp khác với lúc tạo split")

    tr, va, te = s["train_idx"], s["val_idx"], s["test_idx"]
    scaler = MinMaxScaler().fit(X[tr])          # CHỈ fit trên train
    return {
        "X_tr": scaler.transform(X[tr]), "y_tr": y[tr],
        "X_va": scaler.transform(X[va]), "y_va": y[va],
        "X_te": scaler.transform(X[te]), "y_te": y[te],
        "classes": classes, "data_md5": md5,
    }
