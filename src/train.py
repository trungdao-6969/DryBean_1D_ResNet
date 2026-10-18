"""
src/train.py — chạy MỘT run từ một cfg, và ghi log.

train_one_run(cfg) làm đúng thứ tự:
    seed -> data -> model -> callbacks -> fit -> load best.keras -> đánh giá test
    -> lưu artefact (config, history, curve, cm) -> append_log (1 dòng)

append_log là hàm DUY NHẤT được phép ghi vào runs_log.csv (feedback 2.3).
"""
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras

from src.config import make_run_id, save_config
from src.data import load_split
from src.models import build_model

_DETERMINISM_ON = False


def set_seed(seed, deterministic=True):
    """Cố định mọi nguồn ngẫu nhiên: Python, NumPy, TensorFlow (khởi tạo, dropout, shuffle)."""
    global _DETERMINISM_ON
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic and not _DETERMINISM_ON:
        # Ép GPU dùng thuật toán tất định: chạy lại cùng cfg -> cùng kết quả (chậm hơn một chút)
        tf.config.experimental.enable_op_determinism()
        _DETERMINISM_ON = True


def make_optimizer(name, lr):
    opts = {"adam": keras.optimizers.Adam, "nadam": keras.optimizers.Nadam}
    return opts[name.lower()](learning_rate=lr)


def compute_metrics(y_true, y_pred, classes):
    """Accuracy, P/R/F1 macro và weighted, recall từng lớp, confusion matrix (đơn vị %)."""
    out = {"test_acc": 100 * accuracy_score(y_true, y_pred)}
    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"test_prec_{avg}"] = 100 * p
        out[f"test_rec_{avg}"] = 100 * r
        out[f"test_f1_{avg}"] = 100 * f
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    recall_per_class = 100 * np.diag(cm) / cm.sum(axis=1)
    for c, r in zip(classes, recall_per_class):
        out[f"recall_{c}"] = r
    return out, cm


def save_curve(hist, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(hist) + 1)
    for key, a in [("loss", ax[0]), ("accuracy", ax[1])]:
        a.plot(ep, hist[key], label=f"train {key}")
        a.plot(ep, hist[f"val_{key}"], label=f"val {key}")
        a.set_xlabel("epoch")
        a.legend()
    ax[0].axvline(int(np.argmin(hist["val_loss"])) + 1, ls="--", c="gray")
    ax[1].axvline(int(np.argmax(hist["val_accuracy"])) + 1, ls="--", c="gray")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def append_log(row, log_path):
    """Thêm 1 dòng vào log. Ghi ra file tạm rồi đổi tên -> Colab ngắt giữa chừng cũng không hỏng log."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    new = pd.DataFrame([row])
    df = pd.concat([pd.read_csv(log_path), new], ignore_index=True) if os.path.exists(log_path) else new
    tmp = log_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, log_path)


def train_one_run(cfg):
    run_id = make_run_id(cfg)
    run_dir = os.path.join(cfg["output"]["runs_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    tcfg, ccfg = cfg["train"], cfg["callbacks"]

    # 1. Dữ liệu: (n, 16) -> (n, 16, 1) vì Conv1D cần trục kênh
    d = load_split(cfg)
    X_tr, X_va, X_te = (d[k][..., None] for k in ["X_tr", "X_va", "X_te"])
    y_tr, y_va, y_te = d["y_tr"], d["y_va"], d["y_te"]

    # 2. Model (clear_session trước set_seed để tên lớp và RNG bắt đầu lại từ đầu)
    keras.backend.clear_session()
    set_seed(tcfg["seed"], tcfg.get("deterministic", True))
    model = build_model(cfg, n_features=X_tr.shape[1], n_classes=len(d["classes"]))
    # Nhãn là số nguyên 0..6 -> sparse_categorical_crossentropy
    # (tương đương categorical_crossentropy với one-hot, không cần đổi nhãn)
    model.compile(optimizer=make_optimizer(tcfg["optimizer"], tcfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # 3. Callbacks: checkpoint luôn bật; early stopping tuỳ cfg
    ckpt_path = os.path.join(run_dir, "best.keras")
    mode = lambda m: "min" if "loss" in m else "max"
    cbs = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor=ccfg["checkpoint"]["monitor"],
                                           mode=mode(ccfg["checkpoint"]["monitor"]),
                                           save_best_only=True)]
    es = ccfg["early_stopping"]
    if es["enabled"]:
        # restore_best_weights=False: ta tự load best.keras ở bước 5, cách này chắc chắn hơn
        cbs.append(keras.callbacks.EarlyStopping(monitor=es["monitor"], mode=mode(es["monitor"]),
                                                 patience=es["patience"]))

    # 4. Huấn luyện
    t0 = time.time()
    h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=tcfg["epochs"],
                  batch_size=tcfg["batch_size"], shuffle=True, callbacks=cbs,
                  verbose=tcfg.get("verbose", 0))
    train_time = time.time() - t0
    hist = pd.DataFrame(h.history)
    hist.index = np.arange(1, len(hist) + 1)
    hist.index.name = "epoch"

    # Model ở epoch CUỐI (chỉ có ý nghĩa khi không early stopping)
    acc_last = 100 * accuracy_score(y_te, model.predict(X_te, verbose=0).argmax(1))

    # 5. Đánh giá test bằng model TỐT NHẤT trên validation
    best = keras.models.load_model(ckpt_path)
    y_pred = best.predict(X_te, verbose=0).argmax(1)
    metrics, cm = compute_metrics(y_te, y_pred, d["classes"])

    # 6. Lưu artefact
    hist.to_csv(os.path.join(run_dir, "history.csv"))
    np.save(os.path.join(run_dir, "cm.npy"), cm)
    save_curve(hist, os.path.join(run_dir, "curve.png"), run_id)

    # 7. Một dòng log
    row = {
        "run_id": run_id, "group": cfg["experiment"]["group"], "model": cfg["model"]["name"],
        "split_id": cfg["split"]["split_id"], "seed": tcfg["seed"],
        "shortcut_bn": cfg["model"]["shortcut_bn"],
        "es_monitor": es["monitor"] if es["enabled"] else "none",
        "ckpt_monitor": ccfg["checkpoint"]["monitor"],
        "optimizer": tcfg["optimizer"], "lr": tcfg["lr"], "batch_size": tcfg["batch_size"],
        "max_epochs": tcfg["epochs"], "n_params": model.count_params(),
        "epochs_ran": len(hist),
        "best_epoch_val_loss": int(hist["val_loss"].idxmin()),
        "best_epoch_val_acc": int(hist["val_accuracy"].idxmax()),
        "min_val_loss": hist["val_loss"].min(),
        "max_val_acc": 100 * hist["val_accuracy"].max(),
        "train_time_s": round(train_time, 1),
        **metrics,
        "test_acc_last": acc_last,
        "status": "completed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_md5": d["data_md5"], "tf_version": tf.__version__,
        "ckpt_path": ckpt_path,
    }
    append_log(row, cfg["output"]["log_path"])
    return row
