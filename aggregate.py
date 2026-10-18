"""
aggregate.py — kiểm tra log rồi lập bảng kết quả.

    python aggregate.py                      # mọi group có trong log
    python aggregate.py --groups baseline    # chỉ baseline (khi ablation chưa chạy xong)

1. check_log: mỗi group phải đủ 3 model x 5 split, không trùng run_id, cùng md5 dữ liệu,
   run không early stopping phải chạy đủ max_epochs. Sai -> dừng, KHÔNG tính trung bình.
2. mean ± std (ddof=1) và CI 95% = mean ± t(0.975, n-1) * s / sqrt(n)   (n=5 -> t = 2.776)
3. Recall từng lớp, confusion matrix cộng dồn 5 split.
4. So sánh ghép cặp (cùng split): ablation vs baseline, ResNet2/3 vs ResNet1 (paired t-test).
Kết quả ghi vào results/summary/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["resnet1", "resnet2", "resnet3"]
N_SPLITS = 5
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
METRICS = ["test_acc", "test_f1_weighted", "test_f1_macro", "test_prec_weighted", "test_rec_weighted"]
# Paper (Phan 2026), Table 3/5/6, 10 lần hold-out; std trong paper là ddof=0
PAPER = {"resnet1": {"acc": 95.70, "acc_std": 0.51, "f1_w": 95.55},
         "resnet2": {"acc": 93.79, "acc_std": 0.57, "f1_w": 93.81},
         "resnet3": {"acc": 93.71, "acc_std": 0.48, "f1_w": 93.70}}


def check_log(df, groups):
    errors = []
    done = df[df["status"] == "completed"]
    dup = done["run_id"][done["run_id"].duplicated()]
    if len(dup):
        errors.append(f"run_id trùng: {sorted(set(dup))}")
    if done["data_md5"].nunique() > 1:
        errors.append(f"Có {done['data_md5'].nunique()} md5 dữ liệu khác nhau trong log")
    for g in groups:
        d = done[done["group"] == g]
        have = set(zip(d["model"], d["split_id"]))
        need = {(m, k) for m in MODELS for k in range(N_SPLITS)}
        if have != need:
            errors.append(f"[{g}] thiếu run: {sorted(need - have)}  thừa: {sorted(have - need)}")
        no_es = d[d["es_monitor"] == "none"]
        bad = no_es[no_es["epochs_ran"] != no_es["max_epochs"]]
        if len(bad):
            errors.append(f"[{g}] run không ES nhưng chưa chạy đủ epoch: {list(bad['run_id'])}")
        for rd in d["ckpt_path"].map(os.path.dirname):
            if not os.path.exists(os.path.join(rd, "cm.npy")):
                errors.append(f"[{g}] thiếu {rd}/cm.npy")
    failed = df[df["status"] != "completed"]
    if len(failed):
        print(f"(Lưu ý) {len(failed)} dòng failed trong log, đã bỏ qua: {list(failed['run_id'])}")
    if errors:
        print("Log CHƯA đạt:\n  - " + "\n  - ".join(errors))
        sys.exit(1)
    print(f"Log đạt: {len(done)} run completed, groups = {groups}")
    return done


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def summary_table(done, groups):
    rows = []
    for g in groups:
        for m in MODELS:
            d = done[(done["group"] == g) & (done["model"] == m)]
            r = {"group": g, "model": m, "n": len(d), "n_params": int(d["n_params"].iloc[0])}
            for met in METRICS:
                mu, sd, lo, hi = mean_ci(d[met])
                r[met] = f"{mu:.2f} ± {sd:.2f} [{lo:.2f}, {hi:.2f}]"
                r[f"{met}_mean"], r[f"{met}_std"] = mu, sd
            r["epochs_ran"] = f"{d['epochs_ran'].mean():.0f}"
            r["best_epoch"] = f"{d['best_epoch_val_loss'].mean():.0f}"
            r["train_time_s"] = f"{d['train_time_s'].mean():.0f}"
            if (d["es_monitor"] == "none").all():
                r["test_acc_last"] = f"{d['test_acc_last'].mean():.2f}"
            r["paper_acc"] = f"{PAPER[m]['acc']:.2f} ± {PAPER[m]['acc_std']:.2f}"
            r["gap_vs_paper"] = f"{r['test_acc_mean'] - PAPER[m]['acc']:+.2f}"
            rows.append(r)
    return pd.DataFrame(rows)


def per_class_recall(done, groups):
    cols = [f"recall_{c}" for c in CLASSES]
    t = done[done["group"].isin(groups)].groupby(["group", "model"])[cols].mean().round(2)
    t.columns = CLASSES
    return t


def summed_cm(done, group, model):
    d = done[(done["group"] == group) & (done["model"] == model)]
    return sum(np.load(os.path.join(os.path.dirname(p), "cm.npy")) for p in d["ckpt_path"])


def paired(done, a_group, a_model, b_group, b_model, metric="test_acc"):
    """B - A theo từng split (cùng split -> ghép cặp được)."""
    a = done[(done["group"] == a_group) & (done["model"] == a_model)].set_index("split_id")[metric]
    b = done[(done["group"] == b_group) & (done["model"] == b_model)].set_index("split_id")[metric]
    diff = (b - a.loc[b.index]).sort_index()
    p = stats.ttest_rel(b.sort_index(), a.loc[b.index].sort_index()).pvalue
    return {"A": f"{a_group}/{a_model}", "B": f"{b_group}/{b_model}", "metric": metric,
            "mean_diff(B-A)": round(diff.mean(), 3),
            "diff_per_split": " ".join(f"{v:+.2f}" for v in diff),
            "n_B>A": int((diff > 0).sum()), "p_paired_t": round(p, 4)}


def comparisons(done, groups):
    rows = []
    if "baseline" in groups:
        for m in ["resnet2", "resnet3"]:
            rows.append(paired(done, "baseline", "resnet1", "baseline", m))
        for g in groups:
            if g != "baseline":
                for m in MODELS:
                    rows.append(paired(done, "baseline", m, g, m))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/runs_log.csv")
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    groups = args.groups or sorted(df.loc[df["status"] == "completed", "group"].unique(),
                                   key=lambda g: (g != "baseline", g))
    done = check_log(df, groups)
    os.makedirs(args.out, exist_ok=True)

    summ = summary_table(done, groups)
    rec = per_class_recall(done, groups)
    comp = comparisons(done, groups)
    summ.to_csv(f"{args.out}/summary.csv", index=False)
    rec.to_csv(f"{args.out}/per_class_recall.csv")
    comp.to_csv(f"{args.out}/paired_tests.csv", index=False)
    for g in groups:
        for m in MODELS:
            cm = summed_cm(done, g, m)
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{args.out}/cm_{g}_{m}.csv")

    show = ["group", "model", "n_params", "test_acc", "test_f1_weighted", "test_f1_macro",
            "epochs_ran", "best_epoch", "paper_acc", "gap_vs_paper"]
    show += ["test_acc_last"] if "test_acc_last" in summ else []
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== Kết quả (mean ± std [CI 95%], %) ==")
    print(summ[show].to_string(index=False))
    print("\n== Recall từng lớp (mean qua 5 split, %) ==")
    print(rec.to_string())
    if len(comp):
        print("\n== So sánh ghép cặp theo split (test_acc) ==")
        print(comp.to_string(index=False))
    print(f"\nĐã ghi bảng vào {args.out}/")


if __name__ == "__main__":
    main()
