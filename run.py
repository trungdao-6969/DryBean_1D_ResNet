"""
run.py — chạy lưới model x split cho một nhóm thí nghiệm.

Ví dụ:
    python run.py                                     # baseline: 3 model x 5 split
    python run.py --ablation configs/ablation_noBN.yaml
    python run.py --smoke                             # chạy thử 3 epoch, ghi vào results_smoke/
    python run.py --models resnet1 --splits 0         # chỉ 1 run

Chạy lại đúng lệnh sau khi Colab bị ngắt: run nào đã `completed` trong log sẽ được bỏ qua.
"""
import argparse
import os
import traceback

import pandas as pd

from src.config import load_config, make_run_id
from src.data import make_splits, split_path


def done_run_ids(log_path):
    if not os.path.exists(log_path):
        return set()
    df = pd.read_csv(log_path)
    return set(df.loc[df["status"] == "completed", "run_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/baseline.yaml")
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--models", nargs="+", default=["resnet1", "resnet2", "resnet3"])
    ap.add_argument("--splits", nargs="+", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="3 epoch, ghi vào results_smoke/")
    args = ap.parse_args()

    overrides = {}
    if args.smoke:
        overrides = {"train": {"epochs": 3, "verbose": 2},
                     "output": {"runs_dir": "results_smoke/runs",
                                "log_path": "results_smoke/runs_log.csv"}}

    # Tạo split nếu chưa có (chỉ xảy ra một lần; sau đó split được commit lên repo)
    cfg0 = load_config(args.base, args.ablation, overrides)
    n_splits = cfg0["split"]["n_splits"]
    if not all(os.path.exists(split_path(cfg0, k)) for k in range(n_splits)):
        print("Chưa có split -> tạo mới")
        make_splits(cfg0)
    splits = args.splits if args.splits is not None else list(range(n_splits))

    from src.train import append_log, train_one_run   # import TF sau khi đã kiểm tra split

    for m in args.models:
        for k in splits:
            ov = {**overrides, "model": {"name": m}, "split": {"split_id": k}}
            cfg = load_config(args.base, args.ablation, ov)
            run_id = make_run_id(cfg)
            if run_id in done_run_ids(cfg["output"]["log_path"]):
                print(f"[skip ] {run_id}")
                continue
            print(f"[start] {run_id}", flush=True)
            try:
                r = train_one_run(cfg)
                print(f"[done ] {run_id} | params={r['n_params']:,} epochs={r['epochs_ran']} "
                      f"best_ep={r['best_epoch_val_loss']} test_acc={r['test_acc']:.2f} "
                      f"f1_w={r['test_f1_weighted']:.2f} ({r['train_time_s']:.0f}s)", flush=True)
            except Exception as e:
                traceback.print_exc()
                append_log({"run_id": run_id, "group": cfg["experiment"]["group"], "model": m,
                            "split_id": k, "status": "failed", "error": repr(e)[:300]},
                           cfg["output"]["log_path"])
                print(f"[FAIL ] {run_id}: {e}")
    print("== Xong")


if __name__ == "__main__":
    main()
