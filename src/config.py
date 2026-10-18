"""
src/config.py — đọc và ghép cấu hình.

Nguyên tắc: mọi kết quả được sinh HOÀN TOÀN từ một dict cfg, và cfg đó được lưu
cạnh kết quả (results/runs/<run_id>/config.yaml). Không còn biến nào nằm ngoài
config để "sửa tay" giữa các lần chạy.

Thứ tự ghép:  baseline.yaml  <-  ablation_*.yaml  <-  overrides (model.name, split_id, ...)
              rồi chép per_model[model.name] vào cfg["train"].
"""
import copy
import hashlib
import json

import yaml


def read_yaml(path):
    """Đọc 1 file YAML thành dict (file rỗng -> {})."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, extra):
    """Trả về bản sao của `base` đã được ghi đè bởi `extra`, ghép ĐỆ QUY.

    Ví dụ: base = {"model": {"name": "resnet1", "shortcut_bn": True}}
           extra = {"model": {"shortcut_bn": False}}
        -> {"model": {"name": "resnet1", "shortcut_bn": False}}
    Nếu dùng dict.update thông thường thì cả khối "model" bị thay, mất "name".
    """
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(base_path, ablation_path=None, overrides=None):
    """Đọc baseline, ghép ablation và overrides, rồi gắn siêu tham số riêng của model."""
    cfg = read_yaml(base_path)
    if ablation_path:
        cfg = deep_merge(cfg, read_yaml(ablation_path))
    if overrides:
        cfg = deep_merge(cfg, overrides)

    # Chép optimizer / lr / batch_size của đúng model đang chạy vào cfg["train"],
    # rồi bỏ khối per_model để config đã lưu chỉ chứa những gì THỰC SỰ dùng.
    # per_model là giá trị MẶC ĐỊNH: nếu ablation/overrides đã ghi train.lr thì giữ giá trị đó.
    name = cfg["model"]["name"]
    per_model = cfg.pop("per_model", {})
    if name not in per_model:
        raise KeyError(f"Không có per_model.{name} trong config")
    cfg["train"] = deep_merge(per_model[name], cfg["train"])
    return cfg


def config_hash(cfg, n=6):
    """6 ký tự đầu của md5 trên TOÀN BỘ cfg (trừ output và train.verbose).

    sort_keys=True: hai dict giống nhau nhưng khác thứ tự khoá vẫn cho cùng chuỗi JSON,
    nên cùng hash. Bỏ output / verbose: đổi thư mục lưu hay mức in log không làm đổi thí nghiệm.
    """
    core = {k: copy.deepcopy(v) for k, v in cfg.items() if k != "output"}
    core.get("train", {}).pop("verbose", None)
    text = json.dumps(core, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def make_run_id(cfg):
    """Tên run đọc được + hash, ví dụ: resnet1_bn_es-val_loss100_split3_seed0_a1b2c3.

    Phần chữ chỉ để người đọc; phần hash đảm bảo hai config khác nhau dù chỉ một
    trường (vd lr) cũng không bao giờ trùng run_id.
    """
    m = cfg["model"]
    es = cfg["callbacks"]["early_stopping"]
    bn = "bn" if m["shortcut_bn"] else "nobn"
    es_tag = f"es-{es['monitor']}{es['patience']}" if es["enabled"] else "noes"
    return (f"{m['name']}_{bn}_{es_tag}_split{cfg['split']['split_id']}"
            f"_seed{cfg['train']['seed']}_{config_hash(cfg)}")


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
