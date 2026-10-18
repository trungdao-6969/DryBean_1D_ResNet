"""
src/models.py — 3 kiến trúc 1D-ResNet theo paper (Phan 2026, §2.2, Fig. 3).

Phần CỐ ĐỊNH của kiến trúc (số filter, vị trí Dropout, số unit Dense) nằm trong
ARCH bên dưới, vì đó là định nghĩa của ResNet1/2/3 và không bao giờ đổi trong thí
nghiệm. Phần có thể đổi (shortcut_bn, L2, kernel_size) đọc từ cfg["model"].

Residual block (ResNet v1):
    main:     x -> Conv1D(f,k) -> BN -> ReLU -> Conv1D(f,k) -> BN
    shortcut: x -> Conv1D(f,1) [-> BN nếu shortcut_bn]      (projection vì số kênh đổi)
    out = ReLU(main + shortcut)
"""
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ARCH = {
    # filters của từng block | Dropout sau block thứ i (đếm từ 0) | số unit Dense ẩn
    "resnet1": {"filters": [128, 256, 384],      "block_dropout": {},               "dense": 128},
    "resnet2": {"filters": [128, 256, 512],      "block_dropout": {2: 0.5},         "dense": 256},
    "resnet3": {"filters": [128, 256, 384, 512], "block_dropout": {2: 0.3, 3: 0.3}, "dense": 128},
}
STEM_FILTERS = 64
HEAD_DROPOUT = 0.5
INIT = "glorot_uniform"


def conv(filters, kernel_size, l2, name):
    reg = regularizers.l2(l2) if l2 > 0 else None
    return layers.Conv1D(filters, kernel_size, padding="same", kernel_initializer=INIT,
                         kernel_regularizer=reg, name=name)


def residual_block(x, filters, mcfg, name):
    k, l2 = mcfg["kernel_size"], mcfg["conv_l2"]

    # Nhánh chính F(x)
    h = conv(filters, k, l2, f"{name}_conv1")(x)
    h = layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = layers.ReLU(name=f"{name}_relu1")(h)
    h = conv(filters, k, l2, f"{name}_conv2")(h)
    h = layers.BatchNormalization(name=f"{name}_bn2")(h)

    # Nhánh tắt S(x): Conv 1x1 đưa số kênh về `filters` để cộng được với F(x)
    s = conv(filters, 1, l2, f"{name}_proj")(x)
    if mcfg["shortcut_bn"]:
        s = layers.BatchNormalization(name=f"{name}_proj_bn")(s)

    out = layers.Add(name=f"{name}_add")([h, s])
    return layers.ReLU(name=f"{name}_out")(out)


def build_model(cfg, n_features=16, n_classes=7):
    mcfg = cfg["model"]
    arch = ARCH[mcfg["name"]]

    inp = keras.Input(shape=(n_features, 1), name="input")

    # Stem
    x = conv(STEM_FILTERS, mcfg["kernel_size"], mcfg["conv_l2"], "stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Các residual block
    for i, f in enumerate(arch["filters"]):
        x = residual_block(x, f, mcfg, name=f"block{i + 1}")
        if i in arch["block_dropout"]:
            x = layers.Dropout(arch["block_dropout"][i], name=f"block{i + 1}_dropout")(x)

    # Head: L2 CHỈ ở Dense ẩn
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(arch["dense"], activation="relu", kernel_initializer=INIT,
                     kernel_regularizer=regularizers.l2(mcfg["dense_l2"]), name="dense")(x)
    x = layers.Dropout(HEAD_DROPOUT, name="head_dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_initializer=INIT, name="output")(x)

    return keras.Model(inp, out, name=mcfg["name"])
