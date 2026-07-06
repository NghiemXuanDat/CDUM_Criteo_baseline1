"""
euen.py — Kiến trúc EUEN (Explicit Uplift Effect Network)
==========================================================
Triển khai mô hình EUEN cho bài toán Uplift Modeling (Individual Treatment
Effect Estimation) theo PyTorch, port từ bản gốc XDL/TF1.x.

Tham chiếu:
    Ke, Z. et al. "Addressing Exposure Bias in Uplift Modeling for Large-Scale
    Online Advertising." ICDM 2021.

Kiến trúc hai nhánh song song (hc_dim=hu_dim=64, is_self=False):

    X (input, dim=12)
        │
        ├──► ControlNet ─────────────────────────────► c_logit  (E[Y(0)|X])
        │     Linear(12→64) → ELU
        │     → Linear(64→32) → ELU
        │     → Linear(32→16) → ELU
        │     → Linear(16→1)
        │
        └──► UpliftNet ──────────────────────────────► u_tau    (τ(X))
              Linear(12→64) → ELU
              → Linear(64→32) → ELU
              → Linear(32→16) → ELU
              → Linear(16→1)

Cơ chế stop_gradient (detach) trong forward pass:
    c_logit_fix = detach(c_logit)          ← ngăn gradient từ treatment arm
    uc = c_logit                           ← dự đoán cho nhóm control
    ut = c_logit_fix + u_tau               ← dự đoán cho nhóm treatment
    u  = (1−T)·uc + T·ut                  ← dự đoán chung
    uplift_score = u_tau                   ← ITE trực tiếp

Loss function (lift MSE, use_huber=False, use_group_reduce=False):
    L = mean(((1−T)·uc + T·ut − Y)²)
      = mean((u − Y)²)

Các lớp công khai:
    init_weights()       — Kaiming Normal + bias=0.1 (khớp với utils.fc gốc)
    ControlNet           — mạng ước lượng E[Y(0)|X]
    UpliftNet            — mạng ước lượng τ(X)
    EUENModel            — mô hình EUEN đầy đủ (tích hợp 2 nhánh)
    CriteoDataset        — torch.utils.data.Dataset cho Criteo Uplift v2.1

Môi trường yêu cầu:
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Utility — khởi tạo trọng số
# ══════════════════════════════════════════════════════════════════════════════

def init_weights(m):
    """
    Khởi tạo trọng số nn.Linear theo Kaiming Normal (He initialization).

    Công thức: std = sqrt(2 / fan_in),  bias = 0.1.
    Khớp với cách khởi tạo mặc định trong utils.fc của EUEN gốc:
        weight_value = randn(fan_in, fan_out) * sqrt(2 / fan_in)
        bias_initializer = constant(0.1)

    Áp dụng qua module.apply(init_weights) ngay sau khi khai báo.

    Tham số
    -------
    m : nn.Module — chỉ áp dụng khi m là nn.Linear; các module khác bỏ qua.
    """
    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        nn.init.constant_(m.bias, 0.1)


# ══════════════════════════════════════════════════════════════════════════════
# ControlNet — Mạng ước lượng E[Y(0)|X]
# ══════════════════════════════════════════════════════════════════════════════

class ControlNet(nn.Module):
    """
    Control Network: ước lượng outcome kỳ vọng của nhóm control E[Y(0)|X].

    Kiến trúc 3 lớp ẩn (tương ứng với is_self=False trong bản gốc):
        Linear(input_dim → hc_dim)        → ELU    ← không có L2 (lớp đầu)
        → Linear(hc_dim  → hc_dim//2)    → ELU    ← có L2 reg
        → Linear(hc_dim//2 → hc_dim//4)  → ELU    ← có L2 reg
        → Linear(hc_dim//4 → 1)                    ← c_logit (raw, linear output)

    Chú ý: L2 regularization được xử lý ở mức optimizer (weight_decay trong Adam),
    không cần khai báo riêng trong từng lớp.

    Tham số
    -------
    input_dim : int — số chiều feature đầu vào (12 với Criteo f0…f11).
    hc_dim    : int — số neuron lớp ẩn đầu tiên (mặc định 64, theo bản gốc).
    """

    def __init__(self, input_dim: int, hc_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hc_dim),        nn.ELU(),
            nn.Linear(hc_dim, hc_dim // 2),      nn.ELU(),
            nn.Linear(hc_dim // 2, hc_dim // 4), nn.ELU(),
        )
        self.logit_head = nn.Linear(hc_dim // 4, 1)

        # Áp dụng Kaiming init cho toàn bộ sub-module
        self.apply(init_weights)

    def forward(self, x):
        """
        Tham số
        -------
        x : Tensor (n, input_dim) — feature vector.

        Trả về
        ------
        c_logit : Tensor (n, 1) — raw logit ước lượng E[Y(0)|X].
        """
        return self.logit_head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# UpliftNet — Mạng ước lượng ITE τ(X) = E[Y(1)−Y(0)|X]
# ══════════════════════════════════════════════════════════════════════════════

class UpliftNet(nn.Module):
    """
    Uplift Network: ước lượng Individual Treatment Effect (ITE)
    τ(X) = E[Y(1) − Y(0) | X].

    Kiến trúc 3 lớp ẩn (tương ứng với is_self=False trong bản gốc):
        Linear(input_dim → hu_dim)        → ELU    ← không có L2 (lớp đầu)
        → Linear(hu_dim  → hu_dim//2)    → ELU    ← có L2 reg
        → Linear(hu_dim//2 → hu_dim//4)  → ELU    ← có L2 reg
        → Linear(hu_dim//4 → 1)                    ← u_tau (uplift score raw)

    UpliftNet và ControlNet dùng cùng x làm input nhưng là hai mạng HOÀN TOÀN
    độc lập — không chia sẻ tham số. Điều này cho phép mỗi mạng học một
    hàm mục tiêu khác nhau (outcome vs. uplift).

    Tham số
    -------
    input_dim : int — số chiều feature đầu vào (12 với Criteo f0…f11).
    hu_dim    : int — số neuron lớp ẩn đầu tiên (mặc định 64, theo bản gốc).
    """

    def __init__(self, input_dim: int, hu_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hu_dim),        nn.ELU(),
            nn.Linear(hu_dim, hu_dim // 2),      nn.ELU(),
            nn.Linear(hu_dim // 2, hu_dim // 4), nn.ELU(),
        )
        self.tau_head = nn.Linear(hu_dim // 4, 1)

        self.apply(init_weights)

    def forward(self, x):
        """
        Tham số
        -------
        x : Tensor (n, input_dim) — feature vector.

        Trả về
        ------
        u_tau : Tensor (n, 1) — uplift score τ(X) = E[Y(1) − Y(0)|X].
        """
        return self.tau_head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# EUENModel — Mô hình EUEN đầy đủ (tích hợp 2 nhánh)
# ══════════════════════════════════════════════════════════════════════════════

class EUENModel(nn.Module):
    """
    EUEN (Explicit Uplift Effect Network) — mô hình uplift hai nhánh đầy đủ.

    Tích hợp ControlNet và UpliftNet trong một forward pass thống nhất.
    Hai nhánh nhận cùng x làm input nhưng học hai hàm mục tiêu khác nhau:
        - ControlNet  → c_logit  : ước lượng E[Y(0)|X]
        - UpliftNet   → u_tau    : ước lượng τ(X) = E[Y(1)−Y(0)|X]

    Cơ chế stop_gradient (thiết kế then chốt của EUEN):
        c_logit_fix = detach(c_logit)
            Ngăn gradient từ treatment arm lan ngược qua ControlNet.
            Đảm bảo ControlNet chỉ học từ control samples, không bị nhiễu
            bởi tín hiệu từ treatment arm.

        uc = c_logit                      ← prediction khi T=0
        ut = c_logit_fix + u_tau          ← prediction khi T=1
            Treatment prediction = control baseline (frozen) + uplift effect.
            UpliftNet học bù đắp phần chênh lệch giữa treatment và control,
            không cần học lại toàn bộ outcome.

    Tùy chọn BN đầu vào (use_bn=True, default):
        BatchNorm1d(input_dim) trước khi đưa vào cả hai nhánh.
        Giúp ổn định quá trình huấn luyện khi features Criteo có scale khác nhau
        (f0…f11 có khoảng giá trị rất khác nhau).

    Đầu ra (tuple 2 phần tử):
        c_logit : Tensor (n, 1) — ControlNet output  (E[Y(0)|X])
        u_tau   : Tensor (n, 1) — UpliftNet output   (τ(X) = uplift score)

    Tham số
    -------
    input_dim : int  — số chiều feature (12 với Criteo f0…f11).
    hc_dim    : int  — số neuron ControlNet lớp 1 (mặc định 64).
    hu_dim    : int  — số neuron UpliftNet lớp 1 (mặc định 64).
    use_bn    : bool — có dùng BatchNorm1d ở đầu vào không (mặc định True).
    """

    def __init__(self, input_dim: int, hc_dim: int = 64, hu_dim: int = 64,
                 use_bn: bool = True):
        super().__init__()
        # Input BatchNorm: chia sẻ giữa cả hai nhánh, chuẩn hóa features đầu vào
        self.input_bn = nn.BatchNorm1d(input_dim) if use_bn else nn.Identity()

        self.control_net = ControlNet(input_dim, hc_dim)
        self.uplift_net  = UpliftNet(input_dim, hu_dim)

    def forward(self, x):
        """
        Tham số
        -------
        x : Tensor (n, input_dim) — feature vector chưa chuẩn hóa.

        Trả về
        ------
        c_logit : Tensor (n, 1) — ControlNet raw output.
        u_tau   : Tensor (n, 1) — UpliftNet raw output (uplift score).
        """
        # Chuẩn hóa đầu vào (BatchNorm hoặc Identity nếu use_bn=False)
        x = self.input_bn(x)

        c_logit = self.control_net(x)   # (n, 1) — E[Y(0)|X]
        u_tau   = self.uplift_net(x)    # (n, 1) — τ(X)

        return c_logit, u_tau


# ══════════════════════════════════════════════════════════════════════════════
# CriteoDataset — torch.utils.data.Dataset cho Criteo Uplift v2.1
# ══════════════════════════════════════════════════════════════════════════════

class CriteoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrap dữ liệu Criteo Uplift v2.1 (đã qua preprocess).

    Mỗi mẫu trả về tuple (x, t, y) dạng float32 tensor, phục vụ
    trực tiếp cho vòng lặp DataLoader trong train_one_epoch.

    Tham số
    -------
    X : np.ndarray float32 (n, 12) — ma trận đặc trưng f0…f11.
    y : np.ndarray (n,)            — nhãn kết quả binary (0 hoặc 1).
    t : np.ndarray (n,)            — nhãn treatment (0 hoặc 1).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Trả về (x, t, y) — thứ tự này khớp với vòng lặp DataLoader
        return self.X[idx], self.t[idx], self.y[idx]
