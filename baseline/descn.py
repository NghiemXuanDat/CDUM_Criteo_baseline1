"""
descn.py — Kiến trúc DESCN (Deep Entire Space Cross Networks)
=============================================================
Triển khai mô hình DESCN cho bài toán Uplift Modeling (Individual Treatment
Effect Estimation).

Tham chiếu:
    Du, F. et al. "Deep Entire Space Cross Networks for Individual Treatment
    Effect Estimation." KDD 2022.

Kiến trúc multi-task gồm 5 mạng con học đồng thời:

    X (input, dim=12) ──► ShareNetwork ─────────────────────────────────────
                           (shared_h)  ──► PrpsyNetwork  →  p_prpsy
                                      │
                                      ├──► Mu0Network    →  p_mu0  (P(Y|X,T=0))
                                      │
                                      ├──► Mu1Network    →  p_mu1  (P(Y|X,T=1))
                                      │
                                      └──► TauNetwork    →  tau_logit (ITE trực tiếp)

Các đầu ra tổng hợp (DESCNModel.forward()):
    p_estr  = p_prpsy × p_mu1       — Entire Space Treatment Response
    p_escr  = (1−p_prpsy) × p_mu0   — Entire Space Control Response
    uplift  = p_mu1 − p_mu0         — Uplift score cuối cùng

Loss functions (multi-task, trọng số mặc định theo DESCN gốc):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Tên loss         Công thức                           Weight (default)   │
    │  ──────────────   ───────────────────────────────     ────────────────   │
    │  h1_loss          BCE(p_h1[T=1], y[T=1])              0.5               │
    │  h0_loss          BCE(p_h0[T=0], y[T=0])              0.1               │
    │  cross_tr_loss    BCE(σ(μ0+τ)[T=1], y[T=1])           0.0 (disabled)   │
    │  cross_cr_loss    BCE(σ(μ1−τ)[T=0], y[T=0])           0.0 (disabled)   │
    │  prpsy_loss       BCE_logit(p̂_prpsy, T)               0.0 (disabled)   │
    │  estr_loss        BCE_w(p_estr, Y·T)                  0.0 (disabled)   │
    │  escr_loss        BCE_w(p_escr, Y·(1−T))              0.0 (disabled)   │
    │  imb_dist_loss    MMD(shared_h[T=1], shared_h[T=0])   0.1               │
    └──────────────────────────────────────────────────────────────────────────┘

Các lớp công khai:
    init_weights()    — khởi tạo trọng số nn.Linear theo Normal (Xavier-like)
    safe_sqrt()       — sqrt ổn định số học
    gaussian_mmd()    — MMD với Gaussian kernel (thay thế geomloss)
    BaseModel         — DNN backbone 3 lớp (Linear→ELU→Dropout)
    ShareNetwork      — bộ mã hóa chung có BN tùy chọn + L2-normalize
    PrpsyNetwork      — đầu dự đoán propensity score
    Mu0Network        — đầu dự đoán outcome nhóm control
    Mu1Network        — đầu dự đoán outcome nhóm treatment
    TauNetwork        — đầu ước lượng ITE trực tiếp (tau)
    DESCNModel        — mô hình tổng hợp tích hợp 5 mạng con
    CriteoDataset     — torch.utils.data.Dataset cho Criteo Uplift v2.1

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
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def init_weights(m):
    """
    Khởi tạo trọng số nn.Linear theo phân phối chuẩn (Xavier-like).

    Công thức: std = 1 / sqrt(fan_in),  bias = 0.
    Áp dụng qua model.apply(init_weights) ngay sau khi khai báo module.

    Tham số
    -------
    m : nn.Module — chỉ áp dụng khi m là nn.Linear; các module khác bỏ qua.
    """
    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        stdv = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(m.weight, mean=0.0, std=stdv)
        m.bias.data.fill_(0.0)


def safe_sqrt(x):
    """
    Sqrt ổn định số học: clamp giá trị vào [1e-9, 1e9] trước khi lấy căn.

    Tránh NaN khi đầu vào âm hoặc rất nhỏ (thường gặp trong L2-normalize).

    Tham số
    -------
    x : Tensor — giá trị đầu vào (bất kỳ shape).

    Trả về
    ------
    Tensor — sqrt(clamp(x, 1e-9, 1e9)), cùng shape và device với x.
    """
    return torch.sqrt(torch.clamp(x, min=1e-9, max=1e9))


def gaussian_mmd(X, t, sigma: float = 1.0, max_samples: int = 512):
    """
    Maximum Mean Discrepancy (MMD) với Gaussian (RBF) kernel.

    Đo khoảng cách phân phối giữa biểu diễn ẩn (shared representation)
    của nhóm treatment (T=1) và nhóm control (T=0). Dùng làm covariate
    balance loss thay thế Wasserstein distance (tránh phụ thuộc geomloss).

    Công thức MMD unbiased:
        MMD²(p, q) = E_p[k(x,x')] + E_q[k(y,y')] − 2·E_{p,q}[k(x,y)]
        kernel: k(u,v) = exp(−‖u−v‖² / (2σ²))

    Triển khai hiệu quả bộ nhớ:
        ‖u−v‖² = ‖u‖² + ‖v‖² − 2·u·vᵀ  (BLAS matmul, không cần expand 3D)
        max_samples giới hạn kích thước kernel để tránh OOM với batch lớn.

    Tham số
    -------
    X           : Tensor (n, d) — biểu diễn ẩn chung (shared_h).
    t           : Tensor (n,) hoặc (n,1) — nhãn treatment binary (0/1).
    sigma       : float — độ rộng Gaussian kernel (mặc định 1.0).
    max_samples : int   — số mẫu tối đa mỗi nhóm khi tính kernel (mặc định 512).
                          Giảm giá trị này nếu gặp OOM.

    Trả về
    ------
    mmd_val : Tensor scalar — ước lượng MMD² (unbiased, có thể nhỏ âm).
              Trả về 0.0 nếu một nhóm có < 2 mẫu.
    """
    t_flat = t.reshape(-1)
    Xt = X[t_flat == 1]   # (n_t, d) — biểu diễn nhóm treatment
    Xc = X[t_flat == 0]   # (n_c, d) — biểu diễn nhóm control

    # Subsample để tránh OOM khi batch_size lớn
    if Xt.size(0) > max_samples:
        idx = torch.randperm(Xt.size(0), device=X.device)[:max_samples]
        Xt = Xt[idx]
    if Xc.size(0) > max_samples:
        idx = torch.randperm(Xc.size(0), device=X.device)[:max_samples]
        Xc = Xc[idx]

    n_t, n_c = Xt.size(0), Xc.size(0)
    if n_t < 2 or n_c < 2:
        return X.new_zeros(1).squeeze()

    def rbf_kernel(A, B):
        """Gaussian kernel matrix (n_A, n_B) dùng BLAS matmul."""
        A_sq = (A * A).sum(dim=1, keepdim=True)   # (n_A, 1)
        B_sq = (B * B).sum(dim=1, keepdim=True)   # (n_B, 1)
        sq_dist = A_sq + B_sq.T - 2.0 * (A @ B.T) # (n_A, n_B)
        return torch.exp(-sq_dist / (2.0 * sigma * sigma))

    K_tt = rbf_kernel(Xt, Xt)   # (n_t, n_t)
    K_cc = rbf_kernel(Xc, Xc)   # (n_c, n_c)
    K_tc = rbf_kernel(Xt, Xc)   # (n_t, n_c)

    # Unbiased: bỏ diagonal của K_tt và K_cc (tránh bias tự-tương quan)
    mmd = (
        (K_tt.sum() - K_tt.trace()) / (n_t * (n_t - 1))
        + (K_cc.sum() - K_cc.trace()) / (n_c * (n_c - 1))
        - 2.0 * K_tc.mean()
    )
    return mmd


# ══════════════════════════════════════════════════════════════════════════════
# BaseModel — DNN Backbone 3 lớp dùng chung cho tất cả prediction heads
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(nn.Module):
    """
    DNN backbone 3 lớp: Linear(d→d) → ELU → Dropout, lặp 3 lần.

    Được dùng làm nền tảng chung cho PrpsyNetwork, Mu0Network, Mu1Network,
    và TauNetwork. Tất cả đầu vào và đầu ra đều có chiều base_dim.

    Tham số
    -------
    base_dim : int   — số chiều đầu vào và đầu ra (= output của ShareNetwork).
    do_rate  : float — tỷ lệ dropout (mặc định 0.1).
    """

    def __init__(self, base_dim: int, do_rate: float = 0.1):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(base_dim, base_dim), nn.ELU(), nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim), nn.ELU(), nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim), nn.ELU(), nn.Dropout(p=do_rate),
        )
        self.dnn.apply(init_weights)

    def forward(self, x):
        return self.dnn(x)   # (n, base_dim)


# ══════════════════════════════════════════════════════════════════════════════
# ShareNetwork — Bộ mã hóa đặc trưng dùng chung (Shared Feature Encoder)
# ══════════════════════════════════════════════════════════════════════════════

class ShareNetwork(nn.Module):
    """
    Bộ mã hóa đặc trưng dùng chung (shared representation) cho tất cả 4 heads.

    Kiến trúc (khi use_bn=True, normalization='divide'):
        BatchNorm1d(input_dim)
        → Linear(input_dim → share_dim) → ELU → Dropout
        → Linear(share_dim  → share_dim) → ELU → Dropout
        → Linear(share_dim  → base_dim)  → ELU → Dropout
        → L2-normalize(shared_h)

    Tham số
    -------
    input_dim     : int  — số chiều đặc trưng đầu vào (12 với Criteo).
    share_dim     : int  — số neuron của các lớp ẩn trung gian (mặc định 128).
    base_dim      : int  — số chiều đầu ra (mặc định 64).
    do_rate       : float — tỷ lệ dropout (mặc định 0.1).
    use_bn        : bool — có dùng BatchNorm1d ở đầu vào không (mặc định True).
    normalization : str  — 'divide' → L2-normalize đầu ra; bất kỳ str khác → không.
    """

    def __init__(self, input_dim: int, share_dim: int = 128, base_dim: int = 64,
                 do_rate: float = 0.1, use_bn: bool = True,
                 normalization: str = "divide"):
        super().__init__()
        self.normalization = normalization

        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(input_dim))
        layers += [
            nn.Linear(input_dim, share_dim), nn.ELU(), nn.Dropout(p=do_rate),
            nn.Linear(share_dim,  share_dim), nn.ELU(), nn.Dropout(p=do_rate),
            nn.Linear(share_dim,  base_dim),  nn.ELU(), nn.Dropout(p=do_rate),
        ]
        self.dnn = nn.Sequential(*layers)
        self.dnn.apply(init_weights)

    def forward(self, x):
        h = self.dnn(x)   # (n, base_dim)
        if self.normalization == "divide":
            # L2-normalize mỗi hàng: h / ‖h‖₂  (tránh chia cho 0 bằng safe_sqrt)
            h = h / safe_sqrt(torch.sum(h * h, dim=1, keepdim=True))
        return h   # (n, base_dim)


# ══════════════════════════════════════════════════════════════════════════════
# Prediction Heads — 4 mạng con dự đoán (dùng chung cấu trúc BaseModel + Linear)
# ══════════════════════════════════════════════════════════════════════════════

class PrpsyNetwork(nn.Module):
    """
    Mạng ước lượng propensity score: P(T=1 | X).

    Kiến trúc: shared_h → BaseModel → Linear(base_dim, 1)
    Đầu ra: logit (trước sigmoid). Trong DESCNModel.forward():
        p_prpsy = clip(sigmoid(logit), 0.001, 0.999)

    Tham số
    -------
    base_dim : int   — chiều đặc trưng vào (= output ShareNetwork).
    do_rate  : float — tỷ lệ dropout.
    """

    def __init__(self, base_dim: int, do_rate: float = 0.1):
        super().__init__()
        self.backbone    = BaseModel(base_dim, do_rate)
        self.logit_layer = nn.Linear(base_dim, 1)
        self.logit_layer.apply(init_weights)

    def forward(self, x):
        return self.logit_layer(self.backbone(x))   # (n, 1) — propensity logit


class Mu0Network(nn.Module):
    """
    Mạng dự đoán outcome nhóm control: P(Y=1 | X, T=0).

    Kiến trúc: shared_h → BaseModel → Linear(base_dim, 1)
    Đầu ra: logit. Trong DESCNModel.forward():
        p_mu0 = sigmoid(logit)  →  p_h0

    Tham số
    -------
    base_dim : int   — chiều đặc trưng vào.
    do_rate  : float — tỷ lệ dropout.
    """

    def __init__(self, base_dim: int, do_rate: float = 0.1):
        super().__init__()
        self.backbone    = BaseModel(base_dim, do_rate)
        self.logit_layer = nn.Linear(base_dim, 1)
        self.logit_layer.apply(init_weights)

    def forward(self, x):
        return self.logit_layer(self.backbone(x))   # (n, 1) — control outcome logit


class Mu1Network(nn.Module):
    """
    Mạng dự đoán outcome nhóm treatment: P(Y=1 | X, T=1).

    Kiến trúc: shared_h → BaseModel → Linear(base_dim, 1)
    Đầu ra: logit. Trong DESCNModel.forward():
        p_mu1 = sigmoid(logit)  →  p_h1

    Tham số
    -------
    base_dim : int   — chiều đặc trưng vào.
    do_rate  : float — tỷ lệ dropout.
    """

    def __init__(self, base_dim: int, do_rate: float = 0.1):
        super().__init__()
        self.backbone    = BaseModel(base_dim, do_rate)
        self.logit_layer = nn.Linear(base_dim, 1)
        self.logit_layer.apply(init_weights)

    def forward(self, x):
        return self.logit_layer(self.backbone(x))   # (n, 1) — treatment outcome logit


class TauNetwork(nn.Module):
    """
    Mạng ước lượng ITE trực tiếp: τ(X) = E[Y(1) − Y(0) | X].

    Khác với uplift tính bằng hiệu p_mu1 - p_mu0, mạng này học trực tiếp
    từ tín hiệu cross-space (μ0+τ ≈ μ1 và μ1-τ ≈ μ0).

    Kiến trúc: shared_h → BaseModel → Linear(base_dim, 1)
    Đầu ra: tau_logit (raw logit, không qua sigmoid — tau có thể âm hoặc dương).

    Tham số
    -------
    base_dim : int   — chiều đặc trưng vào.
    do_rate  : float — tỷ lệ dropout.
    """

    def __init__(self, base_dim: int, do_rate: float = 0.1):
        super().__init__()
        self.backbone    = BaseModel(base_dim, do_rate)
        self.logit_layer = nn.Linear(base_dim, 1)
        self.logit_layer.apply(init_weights)

    def forward(self, x):
        return self.logit_layer(self.backbone(x))   # (n, 1) — tau logit (raw)


# ══════════════════════════════════════════════════════════════════════════════
# DESCNModel — Mô hình tổng hợp DESCN (tương đương ESX trong bản gốc)
# ══════════════════════════════════════════════════════════════════════════════

class DESCNModel(nn.Module):
    """
    Deep Entire Space Cross Networks (DESCN) — mô hình uplift đầy đủ.

    Tích hợp 5 mạng con trong một forward pass thống nhất. Tất cả prediction
    heads chia sẻ cùng một bộ mã hóa (ShareNetwork), giúp học biểu diễn
    đặc trưng giàu thông tin phục vụ đồng thời cho nhiều tác vụ.

    Sơ đồ forward:
        x ──► ShareNetwork(x)  →  shared_h
        shared_h ──► PrpsyNetwork  →  p_prpsy_logit
                 ├──► Mu0Network   →  mu0_logit → p_mu0 = p_h0
                 ├──► Mu1Network   →  mu1_logit → p_mu1 = p_h1
                 └──► TauNetwork   →  tau_logit
        p_estr = p_prpsy × p_h1
        p_escr = (1 − p_prpsy) × p_h0
        uplift = p_h1 − p_h0   ← dùng khi inference

    Đầu ra (tuple 12 phần tử):
        p_prpsy_logit : Tensor (n,1) — propensity logit (trước sigmoid)
        p_estr        : Tensor (n,1) — P(T=1) × P(Y=1|T=1)
        p_escr        : Tensor (n,1) — P(T=0) × P(Y=1|T=0)
        tau_logit     : Tensor (n,1) — ITE logit trực tiếp
        mu1_logit     : Tensor (n,1) — treatment outcome logit
        mu0_logit     : Tensor (n,1) — control outcome logit
        p_prpsy       : Tensor (n,1) — propensity score ∈ [0.001, 0.999]
        p_mu1         : Tensor (n,1) — P(Y=1|X,T=1) = sigmoid(mu1_logit)
        p_mu0         : Tensor (n,1) — P(Y=1|X,T=0) = sigmoid(mu0_logit)
        p_h1          : Tensor (n,1) — alias p_mu1 (treatment representation)
        p_h0          : Tensor (n,1) — alias p_mu0 (control representation)
        shared_h      : Tensor (n,d) — biểu diễn ẩn chung (dùng cho MMD loss)

    Tham số
    -------
    input_dim     : int   — số chiều feature (12 với Criteo f0…f11).
    share_dim     : int   — chiều ẩn ShareNetwork (mặc định 128).
    base_dim      : int   — chiều output ShareNetwork = input BaseModel (mặc định 64).
    do_rate       : float — tỷ lệ dropout (mặc định 0.1).
    use_bn        : bool  — có dùng BatchNorm1d ở đầu vào ShareNetwork (mặc định True).
    normalization : str   — 'divide' → L2-normalize shared_h (mặc định 'divide').
    """

    def __init__(self, input_dim: int, share_dim: int = 128, base_dim: int = 64,
                 do_rate: float = 0.1, use_bn: bool = True,
                 normalization: str = "divide"):
        super().__init__()
        self.share_network = ShareNetwork(
            input_dim, share_dim, base_dim,
            do_rate=do_rate, use_bn=use_bn, normalization=normalization,
        )
        self.prpsy_network = PrpsyNetwork(base_dim, do_rate)
        self.mu0_network   = Mu0Network(base_dim, do_rate)
        self.mu1_network   = Mu1Network(base_dim, do_rate)
        self.tau_network   = TauNetwork(base_dim, do_rate)

    def forward(self, x):
        # ── Bước 1: Mã hóa đặc trưng chung ───────────────────────────────────
        shared_h = self.share_network(x)            # (n, base_dim)

        # ── Bước 2: Propensity score ──────────────────────────────────────────
        p_prpsy_logit = self.prpsy_network(shared_h)                     # (n, 1)
        # Clip sigmoid vào (0.001, 0.999) để tránh log(0) trong loss
        p_prpsy = torch.clamp(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)  # (n, 1)

        # ── Bước 3: Outcome heads ─────────────────────────────────────────────
        mu1_logit = self.mu1_network(shared_h)      # (n, 1) — treatment logit
        mu0_logit = self.mu0_network(shared_h)      # (n, 1) — control logit
        p_mu1 = torch.sigmoid(mu1_logit)            # (n, 1) — P(Y=1|X,T=1)
        p_mu0 = torch.sigmoid(mu0_logit)            # (n, 1) — P(Y=1|X,T=0)
        p_h1, p_h0 = p_mu1, p_mu0                  # alias (theo tên trong paper)

        # ── Bước 4: Tau head ─────────────────────────────────────────────────
        tau_logit = self.tau_network(shared_h)      # (n, 1) — ITE logit raw

        # ── Bước 5: Entire-space sản phẩm ────────────────────────────────────
        p_estr = p_prpsy * p_h1                     # P(T=1, Y=1) = P(T=1)×P(Y=1|T=1)
        p_escr = (1.0 - p_prpsy) * p_h0            # P(T=0, Y=1) = P(T=0)×P(Y=1|T=0)

        return (
            p_prpsy_logit, p_estr,    p_escr,   tau_logit,
            mu1_logit,     mu0_logit,
            p_prpsy,       p_mu1,     p_mu0,    p_h1,    p_h0,
            shared_h
        )


# ══════════════════════════════════════════════════════════════════════════════
# CriteoDataset — torch.utils.data.Dataset cho Criteo Uplift v2.1
# ══════════════════════════════════════════════════════════════════════════════

class CriteoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrap dữ liệu Criteo Uplift v2.1 (đã qua preprocess).

    Mỗi mẫu trả về tuple (x, t, y, e) dạng float32 tensor, tương thích
    với ESXDataset gốc của DESCN.

    Convention về trường 'e' trong DESCN:
        e = 0 : mẫu "non-randomized" / observational → tính vào mọi loss.
        e = 1 : mẫu "randomized"                    → loại khỏi prpsy/estr/escr.
        Với Criteo (A/B test thuần túy), e = 0 cho tất cả mẫu vì dataset
        đã được thiết kế như thí nghiệm ngẫu nhiên có kiểm soát.

    Tham số
    -------
    X : np.ndarray float32 (n, 12) — ma trận đặc trưng f0…f11.
    y : np.ndarray (n,)           — nhãn kết quả binary (0 hoặc 1).
    t : np.ndarray (n,)           — nhãn treatment (0 hoặc 1).
    e : np.ndarray (n,) hoặc None — nhãn randomized; mặc định None → toàn bộ 0.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray,
                 e: np.ndarray = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32))
        if e is None:
            self.e = torch.zeros(len(X), dtype=torch.float32)
        else:
            self.e = torch.from_numpy(e.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Trả về (x, t, y, e) — thứ tự khớp với ESXDataset gốc
        return self.X[idx], self.t[idx], self.y[idx], self.e[idx]
