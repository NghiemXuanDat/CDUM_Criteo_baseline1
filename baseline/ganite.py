"""
ganite.py — Kiến trúc GANITE (Generative Adversarial Nets for Individual Treatment Effect)
===========================================================================================

Tham chiếu:
    Jinsung Yoon, James Jordon, Mihaela van der Schaar,
    "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
    International Conference on Learning Representations (ICLR), 2018.
    https://openreview.net/forum?id=ByKWUeWA-

Kiến trúc 3 mạng con:
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │  Phase 1 — GAN phase (huấn luyện G và D xen kẽ)                                        │
    │                                                                                         │
    │  X, T, Y  ──► Generator G  ──► [Ŷ₀_logit, Ŷ₁_logit]                                   │
    │                                       │                                                 │
    │                  Discriminator D ◄────┘  (nhận X + factual + counterfactual outcomes)  │
    │                  D dự đoán treatment T từ outcome bundle                                │
    │                  G muốn D không phân biệt được → adversarial loss                      │
    │                                                                                         │
    │  Phase 2 — Inference phase (huấn luyện I)                                              │
    │                                                                                         │
    │  X  ──► InferenceNet I  ──► [Ŷ₀_logit, Ŷ₁_logit]                                      │
    │         (dùng G làm pseudo-label teacher cho outcome không quan sát được)               │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

Loss functions:
    Phase 1:
        D_loss         = BCE_logits(D(X, outcomes), T)        — D học phân biệt treatment
        G_loss_factual = BCE_logits(G_factual_logit, Y)       — G tái tạo outcome quan sát được
        G_loss_GAN     = −D_loss                              — G đánh lừa D
        G_loss         = G_loss_factual + α × G_loss_GAN

    Phase 2:
        label_Y1 = T·Y + (1−T)·sigmoid(G_Ŷ₁)                — obs khi T=1, pseudo khi T=0
        label_Y0 = (1−T)·Y + T·sigmoid(G_Ŷ₀)                — obs khi T=0, pseudo khi T=1
        I_loss1  = BCE_logits(I_Ŷ₁_logit, label_Y1)
        I_loss2  = BCE_logits(I_Ŷ₀_logit, label_Y0)
        I_loss   = I_loss1 + I_loss2

Uplift score (inference):
    τ̂(X) = sigmoid(I_Ŷ₁_logit) − sigmoid(I_Ŷ₀_logit)

Các lớp công khai:
    xavier_normal_init()    — khởi tạo Xavier Normal cho nn.Linear
    GANITEGenerator         — G(X, T, Y) → (n, 2) logits [Y0_logit, Y1_logit]
    GANITEDiscriminator     — D(X, T, Y, gen_logits) → (n, 1) T_logit
    GANITEInferenceNet      — I(X) → (n, 2) logits [Y0_logit, Y1_logit]
    CriteoDataset           — torch.utils.data.Dataset cho Criteo Uplift v2.1

Ghi chú chuyển đổi từ bản gốc TF1:
    Bản gốc GANITE dùng TF1.x (tf.placeholder, tf.Session). File này viết lại
    bằng PyTorch để đồng bộ với môi trường umlc_env và nhất quán với baseline/descn.py.

Môi trường yêu cầu:
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Utility — Xavier initialization
# ══════════════════════════════════════════════════════════════════════════════

def xavier_normal_init(m: nn.Module) -> None:
    """
    Khởi tạo trọng số nn.Linear theo Xavier Normal.

    Áp dụng qua model.apply(xavier_normal_init) ngay sau khi khai báo module.
    Bias được khởi tạo về 0.

    Tham số
    -------
    m : nn.Module — chỉ áp dụng khi m là nn.Linear; bỏ qua các module khác.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Generator G(X, T, Y) — sinh counterfactual potential outcomes
# ══════════════════════════════════════════════════════════════════════════════

class GANITEGenerator(nn.Module):
    """
    Generator G: (X, T, Y) → [Ŷ₀_logit, Ŷ₁_logit].

    Nhận vào features X, treatment T, và observed outcome Y.
    Sinh ra ước lượng logit cho cả hai potential outcomes:
        Ŷ₀ = outcome khi T=0 (counterfactual cho T=1, factual cho T=0)
        Ŷ₁ = outcome khi T=1 (factual cho T=1, counterfactual cho T=0)

    Kiến trúc (giống bản gốc GANITE, multi-task output head):
        ┌─ concat(X, T, Y)  →  Linear(dim+2, h_dim) → ReLU
        │                   →  Linear(h_dim, h_dim)  → ReLU  ─► shared_h
        │
        ├─ Head Y0: shared_h → Linear(h_dim, h_dim) → ReLU → Linear(h_dim, 1)  [Ŷ₀_logit]
        └─ Head Y1: shared_h → Linear(h_dim, h_dim) → ReLU → Linear(h_dim, 1)  [Ŷ₁_logit]

        output: concat([Ŷ₀_logit, Ŷ₁_logit])  →  (n, 2)

    Tham số
    -------
    input_dim : int — số chiều feature X (12 với Criteo f0…f11).
    h_dim     : int — số neuron các lớp ẩn (hyperparameter cần tối ưu).
    """

    def __init__(self, input_dim: int, h_dim: int):
        super().__init__()

        # Trunk chung: nhận concat(X, T, Y)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + 2, h_dim),   # +2 vì T và Y là scalar
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # Head dự đoán Y khi T=0
        self.head_y0 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),               # logit (trước sigmoid)
        )

        # Head dự đoán Y khi T=1
        self.head_y1 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),               # logit (trước sigmoid)
        )

        self.apply(xavier_normal_init)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tham số
        -------
        x : Tensor (n, input_dim) — features đã chuẩn hóa.
        t : Tensor (n, 1)         — treatment indicator (0.0 hoặc 1.0).
        y : Tensor (n, 1)         — observed outcome binary (0.0 hoặc 1.0).

        Trả về
        ------
        logits : Tensor (n, 2) — [Ŷ₀_logit, Ŷ₁_logit] trước sigmoid.
                 Col 0: logit ước lượng Y khi T=0
                 Col 1: logit ước lượng Y khi T=1
        """
        # Ghép X, T, Y làm đầu vào
        inp = torch.cat([x, t, y], dim=1)          # (n, input_dim+2)
        h   = self.trunk(inp)                      # (n, h_dim)
        return torch.cat(
            [self.head_y0(h), self.head_y1(h)],    # (n, 1) mỗi head
            dim=1,
        )                                          # (n, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Discriminator D(X, outcomes) — phân biệt treatment từ outcome bundle
# ══════════════════════════════════════════════════════════════════════════════

class GANITEDiscriminator(nn.Module):
    """
    Discriminator D: (X, T, Y, gen_logits) → T_logit.

    Phân biệt treatment T từ bộ outcomes (factual + counterfactual):
        input0 = (1−T)·Y + T·Ŷ₀   — control outcome:   thật khi T=0, sinh khi T=1
        input1 = T·Y + (1−T)·Ŷ₁   — treatment outcome: thật khi T=1, sinh khi T=0
        D_input = concat(X, input0, input1)  →  (n, dim+2)

    Ý nghĩa adversarial game:
        D muốn phân biệt treatment T từ outcomes (D_loss = BCE(D(X, out), T))
        G muốn D không phân biệt được (G_loss_GAN = −D_loss)

    Kiến trúc:
        D_input  →  Linear(dim+2, h_dim) → ReLU
                 →  Linear(h_dim, h_dim)  → ReLU
                 →  Linear(h_dim, 1)      →  T_logit  (n, 1)

    Tham số
    -------
    input_dim : int — số chiều feature X (12 với Criteo f0…f11).
    h_dim     : int — số neuron các lớp ẩn.
    """

    def __init__(self, input_dim: int, h_dim: int):
        super().__init__()

        # Discriminator nhận X + input0 + input1 = dim + 2 chiều
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),               # logit dự đoán T (trước sigmoid)
        )
        self.apply(xavier_normal_init)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        gen_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tham số
        -------
        x          : Tensor (n, input_dim) — features.
        t          : Tensor (n, 1)         — treatment indicator (0.0 hoặc 1.0).
        y          : Tensor (n, 1)         — observed outcome (0.0 hoặc 1.0).
        gen_logits : Tensor (n, 2)         — Generator output [Ŷ₀_logit, Ŷ₁_logit].
                     Được detach khi huấn luyện D để grad không chảy về G.

        Trả về
        ------
        t_logit : Tensor (n, 1) — logit dự đoán treatment T (trước sigmoid).
        """
        # Chuyển logit thành xác suất để tạo bộ outcomes
        y_hat = torch.sigmoid(gen_logits)      # (n, 2) — Ŷ₀ và Ŷ₁ ∈ (0, 1)
        y0_hat = y_hat[:, 0:1]                 # (n, 1) — ước lượng outcome khi T=0
        y1_hat = y_hat[:, 1:2]                 # (n, 1) — ước lượng outcome khi T=1

        # Bộ outcomes: kết hợp factual (quan sát được) và counterfactual (sinh ra)
        input0 = (1.0 - t) * y + t * y0_hat   # (n, 1) — control outcome bundle
        input1 = t * y + (1.0 - t) * y1_hat   # (n, 1) — treatment outcome bundle

        d_inp = torch.cat([x, input0, input1], dim=1)  # (n, dim+2)
        return self.net(d_inp)                         # (n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# InferenceNet I(X) — mạng inference cuối để dự đoán potential outcomes
# ══════════════════════════════════════════════════════════════════════════════

class GANITEInferenceNet(nn.Module):
    """
    Inference Network I: X → [Ŷ₀_logit, Ŷ₁_logit].

    Mạng cuối dùng để dự đoán potential outcomes chỉ từ X (không cần T, Y).
    Uplift score được tính:
        τ̂(X) = sigmoid(Ŷ₁_logit) − sigmoid(Ŷ₀_logit)

    Huấn luyện dùng Generator làm pseudo-label teacher:
        label_Y1 = T·Y + (1−T)·G_sigmoid(Ŷ₁)   — obs khi T=1, pseudo khi T=0
        label_Y0 = (1−T)·Y + T·G_sigmoid(Ŷ₀)   — obs khi T=0, pseudo khi T=1
        I_loss = BCE(I_Y1_logit, label_Y1) + BCE(I_Y0_logit, label_Y0)

    Kiến trúc (giống Generator nhưng không nhận T, Y):
        X  →  Linear(dim, h_dim) → ReLU
           →  Linear(h_dim, h_dim) → ReLU  →  shared_h
        ├─ Head Y0: shared_h → Linear(h_dim, h_dim) → ReLU → Linear(h_dim, 1)
        └─ Head Y1: shared_h → Linear(h_dim, h_dim) → ReLU → Linear(h_dim, 1)
        output: concat([Ŷ₀_logit, Ŷ₁_logit]) → (n, 2)

    Tham số
    -------
    input_dim : int — số chiều feature X (12 với Criteo f0…f11).
    h_dim     : int — số neuron các lớp ẩn.
    """

    def __init__(self, input_dim: int, h_dim: int):
        super().__init__()

        # Trunk: chỉ nhận X (không có T, Y)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # Head dự đoán Y khi T=0
        self.head_y0 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

        # Head dự đoán Y khi T=1
        self.head_y1 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

        self.apply(xavier_normal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số
        -------
        x : Tensor (n, input_dim) — features.

        Trả về
        ------
        logits : Tensor (n, 2) — [Ŷ₀_logit, Ŷ₁_logit] trước sigmoid.
                 Col 0: logit ước lượng Y khi T=0
                 Col 1: logit ước lượng Y khi T=1
        """
        h = self.trunk(x)                              # (n, h_dim)
        return torch.cat(
            [self.head_y0(h), self.head_y1(h)],
            dim=1,
        )                                              # (n, 2)


# ══════════════════════════════════════════════════════════════════════════════
# CriteoDataset — torch.utils.data.Dataset cho Criteo Uplift v2.1
# ══════════════════════════════════════════════════════════════════════════════

class CriteoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrap dữ liệu Criteo Uplift v2.1 (đã qua preprocess).

    Mỗi mẫu trả về tuple (x, t, y) dạng float32 tensor, dùng cho
    DataLoader trong training loop của GANITE.

    Convention GANITE (khác DESCN, không cần trường 'e'):
        x : feature vector (float32, shape=input_dim)
        t : treatment indicator (float32, giá trị 0.0 hoặc 1.0)
        y : observed outcome (float32, giá trị 0.0 hoặc 1.0)

    Tham số
    -------
    X : np.ndarray float32 (n, input_dim) — ma trận đặc trưng f0…f11.
    y : np.ndarray (n,)                   — nhãn kết quả binary (0 hoặc 1).
    t : np.ndarray (n,)                   — nhãn treatment (0 hoặc 1).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Trả về (x, t, y) — thứ tự khớp với training loop trong run_ganite_criteo.py
        return self.X[idx], self.t[idx], self.y[idx]
