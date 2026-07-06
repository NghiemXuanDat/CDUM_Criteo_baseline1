"""
cevae.py — CEVAE (Causal Effect Variational AutoEncoder) adapted for Criteo
=============================================================================
Tham chiếu:
    Louizos, C., Shalit, U., Mooij, J., Sontag, D., Zemel, R., & Welling, M.
    "Causal Effect Inference with Deep Latent-Variable Models."
    NeurIPS 2017. https://arxiv.org/abs/1705.08821

    Code gốc (Pyro): https://github.com/AMLab-Amsterdam/CEVAE

Thích nghi cho Criteo Uplift v2.1:
    - 12 đặc trưng liên tục → likelihood Normal p(x|z)   (gốc: hỗn hợp binary+cont)
    - Kết quả nhị phân (visit 0/1) → likelihood Bernoulli p(y|t,z)
    - Không có nhãn counterfactual → đánh giá qua uplift metrics
    - Pure PyTorch (không dùng Pyro) — nhất quán với descn.py, ganite.py

Kiến trúc tổng quan:
─────────────────────────────────────────────────────────────────────────────
  [Training]
  x(12), t(1), y(1) ──► CEVAEEncoder ──► q(z|x,t,y) = N(z_loc, z_scale)
                                                │ rsample (reparameterization)
                         CEVAEDecoder ◄──────── z
                               │
              p(x|z) = N(μ_x(z), σ²_x)         — tái tạo features
              p(t|z) = Bernoulli(σ(ψ(z)))       — dự đoán treatment
              p(y|t,z): t=0 → Bernoulli(σ(φ₀(z)))
                        t=1 → Bernoulli(σ(φ₁(z)))

  [Inference — uplift prediction]
  x, t_obs, y_obs ──► Encoder ──► z ~ q(z|x,t_obs,y_obs)
                      Decoder: y0 = σ(φ₀(z)),  y1 = σ(φ₁(z))
                      uplift = E_z[y1 − y0]     — Monte Carlo averaging

Loss (negative ELBO, tối thiểu hóa):
    L = −E_q[log p(x|z) + log p(t|z) + log p(y|t,z)] + KL[q(z|x,t,y) ‖ p(z)]

Encoder q(z|x,t,y):
    h_qz    = MLP([x, y])                   ← trunk chung (n_hidden−1 lớp ẩn)
    z_loc   = t·Wₜ₁·h_qz + (1−t)·Wₜ₀·h_qz  ← dual-branch t-conditional
    z_scale = softplus(...)                  ← luôn dương

Decoder p(·|z):
    h_x     = MLP(z)                        ← trunk chung cho x
    x_loc   = Linear(h_x)                   ← N mean per feature
    x_scale = softplus(learnable param)     ← global per-feature scale
    t_logit = MLP_t(z)                      ← 1 lớp ẩn (theo bản gốc)
    y_logit_{0,1} = MLP_y{0,1}(z)          ← n_hidden lớp ẩn (theo bản gốc)

Các lớp công khai:
    kaiming_init()   — Kaiming Normal initialization cho nn.Linear
    FCNet            — Fully-connected network: Linear → ELU [→ Dropout], stacked
    CEVAEEncoder     — Inference network q(z|x,t,y): dual-branch t-conditional
    CEVAEDecoder     — Generative model p(x|z), p(t|z), p(y|t,z)
    CEVAE            — Mô hình tổng hợp với ELBO và uplift prediction
    CriteoDataset    — torch.utils.data.Dataset cho Criteo Uplift v2.1

Môi trường yêu cầu:
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence


# ══════════════════════════════════════════════════════════════════════════════
# Utility — Initialization
# ══════════════════════════════════════════════════════════════════════════════

def kaiming_init(m: nn.Module) -> None:
    """
    Khởi tạo trọng số nn.Linear theo Kaiming Normal (He initialization).

    Phù hợp với activation ELU/ReLU — giữ phương sai đầu ra ổn định.
    Bias được khởi tạo về 0.

    Tham số
    -------
    m : nn.Module — chỉ áp dụng khi m là nn.Linear; bỏ qua các module khác.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)


# ══════════════════════════════════════════════════════════════════════════════
# FCNet — Fully-connected backbone dùng chung cho Encoder và Decoder
# ══════════════════════════════════════════════════════════════════════════════

class FCNet(nn.Module):
    """
    Stacked fully-connected network: Linear → ELU [→ Dropout], lặp nhiều lớp.

    Dùng làm building block chung cho CEVAEEncoder và CEVAEDecoder.
    Activation lớp cuối có thể tùy chỉnh qua out_act (mặc định None → không có,
    lớp cuối là Linear thuần).

    Tham số
    -------
    in_dim       : int        — số chiều đầu vào.
    hidden_dims  : list[int]  — số neuron từng lớp ẩn; [] → chỉ có output layer.
    out_dim      : int        — số chiều đầu ra.
    out_act      : nn.Module  — activation lớp cuối (VD nn.ELU(), nn.Softplus()).
                               Mặc định None → không áp dụng activation lớp cuối.
    dropout_rate : float      — tỷ lệ dropout sau mỗi lớp ẩn (mặc định 0.0).
    """

    def __init__(self, in_dim: int, hidden_dims, out_dim: int,
                 out_act: nn.Module = None, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ELU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            d = h
        layers.append(nn.Linear(d, out_dim))
        if out_act is not None:
            layers.append(out_act)
        self.net = nn.Sequential(*layers)
        self.net.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# CEVAEEncoder — Inference network q(z|x,t,y)
# ══════════════════════════════════════════════════════════════════════════════

class CEVAEEncoder(nn.Module):
    """
    Mạng inference (amortized posterior) q(z|x,t,y).

    Kiến trúc dual-branch theo treatment:
    ─────────────────────────────────────────────────────────────────────
    Input: [x(n_features), y(1)] → concatenate → trunk h_qz
    Trunk: FCNet(n_features+1, [h]*(n_hidden−1), h, out_act=ELU)
              ↓ h (n, hidden_dim)
    Branch t=0:  z_loc_t0   = Linear(h, z_dim)
                 z_scale_t0 = softplus(Linear(h, z_dim)) + ε
    Branch t=1:  z_loc_t1   = Linear(h, z_dim)
                 z_scale_t1 = softplus(Linear(h, z_dim)) + ε

    Kết hợp (t-conditional):
        z_loc   = t · z_loc_t1   + (1−t) · z_loc_t0
        z_scale = t · z_scale_t1 + (1−t) · z_scale_t0

    Ghi chú:
        Bản gốc CEVAE (Pyro) cũng có thêm q(t|x) và q(y|x,t) trong encoder
        dùng như auxiliary networks ở test time. Phiên bản này bỏ qua vì
        tại test time ta có actual t và y để infer z trực tiếp.

    Tham số
    -------
    n_features   : int   — số đặc trưng đầu vào (12 với Criteo f0…f11).
    z_dim        : int   — số chiều không gian tiềm ẩn z.
    hidden_dim   : int   — số neuron mỗi lớp ẩn.
    n_hidden     : int   — số lớp ẩn trong trunk h_qz (không tính output layer).
    dropout_rate : float — tỷ lệ dropout sau mỗi lớp ẩn (mặc định 0.0).
    """

    def __init__(self, n_features: int, z_dim: int, hidden_dim: int,
                 n_hidden: int, dropout_rate: float = 0.0):
        super().__init__()

        # Trunk q(z|x,t,y): nhận [x, y], có ELU ở đầu ra để giữ nonlinearity
        in_dim = n_features + 1             # x(n_features) + y(1)
        self.h_qz = FCNet(
            in_dim,
            [hidden_dim] * (n_hidden - 1),  # (n_hidden−1) lớp ẩn trước output
            hidden_dim,
            out_act    = nn.ELU(),          # output vẫn qua ELU để là "hidden" repr
            dropout_rate = dropout_rate,
        )

        # Branch t=0: z_loc và log_scale (rồi softplus)
        self.z_loc_t0    = nn.Linear(hidden_dim, z_dim)
        self.z_lscale_t0 = nn.Linear(hidden_dim, z_dim)

        # Branch t=1
        self.z_loc_t1    = nn.Linear(hidden_dim, z_dim)
        self.z_lscale_t1 = nn.Linear(hidden_dim, z_dim)

        # Khởi tạo các linear heads
        for layer in [self.z_loc_t0, self.z_lscale_t0,
                      self.z_loc_t1, self.z_lscale_t1]:
            kaiming_init(layer)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor) -> tuple:
        """
        Tính tham số posterior q(z|x,t,y).

        Tham số
        -------
        x : Tensor (n, n_features) — features (đã chuẩn hóa nếu normalize=True).
        t : Tensor (n,)            — treatment binary (0 hoặc 1), float hoặc long.
        y : Tensor (n,)            — observed outcome binary (0 hoặc 1).

        Trả về
        ------
        z_loc   : Tensor (n, z_dim) — mean của posterior q(z|x,t,y).
        z_scale : Tensor (n, z_dim) — std của posterior q(z|x,t,y), luôn > 0.
        """
        # Ghép x và y thành input của trunk
        y_col = y.float().unsqueeze(-1)               # (n, 1)
        xy    = torch.cat([x, y_col], dim=-1)         # (n, n_features+1)
        h     = self.h_qz(xy)                         # (n, hidden_dim)

        # Tính loc và scale cho cả 2 branches
        loc_t0   = self.z_loc_t0(h)                   # (n, z_dim)
        loc_t1   = self.z_loc_t1(h)
        scale_t0 = F.softplus(self.z_lscale_t0(h)) + 1e-6   # luôn > 0
        scale_t1 = F.softplus(self.z_lscale_t1(h)) + 1e-6

        # T-conditional selection: t_col broadcast sang chiều z_dim
        t_col   = t.float().unsqueeze(-1)             # (n, 1) → broadcast với (n, z_dim)
        z_loc   = t_col * loc_t1   + (1.0 - t_col) * loc_t0
        z_scale = t_col * scale_t1 + (1.0 - t_col) * scale_t0

        return z_loc, z_scale


# ══════════════════════════════════════════════════════════════════════════════
# CEVAEDecoder — Generative model p(x|z), p(t|z), p(y|t,z)
# ══════════════════════════════════════════════════════════════════════════════

class CEVAEDecoder(nn.Module):
    """
    Mạng sinh (generative model) của CEVAE:

        p(x|z) = Normal(μ_x(z), diag(σ²_x))       — tái tạo continuous features
        p(t|z) = Bernoulli(σ(ψ(z)))                — dự đoán treatment
        p(y|t,z): t=0 → Bernoulli(σ(φ₀(z)))       — outcome khi control
                  t=1 → Bernoulli(σ(φ₁(z)))       — outcome khi treatment

    Sub-networks theo bản gốc CEVAE:
    ─────────────────────────────────────────────────────────────────────
    h_x:     FCNet(z_dim, [h]*(n_hidden−1), h, ELU)    — trunk cho p(x|z)
    x_loc:   Linear(h, n_features)
    x_scale: Softplus(learnable param (n_features,))    — global, không phụ thuộc z

    t_logit:  FCNet(z_dim, [h], 1)                      — 1 lớp ẩn (shallow, theo gốc)
    y0_logit: FCNet(z_dim, [h]*n_hidden, 1)             — n_hidden lớp ẩn (deep, theo gốc)
    y1_logit: FCNet(z_dim, [h]*n_hidden, 1)             — n_hidden lớp ẩn

    Tham số
    -------
    n_features   : int   — số đặc trưng đầu ra (12 với Criteo f0…f11).
    z_dim        : int   — số chiều không gian tiềm ẩn.
    hidden_dim   : int   — số neuron mỗi lớp ẩn.
    n_hidden     : int   — số lớp ẩn cho sub-networks (ít nhất 1).
    dropout_rate : float — tỷ lệ dropout (mặc định 0.0).
    """

    def __init__(self, n_features: int, z_dim: int, hidden_dim: int,
                 n_hidden: int, dropout_rate: float = 0.0):
        super().__init__()

        # ── p(x|z): Normal — trunk chung + linear head ─────────────────────────
        self.h_x = FCNet(
            z_dim,
            [hidden_dim] * (n_hidden - 1),
            hidden_dim,
            out_act      = nn.ELU(),
            dropout_rate = dropout_rate,
        )
        self.x_loc = nn.Linear(hidden_dim, n_features)
        # Learnable log-scale per feature — không phụ thuộc vào z (global parameter)
        # Khởi tạo = 0 → softplus(0) ≈ 0.693 (scale ≈ 0.693), tăng dần theo train
        self.x_log_scale = nn.Parameter(torch.zeros(n_features))

        # ── p(t|z): Bernoulli — shallow (1 lớp ẩn theo bản gốc) ──────────────
        self.t_logit_net = FCNet(
            z_dim, [hidden_dim], 1, dropout_rate=dropout_rate
        )

        # ── p(y|t=0,z): Bernoulli — deep (n_hidden lớp ẩn theo bản gốc) ───────
        self.y_logit_t0_net = FCNet(
            z_dim, [hidden_dim] * n_hidden, 1, dropout_rate=dropout_rate
        )

        # ── p(y|t=1,z): Bernoulli — deep ──────────────────────────────────────
        self.y_logit_t1_net = FCNet(
            z_dim, [hidden_dim] * n_hidden, 1, dropout_rate=dropout_rate
        )

        # Khởi tạo x_loc
        kaiming_init(self.x_loc)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Tính tham số generative từ z.

        Tham số
        -------
        z : Tensor (n, z_dim) — mẫu latent (từ posterior hoặc prior).

        Trả về
        ------
        x_loc    : Tensor (n, n_features) — mean Normal cho p(x|z).
        x_scale  : Tensor (n_features,)  — std Normal cho p(x|z), global, > 0.
        t_logit  : Tensor (n,)           — logit Bernoulli cho p(t|z).
        y0_logit : Tensor (n,)           — logit Bernoulli cho p(y|t=0,z).
        y1_logit : Tensor (n,)           — logit Bernoulli cho p(y|t=1,z).
        """
        # p(x|z)
        h_x    = self.h_x(z)                                # (n, hidden_dim)
        x_loc  = self.x_loc(h_x)                            # (n, n_features)
        x_scale = F.softplus(self.x_log_scale) + 1e-6       # (n_features,) — global

        # p(t|z)
        t_logit = self.t_logit_net(z).squeeze(-1)           # (n,)

        # p(y|t,z)
        y0_logit = self.y_logit_t0_net(z).squeeze(-1)       # (n,)
        y1_logit = self.y_logit_t1_net(z).squeeze(-1)       # (n,)

        return x_loc, x_scale, t_logit, y0_logit, y1_logit


# ══════════════════════════════════════════════════════════════════════════════
# CEVAE — Mô hình tổng hợp với ELBO và uplift prediction
# ══════════════════════════════════════════════════════════════════════════════

class CEVAE(nn.Module):
    """
    Causal Effect Variational AutoEncoder — thích nghi cho Criteo Uplift v2.1.

    Dựa trên Louizos et al. (2017), với các điều chỉnh:
        - Pure PyTorch, không dùng Pyro (nhất quán với DESCN, GANITE baselines).
        - Continuous features → Normal p(x|z) thay vì hỗn hợp Binary+Cont.
        - Binary outcome → Bernoulli p(y|t,z) thay vì Normal.
        - Uplift prediction tại test time dùng actual (t, y) qua encoder.

    Training ELBO:
        L = -E_q[log p(x|z) + log p(t|z) + log p(y|t,z)] + KL[q(z|x,t,y) ‖ p(z)]

    Uplift prediction:
        τ(x) = E_{z~q(z|x,t,y)}[σ(φ₁(z)) − σ(φ₀(z))]
        ước lượng bằng Monte Carlo với n_samples mẫu z.

    Tham số
    -------
    n_features   : int   — số đặc trưng đầu vào (12 với Criteo f0…f11).
    z_dim        : int   — số chiều không gian tiềm ẩn z (mặc định 20).
    hidden_dim   : int   — số neuron mỗi lớp ẩn (mặc định 128).
    n_hidden     : int   — số lớp ẩn cho sub-networks (mặc định 3, theo paper).
    dropout_rate : float — tỷ lệ dropout (mặc định 0.0; KL đủ là regularizer).
    """

    def __init__(self, n_features: int, z_dim: int = 20,
                 hidden_dim: int = 128, n_hidden: int = 3,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.encoder = CEVAEEncoder(
            n_features, z_dim, hidden_dim, n_hidden, dropout_rate
        )
        self.decoder = CEVAEDecoder(
            n_features, z_dim, hidden_dim, n_hidden, dropout_rate
        )
        self.z_dim = z_dim

    def elbo(self, x: torch.Tensor, t: torch.Tensor,
             y: torch.Tensor) -> torch.Tensor:
        """
        Tính −ELBO (negative Evidence Lower BOund) để minimize.

        ELBO = E_{z~q}[log p(x|z) + log p(t|z) + log p(y|t,z)]
                     − KL[q(z|x,t,y) ‖ p(z)=N(0,I)]

        KL giữa hai Normal được tính analytic:
            KL[N(μ,σ²) ‖ N(0,1)] = 0.5·(σ² + μ² − 1 − log σ²)

        Tham số
        -------
        x : Tensor (n, n_features) — features (đã chuẩn hóa nếu normalize=True).
        t : Tensor (n,)            — treatment binary (0 hoặc 1).
        y : Tensor (n,)            — observed outcome binary (0 hoặc 1).

        Trả về
        ------
        neg_elbo : Tensor scalar — −ELBO trung bình trên batch (để backward).
        """
        # ── Posterior q(z|x,t,y) ──────────────────────────────────────────────
        z_loc, z_scale = self.encoder(x, t, y)            # (n, z_dim) each

        # ── Sample z bằng reparameterization trick: z = μ + σ·ε, ε~N(0,I) ────
        q_z = Normal(z_loc, z_scale)
        z   = q_z.rsample()                               # (n, z_dim), differentiable

        # ── KL[q(z|x,t,y) ‖ p(z)=N(0,I)] — analytic, sum over z_dim ─────────
        p_z = Normal(torch.zeros_like(z_loc), torch.ones_like(z_scale))
        kl  = kl_divergence(q_z, p_z).sum(dim=-1)        # (n,)

        # ── Generative distributions từ decoder ───────────────────────────────
        x_loc, x_scale, t_logit, y0_logit, y1_logit = self.decoder(z)

        # log p(x|z): Normal, sum over n_features dimensions
        # x_scale broadcast: (n_features,) → (n, n_features)
        log_px = Normal(x_loc, x_scale).log_prob(x).sum(dim=-1)     # (n,)

        # log p(t|z): Bernoulli
        log_pt = Bernoulli(logits=t_logit).log_prob(t.float())       # (n,)

        # log p(y|t,z): Bernoulli, t-conditional — chọn branch theo t thực tế
        t_f     = t.float()
        y_logit = t_f * y1_logit + (1.0 - t_f) * y0_logit
        log_py  = Bernoulli(logits=y_logit).log_prob(y.float())      # (n,)

        # ── ELBO = E[log p] − KL, trả về −ELBO để minimize ──────────────────
        elbo = log_px + log_pt + log_py - kl                         # (n,)
        return -elbo.mean()

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor, t: torch.Tensor,
                       y: torch.Tensor,
                       n_samples: int = 10) -> tuple:
        """
        Dự đoán uplift score: τ(x) = E_z[σ(φ₁(z)) − σ(φ₀(z))].

        z được lấy mẫu từ q(z|x, t_obs, y_obs) — dùng actual treatment và
        outcome để infer latent z tốt nhất cho từng sample.

        Uplift tính bằng Monte Carlo averaging qua n_samples mẫu:
            y0 = (1/L) Σ_{l=1}^L σ(φ₀(z^{(l)}))
            y1 = (1/L) Σ_{l=1}^L σ(φ₁(z^{(l)}))
            uplift = y1 − y0

        Ghi chú:
            Trong bản gốc CEVAE, L=1 trong quá trình train, L=100 cho final eval.
            Với Criteo ~1.4M test samples, n_samples=10 (val) và 50 (final eval)
            là sự cân bằng giữa tốc độ và chính xác.

        Tham số
        -------
        x        : Tensor (n, n_features) — features (đã chuẩn hóa nếu cần).
        t        : Tensor (n,)            — treatment quan sát (0 hoặc 1).
        y        : Tensor (n,)            — outcome quan sát (0 hoặc 1).
        n_samples: int — số mẫu Monte Carlo (mặc định 10).

        Trả về
        ------
        y0_mean : Tensor (n,) — E[P(Y=1|t=0, z)] ước lượng.
        y1_mean : Tensor (n,) — E[P(Y=1|t=1, z)] ước lượng.
        """
        z_loc, z_scale = self.encoder(x, t, y)    # (n, z_dim)

        y0_acc = torch.zeros(x.size(0), device=x.device)
        y1_acc = torch.zeros(x.size(0), device=x.device)

        for _ in range(n_samples):
            # Lấy mẫu z từ posterior
            z = Normal(z_loc, z_scale).sample()                        # (n, z_dim)
            _, _, _, y0_logit, y1_logit = self.decoder(z)
            y0_acc = y0_acc + torch.sigmoid(y0_logit)
            y1_acc = y1_acc + torch.sigmoid(y1_logit)

        y0_mean = y0_acc / n_samples                                    # (n,)
        y1_mean = y1_acc / n_samples                                    # (n,)
        return y0_mean, y1_mean


# ══════════════════════════════════════════════════════════════════════════════
# CriteoDataset — torch.utils.data.Dataset cho Criteo Uplift v2.1
# ══════════════════════════════════════════════════════════════════════════════

class CriteoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrap dữ liệu Criteo Uplift v2.1 (đã qua preprocess).

    Mỗi mẫu trả về tuple (x, t, y) dạng float32 tensor, tương thích với
    training loop trong run_cevae_criteo.py.

    Convention CEVAE (giống GANITE, không cần trường 'e'):
        x : feature vector float32 (n_features,)
        t : treatment indicator float32 (0.0 hoặc 1.0)
        y : observed outcome float32 (0.0 hoặc 1.0)

    Tham số
    -------
    X : np.ndarray float32 (n, n_features) — ma trận đặc trưng f0…f11.
    y : np.ndarray (n,)                    — nhãn kết quả binary (0 hoặc 1).
    t : np.ndarray (n,)                    — nhãn treatment (0 hoặc 1).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Trả về (x, t, y) — thứ tự khớp với training loop trong run_cevae_criteo.py
        return self.X[idx], self.t[idx], self.y[idx]
