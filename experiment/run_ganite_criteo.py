"""
run_ganite_criteo.py — Pipeline huấn luyện và đánh giá GANITE trên Criteo Uplift v2.1
======================================================================================

Orchestrate toàn bộ pipeline:
  1. Load dataset từ file .csv.gz
  2. Split 8:1:1 (train / val / test) thông qua preprocess.data_loader
  3. Chuyển đổi Criteo pandas DataFrame → NumPy float32 (+ tùy chọn chuẩn hóa)
  4. Khởi tạo Generator, Discriminator, InferenceNet từ đầu (KHÔNG dùng pretrained weights)
  5. Phase 1 — GAN phase: huấn luyện G và D xen kẽ (2 bước D / 1 bước G mỗi batch)
  6. Phase 2 — Inference phase: huấn luyện InferenceNet dùng G làm pseudo-label teacher
  7. Đánh giá AUUC, QINI, LIFT@30% thông qua metrics.uplift_metrics

Kiến trúc GANITE:
    G(X,T,Y) → [Ŷ₀_logit, Ŷ₁_logit]  — Generator sinh counterfactual outcomes
    D(X, outcome_bundle) → T_logit     — Discriminator phân biệt treatment từ outcomes
    I(X) → [Ŷ₀_logit, Ŷ₁_logit]       — InferenceNet cho uplift score cuối cùng

Cấu hình:
    Chỉnh sửa class GANITEConfig ở phần "Cấu hình toàn cục" bên dưới.

Phụ thuộc nội bộ:
    from preprocess.data_loader  import load_dataset, split_dataset
    from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
    from baseline.ganite         import GANITEGenerator, GANITEDiscriminator,
                                        GANITEInferenceNet, CriteoDataset

Multi-GPU (tùy chọn):
    - 1 GPU  : mặc định, device='cuda:0' (RTX 3060)
    - 2 GPU  : đặt cfg.multi_gpu = True → DataParallel trên tất cả GPU (RTX 4090×2)
    Mỗi mạng con (G, D, I) được bọc riêng trong DataParallel.
    Checkpoint lưu state_dict của model bên trong (không có prefix 'module.').

Feature normalization:
    cfg.normalize = True (mặc định) → áp dụng StandardScaler từ sklearn trên
    training set. Scaler được fit trên train và transform lên val/test để tránh
    data leakage. Cần thiết vì GANITE không có BatchNorm bên trong mạng.

Môi trường yêu cầu:
    conda activate umlc_env
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.3
    scikit-learn 1.8.0, tensorboard

Cách chạy (từ thư mục gốc /home/datnghiemxuan/Documents/Criteo_CDUM/):
    conda activate umlc_env
    python -m experiment.run_ganite_criteo
    # hoặc:
    python experiment/run_ganite_criteo.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import os
import random
import sys
from dataclasses import dataclass

# Thêm thư mục gốc vào sys.path để import các module nội bộ
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Nội bộ ───────────────────────────────────────────────────────────────────
from preprocess.data_loader  import load_dataset, split_dataset
from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
from baseline.ganite         import (
    GANITEGenerator,
    GANITEDiscriminator,
    GANITEInferenceNet,
    CriteoDataset,
)


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Đường dẫn toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH       = '/home/datnghiemxuan/Documents/Criteo_CDUM/data/criteo-uplift-v2.1.csv.gz'
RESULTS_DIR     = '/home/datnghiemxuan/Documents/Criteo_CDUM/results'
OUTPUT_DIR      = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/GANITE/checkpoints'
TENSORBOARD_DIR = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/GANITE/runs'
MODEL_NAME      = 'ganite_criteo'

FEATURE_COLS = [f'f{i}' for i in range(12)]   # f0 … f11
LABEL_COL    = 'visit'
TREAT_COL    = 'treatment'


# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GANITEConfig:
    """
    Siêu tham số huấn luyện GANITE trên Criteo Uplift v2.1.

    Kiến trúc mạng
    --------------
    h_dim : số neuron các lớp ẩn trong G, D, I (mặc định 64).
            Bản gốc GANITE dùng 30 cho Twin (30 features, 11K samples).
            Criteo có 12 features và ~14M samples → 64 là lựa chọn cân bằng.

    Huấn luyện — Phase 1 (GAN)
    --------------------------
    epochs_gan    : số epoch GAN (mặc định 15).
                    Criteo có positive rate Y=1 chỉ 4.7% và treatment ratio 5.67:1,
                    cần nhiều epoch hơn để GAN hội tụ ra ngoài trivial near-zero solution.
    n_disc_steps  : số bước D update mỗi G update (mặc định 2, theo bản gốc).
    lr_gan        : learning rate cho G và D (mặc định 1e-3).
    alpha         : trọng số adversarial loss trong G_loss (mặc định 1.0, theo bản gốc).

    Huấn luyện — Phase 2 (Inference)
    ----------------------------------
    epochs_inf    : số epoch Inference phase (mặc định 15).
    lr_inf        : learning rate cho InferenceNet (mặc định 1e-3).

    Data
    ----
    batch_size    : kích thước mini-batch (mặc định 4096, lớn hơn bản gốc 256
                    vì Criteo có ~14M samples, cần batch lớn hơn để ổn định).
    num_workers   : số worker DataLoader (mặc định 4; đặt 0 nếu debug).
    normalize     : True → áp dụng StandardScaler (mặc định True, vì GANITE
                    không có BatchNorm bên trong mạng, cần chuẩn hóa ngoài).

    Khác
    ----
    log_step    : log TensorBoard sau mỗi N global step (mặc định 200).
    device      : 'auto' (tự phát hiện GPU/CPU), 'cuda:0', 'cuda:1', 'cpu'.
    multi_gpu   : True → DataParallel trên tất cả GPU (mặc định False).
    seed        : random seed (mặc định 42).
    verbose     : 1 → in log chi tiết; 0 → chỉ in kết quả cuối.
    """
    # Kiến trúc mạng
    h_dim: int = 64

    # GAN phase
    epochs_gan:   int   = 15
    n_disc_steps: int   = 2
    lr_gan:       float = 1e-3
    alpha:        float = 1.0

    # Inference phase
    epochs_inf: int   = 15
    lr_inf:     float = 1e-3

    # Data
    batch_size:  int  = 4096
    num_workers: int  = 4
    normalize:   bool = True

    # Khác
    log_step:  int  = 200
    device:    str  = "auto"
    multi_gpu: bool = False
    seed:      int  = 42
    verbose:   int  = 1


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42) -> None:
    """
    Cố định toàn bộ random seed để đảm bảo tái tạo kết quả (reproducibility).

    Áp dụng cho: Python random, NumPy, PyTorch (CPU + tất cả GPU).

    Tham số
    -------
    seed : int — giá trị seed (mặc định 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# Device setup
# ══════════════════════════════════════════════════════════════════════════════

def get_device(cfg: GANITEConfig) -> torch.device:
    """
    Xác định thiết bị tính toán (CPU / GPU) từ cấu hình.

    Khi cfg.device = 'auto': tự động chọn cuda:0 nếu có GPU, ngược lại CPU.

    Tham số
    -------
    cfg : GANITEConfig.

    Trả về
    ------
    device : torch.device
    """
    if cfg.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    log.info(f'Thiết bị tính toán : {device}')
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        log.info(f'  Số GPU phát hiện  : {n_gpu}')
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            log.info(f'  GPU {i}: {props.name}  '
                     f'({props.total_memory / 1024 ** 3:.1f} GB VRAM)')
    return device


# ══════════════════════════════════════════════════════════════════════════════
# Chuẩn bị dữ liệu — Criteo pandas → NumPy float32 (+ chuẩn hóa tùy chọn)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_criteo_split(x_df, y_series, t_series):
    """
    Chuyển đổi một split Criteo từ pandas sang numpy float32 cho GANITE.

    Tham số
    -------
    x_df     : pd.DataFrame (n, 12) — features f0…f11.
    y_series : pd.Series    (n,)    — binary outcome (0/1).
    t_series : pd.Series    (n,)    — binary treatment (0/1).

    Trả về
    ------
    X : np.ndarray float32 (n, 12)
    y : np.ndarray float32 (n,)
    t : np.ndarray float32 (n,)
    """
    X = x_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=np.float32)
    t = t_series.to_numpy(dtype=np.float32)
    return X, y, t


def normalize_features(X_train, X_val, X_test):
    """
    Chuẩn hóa features bằng StandardScaler (fit trên train, transform trên val/test).

    GANITE không có BatchNorm bên trong mạng, nên cần chuẩn hóa ngoài để
    đảm bảo các feature có cùng thang đo, giúp hội tụ ổn định hơn.

    Scaler được fit CHỈ trên X_train để tránh data leakage.

    Tham số
    -------
    X_train : np.ndarray (n_train, 12) — features tập train.
    X_val   : np.ndarray (n_val, 12)   — features tập val.
    X_test  : np.ndarray (n_test, 12)  — features tập test.

    Trả về
    ------
    X_train_norm, X_val_norm, X_test_norm : np.ndarray float32
    scaler : StandardScaler đã fit (để lưu lại nếu cần)
    """
    scaler  = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train).astype(np.float32)
    X_val_norm   = scaler.transform(X_val).astype(np.float32)
    X_test_norm  = scaler.transform(X_test).astype(np.float32)
    return X_train_norm, X_val_norm, X_test_norm, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions — GAN phase
# ══════════════════════════════════════════════════════════════════════════════

def compute_discriminator_loss(
    generator: nn.Module,
    discriminator: nn.Module,
    x_b: torch.Tensor,
    t_b: torch.Tensor,
    y_b: torch.Tensor,
    t_pos_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Tính D_loss cho một mini-batch.

    D_loss = BCE_logits(D(X, outcomes), T, pos_weight=t_pos_weight)

    t_pos_weight = n_ctrl / n_treat (< 1 vì Criteo có 5.67:1 T/C ratio).
    Down-weight class T=1 (majority) để D buộc phải học từ features thực sự,
    thay vì chỉ predict T=1 luôn và đạt 85% accuracy một cách trivial.

    Generator được chạy trong torch.no_grad() để grad không chảy về G khi
    huấn luyện D, tiết kiệm bộ nhớ.

    Tham số
    -------
    generator     : GANITEGenerator (hoặc DataParallel wrapper).
    discriminator : GANITEDiscriminator (hoặc DataParallel wrapper).
    x_b          : Tensor (n, d) — features batch.
    t_b          : Tensor (n, 1) — treatment batch.
    y_b          : Tensor (n, 1) — outcome batch.
    t_pos_weight : Tensor scalar — pos_weight cho T=1 class trong BCE.

    Trả về
    ------
    d_loss : Tensor scalar — binary cross entropy của D.
    """
    # Generator forward không cần gradient khi huấn luyện D
    with torch.no_grad():
        gen_logits = generator(x_b, t_b, y_b)    # (n, 2)

    # Detach để chắc chắn gen_logits không kéo grad về G
    d_logit = discriminator(x_b, t_b, y_b, gen_logits.detach())  # (n, 1)

    # D cố phân biệt T từ outcomes → labels là T, dùng pos_weight để cân bằng T/C imbalance
    d_loss  = F.binary_cross_entropy_with_logits(d_logit, t_b, pos_weight=t_pos_weight)
    return d_loss


def compute_generator_loss(
    generator: nn.Module,
    discriminator: nn.Module,
    x_b: torch.Tensor,
    t_b: torch.Tensor,
    y_b: torch.Tensor,
    alpha: float,
    t_pos_weight: torch.Tensor,
) -> tuple:
    """
    Tính G_loss cho một mini-batch.

    G_loss = G_loss_factual + alpha × G_loss_GAN
        G_loss_factual = BCE_logits(G_factual_logit, Y)
            — G tái tạo đúng outcome quan sát được (plain BCE, không dùng pos_weight
              để G học phân phối tự nhiên P(Y=1|X,T) ≈ 4.7%, giúp pseudo-label
              counterfactual cho InferenceNet được calibrate đúng).
        G_loss_GAN = -BCE_logits(D(X, outcomes), T, pos_weight=t_pos_weight)
            — G đánh lừa D (D không phân biệt được treatment)

    G_factual_logit = T × G_Y1_logit + (1−T) × G_Y0_logit

    Ghi chú: Không dùng y_pos_weight trong factual loss vì nếu up-weight Y=1 ×20
    thì gradient của G bị chi phối bởi 4.7% mẫu tích cực, làm G học G_Y0 ≈ G_Y1 ≈ cao
    cho mẫu Y=1 treated → uplift pseudo-label ≈ 0 → InferenceNet không học được
    sự khác biệt giữa T=0 và T=1 → QINI sụp đổ.

    D không cập nhật trong bước này.
    Gradient chảy từ d_logit → gen_logits → G parameters.

    Tham số
    -------
    generator     : GANITEGenerator.
    discriminator : GANITEDiscriminator.
    x_b          : Tensor (n, d) — features.
    t_b          : Tensor (n, 1) — treatment.
    y_b          : Tensor (n, 1) — outcome.
    alpha        : float — trọng số GAN loss.
    t_pos_weight : Tensor scalar — pos_weight cho T=1 trong adversarial BCE
                   (giữ lại để D/G adversarial game không bị trivial với Criteo
                   có T=1 chiếm 85%).

    Trả về
    ------
    g_loss         : Tensor scalar — tổng G loss.
    g_loss_factual : float — thành phần factual loss.
    g_loss_gan     : float — thành phần adversarial loss.
    """
    # Generator forward với gradient (để backprop về G)
    gen_logits = generator(x_b, t_b, y_b)        # (n, 2)

    # D forward — D không step nên param D không đổi, nhưng grad chảy về G
    d_logit = discriminator(x_b, t_b, y_b, gen_logits)  # (n, 1)

    # Adversarial loss: G muốn đảo ngược D_loss → -D_loss
    g_loss_gan = -F.binary_cross_entropy_with_logits(d_logit, t_b, pos_weight=t_pos_weight)

    # Factual loss: G phải dự đoán đúng outcome quan sát được.
    # Dùng plain BCE (không pos_weight) — giống bản gốc GANITE ICLR 2018.
    # Lấy logit tương ứng với treatment thực tế:
    #   T=1 → dùng gen_logits[:,1] (Y1 prediction)
    #   T=0 → dùng gen_logits[:,0] (Y0 prediction)
    factual_logit = t_b * gen_logits[:, 1:2] + (1.0 - t_b) * gen_logits[:, 0:1]
    g_loss_factual = F.binary_cross_entropy_with_logits(factual_logit, y_b)

    g_loss = g_loss_factual + alpha * g_loss_gan
    return g_loss, g_loss_factual.item(), g_loss_gan.item()


# ══════════════════════════════════════════════════════════════════════════════
# Loss function — Inference phase
# ══════════════════════════════════════════════════════════════════════════════

def compute_inference_loss(
    generator: nn.Module,
    inference_net: nn.Module,
    x_b: torch.Tensor,
    t_b: torch.Tensor,
    y_b: torch.Tensor,
) -> tuple:
    """
    Tính I_loss cho một mini-batch.

    InferenceNet học từ cả observed outcomes và pseudo-labels từ Generator:
        label_Y1 = T·Y + (1−T)·sigmoid(G_Ŷ₁)   — obs khi T=1, pseudo khi T=0
        label_Y0 = (1−T)·Y + T·sigmoid(G_Ŷ₀)   — obs khi T=0, pseudo khi T=1

        I_loss1 = BCE_logits(I_Ŷ₁_logit, label_Y1)
        I_loss2 = BCE_logits(I_Ŷ₀_logit, label_Y0)
        I_loss  = I_loss1 + I_loss2

    Dùng plain BCE (không pos_weight) — giống bản gốc GANITE ICLR 2018.
    Nếu dùng y_pos_weight ≈ 20 ở đây thì InferenceNet sẽ học cả I_Y1 lẫn I_Y0
    đều cao cho mẫu Y=1, thu hẹp variance của uplift = I_Y1 - I_Y0, dẫn đến
    phân hạng kém và QINI rất thấp.

    Generator được chạy trong torch.no_grad() vì ta chỉ dùng output của G
    làm pseudo-label (không cần backprop về G trong phase này).

    Tham số
    -------
    generator    : GANITEGenerator (weights đã được fix, không update).
    inference_net: GANITEInferenceNet.
    x_b          : Tensor (n, d) — features.
    t_b          : Tensor (n, 1) — treatment.
    y_b          : Tensor (n, 1) — outcome.

    Trả về
    ------
    i_loss  : Tensor scalar — tổng Inference loss.
    i_loss1 : float — thành phần loss cho Y1.
    i_loss2 : float — thành phần loss cho Y0.
    """
    # Generator tạo pseudo-labels (no_grad: không update G trong phase này)
    with torch.no_grad():
        gen_logits = generator(x_b, t_b, y_b)     # (n, 2)
        gen_probs  = torch.sigmoid(gen_logits)     # (n, 2) — G_Ŷ₀, G_Ŷ₁ ∈ (0, 1)
        y0_pseudo  = gen_probs[:, 0:1]             # (n, 1) — G_Ŷ₀ (pseudo-label cho Y0)
        y1_pseudo  = gen_probs[:, 1:2]             # (n, 1) — G_Ŷ₁ (pseudo-label cho Y1)

    # Nhãn tổng hợp (observed khi nhóm tương ứng, pseudo khi không)
    label_y1 = t_b * y_b + (1.0 - t_b) * y1_pseudo   # (n, 1)
    label_y0 = (1.0 - t_b) * y_b + t_b * y0_pseudo   # (n, 1)

    # InferenceNet forward (có gradient → backprop về I)
    inf_logits = inference_net(x_b)               # (n, 2)
    inf_y0_logit = inf_logits[:, 0:1]             # (n, 1) — I_Ŷ₀
    inf_y1_logit = inf_logits[:, 1:2]             # (n, 1) — I_Ŷ₁

    # Plain BCE — không pos_weight (xem docstring)
    i_loss1 = F.binary_cross_entropy_with_logits(inf_y1_logit, label_y1)
    i_loss2 = F.binary_cross_entropy_with_logits(inf_y0_logit, label_y0)
    i_loss  = i_loss1 + i_loss2

    return i_loss, i_loss1.item(), i_loss2.item()


# ══════════════════════════════════════════════════════════════════════════════
# Training — Phase 1: GAN phase (một epoch)
# ══════════════════════════════════════════════════════════════════════════════

def train_gan_epoch(
    generator:     nn.Module,
    discriminator: nn.Module,
    loader:        DataLoader,
    g_optimizer:   torch.optim.Optimizer,
    d_optimizer:   torch.optim.Optimizer,
    cfg:           GANITEConfig,
    device:        torch.device,
    writer:        SummaryWriter,
    epoch:         int,
    global_step:   list,
    t_pos_weight:  torch.Tensor,
) -> tuple:
    """
    Chạy một epoch của GAN phase (huấn luyện G và D xen kẽ).

    Với mỗi batch:
        1. n_disc_steps lần: huấn luyện Discriminator
        2. 1 lần: huấn luyện Generator

    Gradient clipping (max_norm=5.0) được áp dụng cho cả G và D sau backward()
    để ngăn gradient explode khi huấn luyện GAN trên Criteo scale.

    Tham số
    -------
    generator     : GANITEGenerator (hoặc DataParallel).
    discriminator : GANITEDiscriminator (hoặc DataParallel).
    loader        : DataLoader tập train.
    g_optimizer   : Adam optimizer cho Generator.
    d_optimizer   : Adam optimizer cho Discriminator.
    cfg           : GANITEConfig.
    device        : torch.device.
    writer        : SummaryWriter (TensorBoard).
    epoch         : chỉ số epoch hiện tại (0-indexed).
    global_step   : list[int] — biến mutable đếm tổng số batch GAN.
    t_pos_weight  : Tensor scalar — pos_weight cho BCE treatment (D và G adversarial).
                    Giữ lại để D/G game không bị trivial với Criteo (T=1 chiếm 85%).

    Trả về
    ------
    avg_d_loss : float — D loss trung bình epoch.
    avg_g_loss : float — G loss trung bình epoch.
    """
    generator.train()
    discriminator.train()

    running_d = 0.0
    running_g = 0.0
    n_batches = 0

    for x_b, t_b, y_b in loader:
        # Bỏ batch cuối quá nhỏ để tránh BatchNorm unstable (nếu có)
        if x_b.size(0) < cfg.batch_size:
            continue

        x_b = x_b.to(device)
        t_b = t_b.to(device).unsqueeze(1)    # (n, 1)
        y_b = y_b.to(device).unsqueeze(1)    # (n, 1)

        # ── Huấn luyện Discriminator (n_disc_steps lần) ───────────────────────
        d_loss_last = 0.0
        for _ in range(cfg.n_disc_steps):
            d_loss = compute_discriminator_loss(
                generator, discriminator, x_b, t_b, y_b, t_pos_weight
            )
            d_optimizer.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            d_optimizer.step()
            d_loss_last = d_loss.item()

        # ── Huấn luyện Generator (1 lần) ─────────────────────────────────────
        g_loss, g_fact, g_adv = compute_generator_loss(
            generator, discriminator, x_b, t_b, y_b, cfg.alpha,
            t_pos_weight,
        )
        g_optimizer.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        g_optimizer.step()

        running_d += d_loss_last
        running_g += g_loss.item()
        n_batches += 1
        global_step[0] += 1

        # Log TensorBoard theo log_step
        if global_step[0] % cfg.log_step == 0:
            writer.add_scalar('gan/d_loss',       d_loss_last, global_step[0])
            writer.add_scalar('gan/g_loss',        g_loss.item(), global_step[0])
            writer.add_scalar('gan/g_loss_factual', g_fact,    global_step[0])
            writer.add_scalar('gan/g_loss_gan',     g_adv,     global_step[0])

            if cfg.verbose:
                log.info(
                    f'  [GAN] epoch={epoch + 1:>3d}  step={global_step[0]:>7d}'
                    f'  D_loss={d_loss_last:.4f}'
                    f'  G_loss={g_loss.item():.4f}'
                    f'  (factual={g_fact:.4f}  adv={g_adv:.4f})'
                )

    avg_d = running_d / max(n_batches, 1)
    avg_g = running_g / max(n_batches, 1)
    return avg_d, avg_g


# ══════════════════════════════════════════════════════════════════════════════
# Training — Phase 2: Inference phase (một epoch)
# ══════════════════════════════════════════════════════════════════════════════

def train_inference_epoch(
    generator:     nn.Module,
    inference_net: nn.Module,
    loader:        DataLoader,
    i_optimizer:   torch.optim.Optimizer,
    cfg:           GANITEConfig,
    device:        torch.device,
    writer:        SummaryWriter,
    epoch:         int,
    global_step:   list,
) -> float:
    """
    Chạy một epoch của Inference phase.

    Generator được đặt về eval mode (weights cố định, không update).
    InferenceNet được đặt về train mode và update theo I_loss.

    Gradient clipping (max_norm=5.0) được áp dụng để đảm bảo stability.

    Tham số
    -------
    generator     : GANITEGenerator (eval mode, không update).
    inference_net : GANITEInferenceNet (train mode).
    loader        : DataLoader tập train.
    i_optimizer   : Adam optimizer cho InferenceNet.
    cfg           : GANITEConfig.
    device        : torch.device.
    writer        : SummaryWriter.
    epoch         : chỉ số epoch (0-indexed, tính trong Inference phase).
    global_step   : list[int] — biến mutable đếm tổng số batch Inference.

    Trả về
    ------
    avg_i_loss : float — I loss trung bình epoch.
    """
    generator.eval()       # G cố định, không update trong phase này
    inference_net.train()

    running_i = 0.0
    n_batches = 0

    for x_b, t_b, y_b in loader:
        if x_b.size(0) < cfg.batch_size:
            continue

        x_b = x_b.to(device)
        t_b = t_b.to(device).unsqueeze(1)
        y_b = y_b.to(device).unsqueeze(1)

        i_loss, i_l1, i_l2 = compute_inference_loss(
            generator, inference_net, x_b, t_b, y_b,
        )

        i_optimizer.zero_grad()
        i_loss.backward()
        nn.utils.clip_grad_norm_(inference_net.parameters(), max_norm=5.0)
        i_optimizer.step()

        running_i += i_loss.item()
        n_batches += 1
        global_step[0] += 1

        if global_step[0] % cfg.log_step == 0:
            writer.add_scalar('inference/i_loss',  i_loss.item(), global_step[0])
            writer.add_scalar('inference/i_loss1', i_l1,          global_step[0])
            writer.add_scalar('inference/i_loss2', i_l2,          global_step[0])

            if cfg.verbose:
                log.info(
                    f'  [INF] epoch={epoch + 1:>3d}  step={global_step[0]:>7d}'
                    f'  I_loss={i_loss.item():.4f}'
                    f'  (Y1={i_l1:.4f}  Y0={i_l2:.4f})'
                )

    avg_i = running_i / max(n_batches, 1)
    return avg_i


# ══════════════════════════════════════════════════════════════════════════════
# Inference — tính uplift scores τ̂(X) = sigmoid(I_Ŷ₁) − sigmoid(I_Ŷ₀)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_uplift(
    inference_net: nn.Module,
    X:             np.ndarray,
    device:        torch.device,
    batch_size:    int = 4096,
) -> np.ndarray:
    """
    Tính uplift scores trên một tập dữ liệu bằng InferenceNet.

    Chạy inference theo từng mini-batch để tránh OOM với tập test 1.4M mẫu.
    InferenceNet được đưa về eval mode (tắt Dropout nếu có).

    Uplift score:
        τ̂(x) = sigmoid(I_Y1_logit(x)) − sigmoid(I_Y0_logit(x))

    Tham số
    -------
    inference_net : GANITEInferenceNet (eval mode).
    X             : np.ndarray float32 (n, d) — features (đã chuẩn hóa nếu cfg.normalize).
    device        : torch.device.
    batch_size    : int — batch size cho inference (mặc định 4096).

    Trả về
    ------
    uplift_scores : np.ndarray float32 (n,) — τ̂(X) = P(Y=1|T=1,X) − P(Y=1|T=0,X).
    """
    inference_net.eval()
    X_tensor = torch.from_numpy(X).float()
    n        = len(X_tensor)
    all_uplift = []

    for start in range(0, n, batch_size):
        x_b       = X_tensor[start: start + batch_size].to(device)
        inf_logits = inference_net(x_b)                   # (batch, 2)
        # Uplift = P(Y=1|T=1) - P(Y=1|T=0)
        y1_prob   = torch.sigmoid(inf_logits[:, 1])       # (batch,)
        y0_prob   = torch.sigmoid(inf_logits[:, 0])       # (batch,)
        uplift_b  = (y1_prob - y0_prob).cpu().numpy()
        all_uplift.append(uplift_b)

    return np.concatenate(all_uplift, axis=0)             # (n,)


# ══════════════════════════════════════════════════════════════════════════════
# Đánh giá uplift metrics — AUUC, QINI, LIFT@30%
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_uplift(
    inference_net: nn.Module,
    X:             np.ndarray,
    y:             np.ndarray,
    t:             np.ndarray,
    device:        torch.device,
    cfg:           GANITEConfig,
    split_name:    str            = 'test',
    writer:        SummaryWriter  = None,
    epoch:         int            = 0,
) -> tuple:
    """
    Đánh giá InferenceNet với 3 uplift metrics: AUUC, QINI, LIFT@30%.

    Sử dụng các hàm từ metrics/uplift_metrics.py (cùng metric với CDUM và DESCN).

    Tham số
    -------
    inference_net : GANITEInferenceNet.
    X             : np.ndarray (n, d) — features (đã chuẩn hóa nếu cfg.normalize).
    y             : np.ndarray (n,)   — binary outcome.
    t             : np.ndarray (n,)   — binary treatment.
    device        : torch.device.
    cfg           : GANITEConfig.
    split_name    : str — tên split để log ('val', 'test').
    writer        : SummaryWriter hoặc None.
    epoch         : int — epoch hiện tại (dùng cho trục x TensorBoard).

    Trả về
    ------
    auuc   : float — normalized AUUC.
    qini   : float — normalized QINI coefficient.
    lift30 : float — LIFT@30%.
    """
    uplift_scores = predict_uplift(inference_net, X, device, cfg.batch_size)

    log.info(
        f'  [{split_name.upper()}] Uplift stats:'
        f'  mean={uplift_scores.mean():.5f}'
        f'  std={uplift_scores.std():.5f}'
        f'  min={uplift_scores.min():.5f}'
        f'  max={uplift_scores.max():.5f}'
    )

    auuc   = uplift_auc_score1(y_true=y, uplift=uplift_scores, treatment=t)
    qini   = qini_auc_score1  (y_true=y, uplift=uplift_scores, treatment=t)
    lift30 = uplift_at_k1     (y_true=y, uplift=uplift_scores, treatment=t,
                                strategy='overall', k=0.3)

    log.info(
        f'  [{split_name.upper()}]'
        f'  AUUC={auuc:.6f}  QINI={qini:.6f}  LIFT@30%={lift30:.6f}'
    )

    if writer is not None:
        writer.add_scalar(f'{split_name}/auuc',   auuc,   epoch)
        writer.add_scalar(f'{split_name}/qini',   qini,   epoch)
        writer.add_scalar(f'{split_name}/lift30', lift30, epoch)

    return auuc, qini, lift30


# ══════════════════════════════════════════════════════════════════════════════
# Hàm hỗ trợ: lấy model bên trong DataParallel (để lưu checkpoint sạch)
# ══════════════════════════════════════════════════════════════════════════════

def unwrap(model: nn.Module) -> nn.Module:
    """
    Trả về model bên trong DataParallel (nếu có), ngược lại trả về model gốc.

    Checkpoint lưu theo state_dict của model bên trong để tránh prefix 'module.'
    và tương thích khi load lại trên cấu hình GPU khác.

    Tham số
    -------
    model : nn.Module hoặc nn.DataParallel.

    Trả về
    ------
    nn.Module — model bên trong.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg    = GANITEConfig()
    seed_everything(cfg.seed)
    device = get_device(cfg)

    # ── [1/7] Load dataset ────────────────────────────────────────────────────
    log.info('\n[1/7] Loading dataset ...')
    df_all = load_dataset(DATA_PATH, FEATURE_COLS, LABEL_COL, TREAT_COL)

    # ── [2/7] Split 8:1:1 ─────────────────────────────────────────────────────
    log.info('\n[2/7] Splitting data 8:1:1 ...')
    (x_train_df, y_train_s, t_train_s,
     x_val_df,   y_val_s,   t_val_s,
     x_test_df,  y_test_s,  t_test_s,
     _denominators) = split_dataset(df_all, FEATURE_COLS, LABEL_COL, TREAT_COL)
    # _denominators: dùng cho CDUM bucket embedding, không cần với GANITE

    # ── [3/7] Chuẩn bị và chuẩn hóa dữ liệu ──────────────────────────────────
    log.info('\n[3/7] Preparing data for GANITE ...')
    X_train, y_train, t_train = prepare_criteo_split(x_train_df, y_train_s, t_train_s)
    X_val,   y_val,   t_val   = prepare_criteo_split(x_val_df,   y_val_s,   t_val_s)
    X_test,  y_test,  t_test  = prepare_criteo_split(x_test_df,  y_test_s,  t_test_s)

    if cfg.normalize:
        log.info('  Applying StandardScaler normalization (fit on train) ...')
        X_train, X_val, X_test, _ = normalize_features(X_train, X_val, X_test)
        log.info('  Normalization done.')

    log.info(f'  Train : n={len(X_train):,}  '
             f'treatment_rate={t_train.mean():.4f}  '
             f'positive_rate={y_train.mean():.4f}')
    log.info(f'  Val   : n={len(X_val):,}')
    log.info(f'  Test  : n={len(X_test):,}')

    # ── Tính pos_weight để bù imbalance ────────────────────────────────────────
    # Criteo: T=1 treatment rate  ~85% → t_pos_weight ≈ 0.18 (down-weight T=1)
    #   → Dùng cho D loss và G adversarial loss để D/G adversarial game không bị
    #     trivial (nếu không, D luôn predict T=1 và G không học được gì).
    #
    # Criteo: Y=1 positive rate ~4.7% → y_pos_weight ≈ 20 (chỉ tính để theo dõi)
    #   → KHÔNG dùng trong G factual loss hay I loss vì sẽ làm G học G_Y0 ≈ G_Y1
    #     cho mẫu Y=1 (pseudo-label counterfactual bị corrupted) → variance của
    #     uplift = I_Y1 - I_Y0 bị thu hẹp → QINI rất thấp.
    #     Xem phân tích trong docstring của compute_generator_loss và compute_inference_loss.
    y_rate = float(y_train.mean())
    t_rate = float(t_train.mean())
    y_pos_weight = torch.tensor(
        [(1.0 - y_rate) / max(y_rate, 1e-9)], dtype=torch.float32
    ).to(device)
    t_pos_weight = torch.tensor(
        [(1.0 - t_rate) / max(t_rate, 1e-9)], dtype=torch.float32
    ).to(device)
    log.info(f'  y_pos_weight = {y_pos_weight.item():.4f}  '
             f'(chỉ tính, KHÔNG dùng trong G/I loss, positive_rate={y_rate:.4f})')
    log.info(f'  t_pos_weight = {t_pos_weight.item():.4f}  '
             f'(dùng cho D/G adversarial, treatment_rate={t_rate:.4f})')

    # DataLoader tập train (dùng chung cho cả GAN phase và Inference phase)
    train_dataset = CriteoDataset(X_train, y_train, t_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,    # tăng tốc H2D transfer
        drop_last   = False,
    )

    # ── [4/7] Khởi tạo các mạng GANITE ───────────────────────────────────────
    log.info('\n[4/7] Building GANITE networks ...')
    input_dim = X_train.shape[1]   # 12 features Criteo

    generator     = GANITEGenerator(input_dim,  cfg.h_dim).to(device)
    discriminator = GANITEDiscriminator(input_dim, cfg.h_dim).to(device)
    inference_net = GANITEInferenceNet(input_dim, cfg.h_dim).to(device)

    # Multi-GPU: mỗi mạng được bọc riêng trong DataParallel
    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        generator     = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        inference_net = nn.DataParallel(inference_net)
        log.info(f'  DataParallel: {n_gpu} GPU')

    n_g = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
    n_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    n_i = sum(p.numel() for p in inference_net.parameters() if p.requires_grad)
    log.info(f'  Generator params     : {n_g:,}')
    log.info(f'  Discriminator params : {n_d:,}')
    log.info(f'  InferenceNet params  : {n_i:,}')
    log.info(f'  Tổng params          : {n_g + n_d + n_i:,}')

    # Optimizers — mỗi mạng có optimizer riêng, tránh cập nhật nhầm
    g_optimizer = torch.optim.Adam(generator.parameters(),     lr=cfg.lr_gan)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_gan)
    i_optimizer = torch.optim.Adam(inference_net.parameters(), lr=cfg.lr_inf)

    # TensorBoard writer
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    tb_path = os.path.join(TENSORBOARD_DIR, MODEL_NAME)
    writer  = SummaryWriter(tb_path)
    log.info(f'  TensorBoard logs: {tb_path}')
    log.info(f'  Xem bằng: tensorboard --logdir={TENSORBOARD_DIR}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── [5/7] Phase 1 — GAN phase ─────────────────────────────────────────────
    log.info(f'\n[5/7] Phase 1: GAN training for {cfg.epochs_gan} epochs ...')
    log.info(f'  n_disc_steps={cfg.n_disc_steps}  alpha={cfg.alpha}  lr_gan={cfg.lr_gan}')

    gan_step = [0]
    for epoch in range(cfg.epochs_gan):
        log.info(f'\n── GAN Epoch {epoch + 1}/{cfg.epochs_gan} ──────────────────')

        avg_d, avg_g = train_gan_epoch(
            generator, discriminator, train_loader,
            g_optimizer, d_optimizer,
            cfg, device, writer, epoch, gan_step,
            t_pos_weight,
        )

        writer.add_scalar('gan/epoch_d_loss', avg_d, epoch + 1)
        writer.add_scalar('gan/epoch_g_loss', avg_g, epoch + 1)
        log.info(f'  GAN epoch avg — D_loss={avg_d:.5f}  G_loss={avg_g:.5f}')

    # Lưu checkpoint Generator sau GAN phase (dùng lại cho Inference phase)
    ckpt_gen = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_generator.pth')
    torch.save(unwrap(generator).state_dict(), ckpt_gen)
    log.info(f'\n  Generator checkpoint : {ckpt_gen}')

    # ── [6/7] Phase 2 — Inference phase ──────────────────────────────────────
    log.info(f'\n[6/7] Phase 2: Inference training for {cfg.epochs_inf} epochs ...')
    log.info(f'  lr_inf={cfg.lr_inf}')

    inf_step = [0]
    best_qini  = -float('inf')
    best_epoch = 0

    for epoch in range(cfg.epochs_inf):
        log.info(f'\n── Inference Epoch {epoch + 1}/{cfg.epochs_inf} ──────────────')

        avg_i = train_inference_epoch(
            generator, inference_net, train_loader,
            i_optimizer, cfg, device, writer, epoch, inf_step,
        )

        writer.add_scalar('inference/epoch_i_loss', avg_i, epoch + 1)
        log.info(f'  Inference epoch avg — I_loss={avg_i:.5f}')

        # Đánh giá trên validation set sau mỗi epoch Inference
        log.info('  Evaluating on validation set ...')
        _, qini_val, _ = evaluate_uplift(
            inference_net, X_val, y_val, t_val,
            device, cfg, 'val', writer, epoch + 1,
        )

        # Lưu checkpoint tốt nhất theo QINI trên validation
        if qini_val > best_qini:
            best_qini  = qini_val
            best_epoch = epoch + 1
            ckpt_best  = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
            torch.save(unwrap(inference_net).state_dict(), ckpt_best)
            log.info(f'  Checkpoint lưu : {ckpt_best}  (QINI_val={best_qini:.6f})')

    # Lưu checkpoint epoch cuối
    ckpt_final = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_final.pth')
    torch.save(unwrap(inference_net).state_dict(), ckpt_final)
    log.info(f'\nCheckpoint cuối   : {ckpt_final}')
    log.info(f'Best val QINI      : {best_qini:.6f}  (epoch {best_epoch})')

    # ── [7/7] Đánh giá trên test set ─────────────────────────────────────────
    log.info('\n[7/7] Final evaluation on test set ...')

    # Load best checkpoint trước khi đánh giá test
    best_ckpt = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
    if os.path.exists(best_ckpt):
        log.info(f'  Loading best checkpoint: {best_ckpt}')
        state = torch.load(best_ckpt, map_location=device)
        unwrap(inference_net).load_state_dict(state)
    else:
        log.warning(f'  Không tìm thấy {best_ckpt}, dùng model epoch cuối.')

    auuc_test, qini_test, lift_test = evaluate_uplift(
        inference_net, X_test, y_test, t_test,
        device, cfg, 'test', writer, cfg.epochs_inf + 1,
    )

    writer.close()

    # ── In kết quả cuối ────────────────────────────────────────────────────────
    print('\n╔══════════════════════════════════════╗')
    print('║   GANITE — Criteo Uplift Results     ║')
    print('╠══════════════════════════════════════╣')
    print(f'║  AUUC     : {auuc_test:>16.6f}      ║')
    print(f'║  QINI     : {qini_test:>16.6f}      ║')
    print(f'║  LIFT@30% : {lift_test:>16.6f}      ║')
    print('╠══════════════════════════════════════╣')
    print(f'║  Best val QINI  : {best_qini:.6f}       ║')
    print(f'║  Best val epoch : {best_epoch:>3d}                ║')
    print('╚══════════════════════════════════════╝')

    # ── Lưu kết quả metrics vào JSON ──────────────────────────────────────────
    results_record = {
        'model'         : 'GANITE',
        'dataset'       : 'Criteo Uplift v2.1',
        'auuc'          : float(auuc_test),
        'qini'          : float(qini_test),
        'lift30'        : float(lift_test),
        'best_val_qini' : float(best_qini),
        'best_val_epoch': int(best_epoch),
        'config': {
            'h_dim'        : cfg.h_dim,
            'epochs_gan'   : cfg.epochs_gan,
            'epochs_inf'   : cfg.epochs_inf,
            'n_disc_steps' : cfg.n_disc_steps,
            'alpha'        : cfg.alpha,
            'lr_gan'       : cfg.lr_gan,
            'lr_inf'       : cfg.lr_inf,
            'batch_size'   : cfg.batch_size,
            'normalize'    : cfg.normalize,
            'seed'         : cfg.seed,
            'y_pos_weight' : float(y_pos_weight.item()),
            't_pos_weight' : float(t_pos_weight.item()),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_results.json')
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(results_record, fp, indent=4, ensure_ascii=False)
    log.info(f'Kết quả đã lưu: {result_path}')


if __name__ == '__main__':
    main()
