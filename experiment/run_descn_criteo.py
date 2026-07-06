"""
run_descn_criteo.py — Pipeline huấn luyện và đánh giá DESCN trên Criteo Uplift v2.1
=====================================================================================
Orchestrate toàn bộ pipeline:
  1. Load dataset từ file .csv.gz
  2. Split 8:1:1 (train / val / test) thông qua preprocess.data_loader
  3. Chuyển đổi Criteo pandas DataFrame → NumPy float32 → CriteoDataset
  4. Khởi tạo và huấn luyện DESCNModel từ đầu (KHÔNG dùng pretrained weights)
  5. Inference trên test set → uplift scores = p_h1 − p_h0
  6. Đánh giá AUUC, QINI, LIFT@30% thông qua metrics.uplift_metrics

Cấu hình:
    Chỉnh sửa class DESCNConfig ở phần "Cấu hình toàn cục" bên dưới.

Phụ thuộc nội bộ:
    from preprocess.data_loader  import load_dataset, split_dataset
    from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
    from baseline.descn          import DESCNModel, CriteoDataset, gaussian_mmd

Multi-GPU (tùy chọn):
    - 1 GPU  : mặc định, device='cuda:0' (RTX 3060)
    - 2 GPU  : đặt cfg.multi_gpu = True → DataParallel trên tất cả GPU (RTX 4090×2)
    Lưu ý: DataParallel scatter batch theo chiều 0, checkpoint lưu theo state
    của model bên trong (không có prefix 'module.') để tương thích khi load lại.

Môi trường yêu cầu:
    conda activate umlc_env
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0

Cách chạy (từ thư mục gốc /home/datnghiemxuan/Documents/Criteo_CDUM/):
    conda activate umlc_env
    python -m experiment.run_descn_criteo
    # hoặc:
    python experiment/run_descn_criteo.py
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Nội bộ ───────────────────────────────────────────────────────────────────
from preprocess.data_loader  import load_dataset, split_dataset
from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
from baseline.descn          import DESCNModel, CriteoDataset, gaussian_mmd


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
OUTPUT_DIR      = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/DESCN/checkpoints'
TENSORBOARD_DIR = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/DESCN/runs'
MODEL_NAME      = 'descn_criteo'

FEATURE_COLS = [f'f{i}' for i in range(12)]   # f0 … f11
LABEL_COL    = 'visit'
TREAT_COL    = 'treatment'


# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DESCNConfig:
    """
    Siêu tham số huấn luyện DESCN trên Criteo Uplift v2.1.

    Kiến trúc mạng
    --------------
    share_dim     : chiều ẩn của ShareNetwork (mặc định 128).
    base_dim      : chiều output ShareNetwork = input BaseModel (mặc định 64).
    do_rate       : tỷ lệ dropout mỗi lớp (mặc định 0.1).
    use_bn        : có dùng BatchNorm1d ở đầu vào ShareNetwork không.
    normalization : 'divide' → L2-normalize shared_h; '' → không normalize.

    Huấn luyện
    ----------
    lr              : learning rate ban đầu (mặc định 1e-3).
    l2              : hệ số L2 weight decay trong optimizer (mặc định 1e-3).
    decay_rate      : gamma của StepLR scheduler (mặc định 0.95).
    decay_step_size : LR decay sau mỗi N epoch (mặc định 1).
    epochs          : tổng số epoch huấn luyện (mặc định 10).
    batch_size      : kích thước mini-batch (mặc định 4096).
    optim           : 'Adam' hoặc 'SGD' (mặc định 'Adam').
    num_workers     : số worker DataLoader (mặc định 4; đặt 0 nếu debug).

    Trọng số loss
    -------------
    h1_w       : trọng số BCE(p_h1[T=1], y[T=1])            — mặc định 0.5.
    h0_w       : trọng số BCE(p_h0[T=0], y[T=0])            — mặc định 0.1.
    mu1hat_w   : trọng số cross-treatment BCE(σ(μ0+τ)[T=1]) — mặc định 0.0.
    mu0hat_w   : trọng số cross-control  BCE(σ(μ1−τ)[T=0]) — mặc định 0.0.
    prpsy_w    : trọng số propensity BCE_logit               — mặc định 0.0.
    escvr1_w   : trọng số ESTR BCE_w(p_estr, Y·T)           — mặc định 0.0.
    escvr0_w   : trọng số ESCR BCE_w(p_escr, Y·(1−T))       — mặc định 0.0.
    imb_dist_w : trọng số MMD balance loss                   — mặc định 0.1.
    imb_dist   : 'mmd' (Gaussian MMD, không cần geomloss) hoặc
                 'wass' (Wasserstein, cần cài thêm: pip install geomloss).

    Khác
    ----
    reweight_sample : dùng IPW (inverse propensity weighting) cho BCELoss.
    log_step        : log TensorBoard sau mỗi N global step (mặc định 200).
    device          : 'auto' (tự phát hiện GPU/CPU), 'cuda:0', 'cuda:1', 'cpu'.
    multi_gpu       : True → DataParallel trên tất cả GPU (mặc định False).
    seed            : random seed (mặc định 42).
    verbose         : 1 → in log chi tiết; 0 → chỉ in kết quả cuối.
    """
    # Kiến trúc mạng
    share_dim:     int   = 128
    base_dim:      int   = 64
    do_rate:       float = 0.1
    use_bn:        bool  = True
    normalization: str   = "divide"

    # Huấn luyện
    lr:              float = 1e-3
    l2:              float = 1e-3
    decay_rate:      float = 0.95
    decay_step_size: int   = 1
    epochs:          int   = 10
    batch_size:      int   = 4096
    optim:           str   = "Adam"
    num_workers:     int   = 4

    # Trọng số loss
    prpsy_w:   float = 0.0
    escvr1_w:  float = 0.0
    escvr0_w:  float = 0.0
    h1_w:      float = 0.5
    h0_w:      float = 0.1
    mu1hat_w:  float = 0.0
    mu0hat_w:  float = 0.0
    imb_dist_w: float = 0.1
    imb_dist:  str   = "mmd"

    # Khác
    reweight_sample: bool = True
    log_step:        int  = 200
    device:          str  = "auto"
    multi_gpu:       bool = False
    seed:            int  = 42
    verbose:         int  = 1


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Device setup
# ══════════════════════════════════════════════════════════════════════════════

def get_device(cfg: DESCNConfig) -> torch.device:
    """
    Xác định thiết bị tính toán (CPU / GPU) từ cấu hình.

    Khi cfg.device = 'auto': tự động chọn cuda:0 nếu có GPU, ngược lại CPU.

    Tham số
    -------
    cfg : DESCNConfig.

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
                     f'({props.total_memory / 1024**3:.1f} GB VRAM)')
    return device


# ══════════════════════════════════════════════════════════════════════════════
# Chuẩn bị dữ liệu — Criteo pandas → NumPy float32
# ══════════════════════════════════════════════════════════════════════════════

def prepare_criteo_split(x_df, y_series, t_series):
    """
    Chuyển đổi một split Criteo từ pandas sang numpy float32 cho DESCN.

    Ghi chú về trường 'e' (randomized indicator):
        Convention DESCN: e=0 → observational, e=1 → randomized.
        Criteo Uplift v2.1 là A/B test thuần túy (fully randomized RCT),
        vì vậy ta đặt e=0 cho tất cả mẫu — mọi mẫu đều được tính vào loss.
        Điều này không ảnh hưởng đến kết quả thực tế vì các loss sử dụng
        ~e mask (prpsy_loss, estr_loss, escr_loss) đều có weight=0 mặc định.

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
    e : np.ndarray float32 (n,) — toàn bộ là 0
    """
    X = x_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=np.float32)
    t = t_series.to_numpy(dtype=np.float32)
    e = np.zeros(len(X), dtype=np.float32)   # toàn bộ là 0 (observational)
    return X, y, t, e


# ══════════════════════════════════════════════════════════════════════════════
# Tính toán loss — multi-task DESCN
# ══════════════════════════════════════════════════════════════════════════════

def compute_descn_loss(model_outputs, y, t, e, cfg: DESCNConfig):
    """
    Tính tổng loss multi-task của DESCN trên một mini-batch.

    Cấu trúc loss:
        total = h1_loss + h0_loss
              + cross_tr_loss + cross_cr_loss
              + prpsy_loss + estr_loss + escr_loss
              + imb_dist_loss

    Chỉ tính các loss có weight > 0 để tiết kiệm tính toán. Loss nào có
    weight = 0 sẽ bị bỏ qua hoàn toàn (không forward qua BCELoss).

    Tham số
    -------
    model_outputs : tuple (12 phần tử) từ DESCNModel.forward().
    y   : Tensor (n, 1) float32 — binary outcome, trên device.
    t   : Tensor (n, 1) float32 — binary treatment, trên device.
    e   : Tensor (n, 1) float32 — randomized indicator (0=obs), trên device.
    cfg : DESCNConfig.

    Trả về
    ------
    total_loss : Tensor scalar — tổng loss để backprop.
    loss_dict  : dict[str, float] — giá trị từng thành phần để log.
    """
    (p_prpsy_logit, p_estr, p_escr, tau_logit,
     mu1_logit, mu0_logit,
     p_prpsy, p_mu1, p_mu0, p_h1, p_h0,
     shared_h) = model_outputs

    device = t.device
    total_loss = torch.tensor(0.0, device=device)
    loss_dict  = {}

    # ── Sample reweighting (Inverse Propensity Weighting) ─────────────────────
    # Cân bằng đóng góp của nhóm treatment và control vào loss.
    # sample_weight[i] = t[i]/(2p_t) + (1-t[i])/(2(1-p_t))
    p_t = torch.mean(t).item()
    if cfg.reweight_sample and 0.0 < p_t < 1.0:
        w_t = t / (2.0 * p_t)
        w_c = (1.0 - t) / (2.0 * (1.0 - p_t))
        sample_weight = w_t + w_c               # (n, 1)
    else:
        sample_weight = torch.ones_like(t)
        p_t = 0.5

    # Mặt nạ mẫu observational (~e): mask hình (n,1) bool
    obs_mask = ~e.bool()                        # True ở mẫu có e=0

    # ── h1_loss: supervised BCE cho nhóm treatment ────────────────────────────
    if cfg.h1_w > 0:
        t_mask = t.bool()                           # (n,1) bool — T=1
        if t_mask.any():
            h1_loss = cfg.h1_w * nn.functional.binary_cross_entropy(
                p_h1[t_mask], y[t_mask]
            )
            total_loss = total_loss + h1_loss
            loss_dict['h1_loss'] = h1_loss.item()

    # ── h0_loss: supervised BCE cho nhóm control ──────────────────────────────
    if cfg.h0_w > 0:
        c_mask = (~t.bool())                        # (n,1) bool — T=0
        if c_mask.any():
            h0_loss = cfg.h0_w * nn.functional.binary_cross_entropy(
                p_h0[c_mask], y[c_mask]
            )
            total_loss = total_loss + h0_loss
            loss_dict['h0_loss'] = h0_loss.item()

    # ── cross_tr_loss: σ(μ0+τ) ≈ μ1, tín hiệu nhất quán qua nhóm T=1 ────────
    # Ý nghĩa: dự đoán outcome của nhóm T=1 bằng (μ0 + τ) thay vì μ1 trực tiếp.
    # Buộc mô hình học τ phải nhất quán với hiệu μ1 - μ0.
    if cfg.mu1hat_w > 0:
        t_mask = t.bool()
        if t_mask.any():
            mu1_hat = torch.sigmoid(mu0_logit + tau_logit)
            cross_tr_loss = cfg.mu1hat_w * nn.functional.binary_cross_entropy(
                mu1_hat[t_mask], y[t_mask]
            )
            total_loss = total_loss + cross_tr_loss
            loss_dict['cross_tr_loss'] = cross_tr_loss.item()

    # ── cross_cr_loss: σ(μ1−τ) ≈ μ0, tín hiệu nhất quán qua nhóm T=0 ────────
    if cfg.mu0hat_w > 0:
        c_mask = (~t.bool())
        if c_mask.any():
            mu0_hat = torch.sigmoid(mu1_logit - tau_logit)
            cross_cr_loss = cfg.mu0hat_w * nn.functional.binary_cross_entropy(
                mu0_hat[c_mask], y[c_mask]
            )
            total_loss = total_loss + cross_cr_loss
            loss_dict['cross_cr_loss'] = cross_cr_loss.item()

    # ── prpsy_loss: BCE với logit cho propensity (chỉ mẫu observational) ──────
    if cfg.prpsy_w > 0 and obs_mask.any():
        pos_w = torch.tensor(1.0 / (2.0 * p_t), device=device)
        prpsy_loss = cfg.prpsy_w * nn.functional.binary_cross_entropy_with_logits(
            p_prpsy_logit[obs_mask], t[obs_mask], pos_weight=pos_w
        )
        total_loss = total_loss + prpsy_loss
        loss_dict['prpsy_loss'] = prpsy_loss.item()

    # ── estr_loss: entire-space treatment response (mẫu observational) ────────
    if cfg.escvr1_w > 0 and obs_mask.any():
        sw_obs = sample_weight[obs_mask]            # (m,) — weights mẫu obs
        estr_loss = cfg.escvr1_w * nn.functional.binary_cross_entropy(
            p_estr[obs_mask], (y * t)[obs_mask], weight=sw_obs
        )
        total_loss = total_loss + estr_loss
        loss_dict['estr_loss'] = estr_loss.item()

    # ── escr_loss: entire-space control response (mẫu observational) ──────────
    if cfg.escvr0_w > 0 and obs_mask.any():
        sw_obs = sample_weight[obs_mask]
        escr_loss = cfg.escvr0_w * nn.functional.binary_cross_entropy(
            p_escr[obs_mask], (y * (1.0 - t))[obs_mask], weight=sw_obs
        )
        total_loss = total_loss + escr_loss
        loss_dict['escr_loss'] = escr_loss.item()

    # ── imb_dist_loss: covariate balance qua MMD / Wasserstein ───────────────
    # Phạt sự khác biệt phân phối giữa shared_h của T=1 và T=0.
    # Giúp mô hình học biểu diễn trung lập (không bị confounded bởi treatment).
    if cfg.imb_dist_w > 0:
        if cfg.imb_dist == "mmd":
            imb = gaussian_mmd(shared_h, t)
        elif cfg.imb_dist == "wass":
            # Wasserstein cần geomloss: pip install geomloss
            # Nếu chưa cài, chuyển sang mmd hoặc tắt imb_dist_w
            try:
                from geomloss import SamplesLoss
                t_flat = t.reshape(-1)
                Xt = shared_h[t_flat == 1]
                Xc = shared_h[t_flat == 0]
                loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
                imb = loss_fn(Xt, Xc)
            except ImportError:
                log.warning("geomloss chưa được cài. Fallback về Gaussian MMD.")
                imb = gaussian_mmd(shared_h, t)
        else:
            imb = torch.tensor(0.0, device=device)

        imb_loss = cfg.imb_dist_w * imb
        total_loss = total_loss + imb_loss
        loss_dict['imb_dist_loss'] = imb_loss.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


# ══════════════════════════════════════════════════════════════════════════════
# Huấn luyện — một epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, cfg: DESCNConfig,
                    device, writer: SummaryWriter, epoch: int,
                    global_step: list) -> float:
    """
    Chạy huấn luyện qua một epoch đầy đủ.

    Cơ chế bỏ batch cuối nhỏ hơn batch_size giữ nguyên từ DESCN gốc —
    BatchNorm hoạt động không ổn định với batch rất nhỏ.

    Tham số
    -------
    model       : DESCNModel (có thể bọc trong nn.DataParallel).
    loader      : DataLoader.
    optimizer   : torch optimizer.
    cfg         : DESCNConfig.
    device      : torch.device.
    writer      : SummaryWriter (TensorBoard).
    epoch       : chỉ số epoch hiện tại (0-indexed).
    global_step : list[int] — biến mutable đếm tổng số batch đã huấn luyện.

    Trả về
    ------
    avg_loss : float — loss trung bình của epoch (trung bình theo n_batches).
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_idx, (x_b, t_b, y_b, e_b) in enumerate(loader):
        # Bỏ batch cuối nếu nhỏ hơn batch_size (BatchNorm unstable với n nhỏ)
        if x_b.size(0) < cfg.batch_size:
            continue

        # Chuyển sang device và reshape nhãn thành (n, 1)
        x_b = x_b.to(device)
        t_b = t_b.to(device).unsqueeze(1)   # (n, 1)
        y_b = y_b.to(device).unsqueeze(1)   # (n, 1)
        e_b = e_b.to(device).unsqueeze(1)   # (n, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_b)

        # Tính tổng loss multi-task
        total_loss, loss_dict = compute_descn_loss(outputs, y_b, t_b, e_b, cfg)

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        n_batches += 1
        global_step[0] += 1

        # Log TensorBoard theo log_step
        if global_step[0] % cfg.log_step == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v, global_step[0])

            if cfg.verbose:
                log.info(
                    f'  epoch={epoch+1:>3d}  step={global_step[0]:>7d}'
                    f'  total={loss_dict["total_loss"]:.4f}'
                    f'  h1={loss_dict.get("h1_loss", 0.0):.4f}'
                    f'  h0={loss_dict.get("h0_loss", 0.0):.4f}'
                    f'  mmd={loss_dict.get("imb_dist_loss", 0.0):.4f}'
                )

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════════════
# Inference — tính uplift scores τ̂(X) = p_h1 − p_h0
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_uplift(model, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    """
    Tính uplift scores trên một tập dữ liệu.

    Chạy inference theo từng mini-batch để tránh OOM với tập test 1.4M mẫu.
    Mô hình được đưa về eval mode (tắt Dropout và BatchNorm train mode).

    Tham số
    -------
    model      : DESCNModel (eval mode).
    X          : np.ndarray float32 (n, 12) — features.
    device     : torch.device.
    batch_size : int — batch size cho inference (mặc định 4096).

    Trả về
    ------
    uplift_scores : np.ndarray float32 (n,) — τ̂(X) = p_h1 − p_h0.
    """
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    n = len(X_tensor)
    all_uplift = []

    for start in range(0, n, batch_size):
        x_b = X_tensor[start: start + batch_size].to(device)
        outputs = model(x_b)
        _, _, _, _, _, _, _, _, _, p_h1, p_h0, _ = outputs
        uplift_b = (p_h1 - p_h0).squeeze(1).cpu().numpy()
        all_uplift.append(uplift_b)

    return np.concatenate(all_uplift, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Đánh giá uplift metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_uplift(model, X: np.ndarray, y: np.ndarray, t: np.ndarray,
                    device, cfg: DESCNConfig, split_name: str = 'test',
                    writer: SummaryWriter = None, epoch: int = 0):
    """
    Đánh giá mô hình với 3 uplift metrics: AUUC, QINI, LIFT@30%.

    Sử dụng các hàm từ metrics/uplift_metrics.py (thuật toán theo CDUM paper).

    Tham số
    -------
    model      : DESCNModel.
    X          : np.ndarray (n, 12) — features.
    y          : np.ndarray (n,)    — binary outcome.
    t          : np.ndarray (n,)    — binary treatment.
    device     : torch.device.
    cfg        : DESCNConfig.
    split_name : str — tên split để log ('val', 'test').
    writer     : SummaryWriter hoặc None.
    epoch      : int — epoch hiện tại (dùng cho trục x TensorBoard).

    Trả về
    ------
    auuc : float, qini : float, lift30 : float
    """
    uplift_scores = predict_uplift(model, X, device, cfg.batch_size)

    log.info(f'  [{split_name.upper()}] Uplift stats:'
             f'  mean={uplift_scores.mean():.5f}'
             f'  std={uplift_scores.std():.5f}'
             f'  min={uplift_scores.min():.5f}'
             f'  max={uplift_scores.max():.5f}')

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
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = DESCNConfig()
    seed_everything(cfg.seed)
    device = get_device(cfg)

    # ── [1/6] Load dataset ────────────────────────────────────────────────────
    log.info('\n[1/6] Loading dataset ...')
    df_all = load_dataset(DATA_PATH, FEATURE_COLS, LABEL_COL, TREAT_COL)

    # ── [2/6] Split 8:1:1 ─────────────────────────────────────────────────────
    log.info('\n[2/6] Splitting data 8:1:1 ...')
    (x_train_df, y_train_s, t_train_s,
     x_val_df,   y_val_s,   t_val_s,
     x_test_df,  y_test_s,  t_test_s,
     _denominators) = split_dataset(df_all, FEATURE_COLS, LABEL_COL, TREAT_COL)
    # _denominators: giá trị max feature (dùng cho CDUM embedding), không cần với DESCN

    # ── [3/6] Chuẩn bị dữ liệu ───────────────────────────────────────────────
    log.info('\n[3/6] Preparing data for DESCN ...')
    X_train, y_train, t_train, e_train = prepare_criteo_split(x_train_df, y_train_s, t_train_s)
    X_val,   y_val,   t_val,   e_val   = prepare_criteo_split(x_val_df,   y_val_s,   t_val_s)
    X_test,  y_test,  t_test,  e_test  = prepare_criteo_split(x_test_df,  y_test_s,  t_test_s)

    log.info(f'  Train : n={len(X_train):,}  '
             f'treatment_rate={t_train.mean():.4f}  '
             f'positive_rate={y_train.mean():.4f}')
    log.info(f'  Val   : n={len(X_val):,}')
    log.info(f'  Test  : n={len(X_test):,}')

    # DataLoader cho tập train (shuffle=True để ngẫu nhiên hóa thứ tự batch)
    train_dataset = CriteoDataset(X_train, y_train, t_train, e_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,   # tăng tốc H2D transfer cho GPU
        drop_last   = False,  # giữ batch cuối (sẽ bị skip trong train_one_epoch nếu nhỏ)
    )

    # ── [4/6] Khởi tạo mô hình ───────────────────────────────────────────────
    log.info('\n[4/6] Building DESCN model ...')
    input_dim = X_train.shape[1]   # 12 features Criteo

    model = DESCNModel(
        input_dim     = input_dim,
        share_dim     = cfg.share_dim,
        base_dim      = cfg.base_dim,
        do_rate       = cfg.do_rate,
        use_bn        = cfg.use_bn,
        normalization = cfg.normalization,
    ).to(device)

    # Multi-GPU: DataParallel tự động chia batch cho tất cả GPU có sẵn
    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log.info(f'  DataParallel: {torch.cuda.device_count()} GPU')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  Tổng tham số trainable: {n_params:,}')

    # Optimizer
    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.l2
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.l2
        )

    # StepLR: giảm LR theo cfg.decay_rate sau mỗi cfg.decay_step_size epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.decay_step_size, gamma=cfg.decay_rate
    )

    # TensorBoard writer
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    tb_path = os.path.join(TENSORBOARD_DIR, MODEL_NAME)
    writer = SummaryWriter(tb_path)
    log.info(f'  TensorBoard logs: {tb_path}')
    log.info(f'  Xem bằng: tensorboard --logdir={TENSORBOARD_DIR}')

    # ── [5/6] Huấn luyện ─────────────────────────────────────────────────────
    log.info(f'\n[5/6] Training DESCN for {cfg.epochs} epochs ...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global_step = [0]              # list để truyền tham chiếu mutable vào hàm
    best_qini   = -float('inf')
    best_epoch  = 0
    # Truy cập model bên trong DataParallel (tránh prefix 'module.' khi lưu checkpoint)
    inner_model = model.module if isinstance(model, nn.DataParallel) else model

    for epoch in range(cfg.epochs):
        log.info(f'\n── Epoch {epoch+1}/{cfg.epochs} ──────────────────────────')

        # Một epoch huấn luyện
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, writer, epoch, global_step
        )
        writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        log.info(f'  Train avg_loss : {avg_loss:.5f}')

        # Cập nhật LR scheduler
        lr_scheduler.step()
        new_lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar('train/lr', new_lr, epoch + 1)
        if cfg.verbose:
            log.info(f'  Learning rate  : {new_lr:.6f}')

        # Đánh giá trên validation set
        log.info(f'  Evaluating on validation set ...')
        _, qini_val, _ = evaluate_uplift(
            model, X_val, y_val, t_val, device, cfg, 'val', writer, epoch + 1
        )

        # Lưu checkpoint tốt nhất theo QINI trên validation
        if qini_val > best_qini:
            best_qini  = qini_val
            best_epoch = epoch + 1
            ckpt_best  = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
            torch.save(inner_model.state_dict(), ckpt_best)
            log.info(f'  Checkpoint lưu : {ckpt_best}  (QINI_val={best_qini:.6f})')

    # Lưu checkpoint của epoch cuối
    ckpt_final = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_final.pth')
    torch.save(inner_model.state_dict(), ckpt_final)
    log.info(f'\nCheckpoint cuối   : {ckpt_final}')
    log.info(f'Best val QINI      : {best_qini:.6f}  (epoch {best_epoch})')

    # ── [6/6] Đánh giá trên test set ─────────────────────────────────────────
    log.info('\n[6/6] Final evaluation on test set ...')

    # Load best checkpoint trước khi đánh giá test
    best_ckpt = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
    if os.path.exists(best_ckpt):
        log.info(f'  Loading best checkpoint: {best_ckpt}')
        state = torch.load(best_ckpt, map_location=device)
        inner_model = model.module if isinstance(model, nn.DataParallel) else model
        inner_model.load_state_dict(state)
    else:
        log.warning(f'  Không tìm thấy {best_ckpt}, dùng model epoch cuối.')

    auuc_test, qini_test, lift_test = evaluate_uplift(
        model, X_test, y_test, t_test, device, cfg, 'test', writer, cfg.epochs + 1
    )

    writer.close()

    # ── In kết quả cuối ────────────────────────────────────────────────────────
    print('\n╔══════════════════════════════════╗')
    print('║   DESCN — Criteo Uplift Results  ║')
    print('╠══════════════════════════════════╣')
    print(f'║  AUUC     : {auuc_test:>14.6f}    ║')
    print(f'║  QINI     : {qini_test:>14.6f}    ║')
    print(f'║  LIFT@30% : {lift_test:>14.6f}    ║')
    print('╠══════════════════════════════════╣')
    print(f'║  Best val QINI  : {best_qini:.6f}   ║')
    print(f'║  Best val epoch : {best_epoch:>3d}              ║')
    print('╚══════════════════════════════════╝')

    # ── Lưu kết quả metrics vào JSON ──────────────────────────────────────────
    results_record = {
        'model'         : 'DESCN',
        'dataset'       : 'Criteo Uplift v2.1',
        'auuc'          : float(auuc_test),
        'qini'          : float(qini_test),
        'lift30'        : float(lift_test),
        'best_val_qini' : float(best_qini),
        'best_val_epoch': int(best_epoch),
        'config': {
            'share_dim'  : cfg.share_dim,
            'base_dim'   : cfg.base_dim,
            'epochs'     : cfg.epochs,
            'batch_size' : cfg.batch_size,
            'lr'         : cfg.lr,
            'l2'         : cfg.l2,
            'h1_w'       : cfg.h1_w,
            'h0_w'       : cfg.h0_w,
            'imb_dist_w' : cfg.imb_dist_w,
            'imb_dist'   : cfg.imb_dist,
            'seed'       : cfg.seed,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_results.json')
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(results_record, fp, indent=4, ensure_ascii=False)
    log.info(f'Kết quả đã lưu: {result_path}')


if __name__ == '__main__':
    main()
