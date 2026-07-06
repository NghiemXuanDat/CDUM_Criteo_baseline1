"""
run_cevae_criteo.py — Pipeline huấn luyện và đánh giá CEVAE trên Criteo Uplift v2.1
=====================================================================================
Tham chiếu:
    Louizos et al., "Causal Effect Inference with Deep Latent-Variable Models",
    NeurIPS 2017. https://arxiv.org/abs/1705.08821

Orchestrate toàn bộ pipeline:
  1. Load dataset từ file .csv.gz
  2. Split 8:1:1 (train / val / test) thông qua preprocess.data_loader
  3. Chuyển đổi Criteo pandas DataFrame → NumPy float32 + StandardScaler
  4. Khởi tạo và huấn luyện CEVAE từ đầu (KHÔNG dùng pretrained weights)
  5. Inference trên val/test set → uplift scores = E_z[y1 − y0]
  6. Đánh giá AUUC, QINI, LIFT@30% thông qua metrics.uplift_metrics

Cấu hình:
    Chỉnh sửa class CEVAEConfig ở phần "Cấu hình toàn cục" bên dưới.

Phụ thuộc nội bộ:
    from preprocess.data_loader  import load_dataset, split_dataset
    from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
    from baseline.cevae          import CEVAE, CriteoDataset

Multi-GPU (tùy chọn):
    - 1 GPU  : mặc định, device='cuda:0' (RTX 3060)
    - 2 GPU  : đặt cfg.multi_gpu = True → DataParallel trên tất cả GPU (RTX 4090×2)
    Với DataParallel: model.module.elbo() được gọi thay vì model.elbo().
    Checkpoint lưu state_dict của model bên trong (không có prefix 'module.').

Feature normalization:
    cfg.normalize = True (mặc định) → áp dụng StandardScaler fit trên train set.
    Scaler được fit CHỈ trên X_train để tránh data leakage. CEVAE không có
    BatchNorm bên trong, nên cần chuẩn hóa ngoài để đảm bảo training ổn định.

Uplift prediction:
    Tại test/val time, actual (t, y) được truyền qua encoder để infer z tốt nhất.
    Uplift = E_z[P(Y=1|t=1,z) − P(Y=1|t=0,z)], ước lượng bằng Monte Carlo.
    cfg.n_samples       : số mẫu MC cho validation per-epoch (mặc định 10 → nhanh).
    cfg.n_samples_final : số mẫu MC cho final test evaluation (mặc định 50 → chính xác).

Gradient clipping:
    VAE training đôi khi bị unstable vì ELBO có log terms có thể rất âm.
    cfg.max_grad_norm = 5.0 giới hạn L2 norm của gradient trước mỗi optimizer step.

Môi trường yêu cầu:
    conda activate umlc_env
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.3,
    scikit-learn 1.8.0, tensorboard

Cách chạy (từ thư mục gốc /home/datnghiemxuan/Documents/Criteo_CDUM/):
    conda activate umlc_env
    python -m experiment.run_cevae_criteo
    # hoặc:
    python experiment/run_cevae_criteo.py
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Nội bộ ───────────────────────────────────────────────────────────────────
from preprocess.data_loader  import load_dataset, split_dataset
from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1
from baseline.cevae          import CEVAE, CriteoDataset


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
OUTPUT_DIR      = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/CEVAE/checkpoints'
TENSORBOARD_DIR = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/CEVAE/runs'
MODEL_NAME      = 'cevae_criteo'

FEATURE_COLS = [f'f{i}' for i in range(12)]   # f0 … f11
LABEL_COL    = 'visit'
TREAT_COL    = 'treatment'


# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CEVAEConfig:
    """
    Siêu tham số huấn luyện CEVAE trên Criteo Uplift v2.1.

    Kiến trúc mạng
    --------------
    z_dim        : số chiều không gian tiềm ẩn z (mặc định 20, theo paper).
    hidden_dim   : số neuron mỗi lớp ẩn (mặc định 128; paper dùng 200).
                   Criteo có 12 features (nhỏ) nên 128 đủ hiệu quả.
    n_hidden     : số lớp ẩn cho sub-networks (mặc định 3, theo paper).
    dropout_rate : tỷ lệ dropout (mặc định 0.0 — KL đủ là regularizer trong VAE).

    Huấn luyện
    ----------
    lr            : learning rate ban đầu cho Adam (mặc định 1e-3, theo paper).
    weight_decay  : L2 regularization trong Adam (mặc định 1e-4).
    epochs        : tổng số epoch huấn luyện (mặc định 10).
    batch_size    : kích thước mini-batch (mặc định 4096).
    num_workers   : số worker DataLoader (mặc định 4; đặt 0 nếu debug).
    max_grad_norm : ngưỡng clip gradient L2-norm (mặc định 5.0).
                    VAE training đôi khi bất ổn → clipping giúp ổn định.

    Uplift prediction (Monte Carlo)
    --------------------------------
    n_samples       : số mẫu MC cho mỗi lần gọi predict_uplift() trong quá trình
                      training/validation (mặc định 10 → nhanh).
    n_samples_final : số mẫu MC cho đánh giá cuối trên test set
                      (mặc định 50 → chính xác hơn).

    Feature normalization
    ---------------------
    normalize : True → áp dụng StandardScaler (fit trên train, transform val/test).
                Cần thiết vì CEVAE không có BatchNorm bên trong.

    Khác
    ----
    log_step  : log TensorBoard sau mỗi N global step (mặc định 200).
    device    : 'auto' (tự phát hiện GPU/CPU), 'cuda:0', 'cuda:1', 'cpu'.
    multi_gpu : True → DataParallel trên tất cả GPU có sẵn (mặc định False).
    seed      : random seed (mặc định 42).
    verbose   : 1 → in log chi tiết; 0 → chỉ in kết quả cuối.
    """
    # Kiến trúc mạng
    z_dim:        int   = 20
    hidden_dim:   int   = 128
    n_hidden:     int   = 3
    dropout_rate: float = 0.0

    # Huấn luyện
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    epochs:        int   = 10
    batch_size:    int   = 4096
    num_workers:   int   = 4
    max_grad_norm: float = 5.0

    # Uplift prediction
    n_samples:       int = 10
    n_samples_final: int = 50

    # Feature normalization
    normalize: bool = True

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

def get_device(cfg: CEVAEConfig) -> torch.device:
    """
    Xác định thiết bị tính toán (CPU / GPU) từ cấu hình.

    Khi cfg.device = 'auto': tự động chọn cuda:0 nếu có GPU, ngược lại CPU.

    Tham số
    -------
    cfg : CEVAEConfig.

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
# Chuẩn bị dữ liệu — Criteo pandas → NumPy float32
# ══════════════════════════════════════════════════════════════════════════════

def prepare_criteo_split(x_df, y_series, t_series):
    """
    Chuyển đổi một split Criteo từ pandas sang numpy float32 cho CEVAE.

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


def normalize_features(X_train: np.ndarray, X_val: np.ndarray,
                        X_test: np.ndarray):
    """
    Chuẩn hóa features bằng StandardScaler (fit trên train, transform val/test).

    CEVAE không có BatchNorm bên trong encoder/decoder, nên cần chuẩn hóa
    ngoài để đảm bảo các feature có cùng thang đo và giúp VAE training ổn định.
    Scaler được fit CHỈ trên X_train để tránh data leakage.

    Tham số
    -------
    X_train : np.ndarray (n_train, 12) — features tập train.
    X_val   : np.ndarray (n_val, 12)   — features tập val.
    X_test  : np.ndarray (n_test, 12)  — features tập test.

    Trả về
    ------
    X_train_norm, X_val_norm, X_test_norm : np.ndarray float32 đã chuẩn hóa.
    scaler : StandardScaler đã fit (lưu lại để dùng sau nếu cần).
    """
    scaler       = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train).astype(np.float32)
    X_val_norm   = scaler.transform(X_val).astype(np.float32)
    X_test_norm  = scaler.transform(X_test).astype(np.float32)
    return X_train_norm, X_val_norm, X_test_norm, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Huấn luyện — một epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    cfg: CEVAEConfig, device: torch.device,
                    writer: SummaryWriter, epoch: int,
                    global_step: list) -> float:
    """
    Chạy huấn luyện CEVAE qua một epoch đầy đủ.

    Mỗi bước tối thiểu hóa −ELBO trên một mini-batch bằng Adam.
    Gradient clipping (max_grad_norm) được áp dụng để ổn định VAE training.

    Với DataParallel: elbo() được gọi qua model.module.elbo() (unwrapped).
    Lý do: CEVAE.elbo() trả về scalar; DataParallel yêu cầu output là tuple/tensor
    có shape nhất định. Workaround: gọi trực tiếp inner model.

    Tham số
    -------
    model       : CEVAE (hoặc nn.DataParallel wrapper).
    loader      : DataLoader tập train.
    optimizer   : Adam optimizer.
    cfg         : CEVAEConfig.
    device      : torch.device.
    writer      : SummaryWriter (TensorBoard).
    epoch       : chỉ số epoch hiện tại (0-indexed).
    global_step : list[int] — biến mutable đếm tổng số batch đã huấn luyện.

    Trả về
    ------
    avg_loss : float — −ELBO trung bình của epoch.
    """
    # Lấy inner model để tính ELBO (tránh vấn đề DataParallel với scalar output)
    inner_model = model.module if isinstance(model, nn.DataParallel) else model
    inner_model.train()

    running_loss = 0.0
    n_batches    = 0

    for x_b, t_b, y_b in loader:
        # Bỏ batch quá nhỏ (< 2 mẫu): tránh NaN trong Normal distribution stats
        if x_b.size(0) < 2:
            continue

        x_b = x_b.to(device)
        t_b = t_b.to(device)   # (n,) float
        y_b = y_b.to(device)   # (n,) float

        optimizer.zero_grad()

        # Tính −ELBO = −E_q[log p] + KL
        neg_elbo = inner_model.elbo(x_b, t_b, y_b)

        neg_elbo.backward()

        # Gradient clipping — giới hạn L2-norm gradient để ổn định VAE training
        nn.utils.clip_grad_norm_(inner_model.parameters(), cfg.max_grad_norm)

        optimizer.step()

        running_loss += neg_elbo.item()
        n_batches    += 1
        global_step[0] += 1

        # Log TensorBoard theo log_step
        if global_step[0] % cfg.log_step == 0:
            writer.add_scalar('train/neg_elbo', neg_elbo.item(), global_step[0])

            if cfg.verbose:
                log.info(
                    f'  epoch={epoch + 1:>3d}  step={global_step[0]:>7d}'
                    f'  −ELBO={neg_elbo.item():.4f}'
                )

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════════════
# Inference — tính uplift scores τ̂(x) = E_z[y1 − y0]
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_uplift(model: nn.Module, X: np.ndarray, t: np.ndarray,
                   y: np.ndarray, device: torch.device,
                   batch_size: int, n_samples: int) -> np.ndarray:
    """
    Tính uplift scores trên một tập dữ liệu.

    CEVAE cần actual (t, y) để infer z qua encoder, sau đó dùng decoder
    để tính E[P(Y=1|z,t=1) − P(Y=1|z,t=0)] với Monte Carlo averaging.

    Chạy inference theo từng mini-batch để tránh OOM với tập test 1.4M mẫu.
    Model được đặt về eval mode (dropout tắt nếu có).

    Tham số
    -------
    model      : CEVAE (eval mode, hoặc DataParallel).
    X          : np.ndarray float32 (n, 12) — features (đã chuẩn hóa).
    t          : np.ndarray float32 (n,)    — treatment (0 hoặc 1).
    y          : np.ndarray float32 (n,)    — outcome (0 hoặc 1).
    device     : torch.device.
    batch_size : int — batch size cho inference (mặc định = cfg.batch_size).
    n_samples  : int — số mẫu Monte Carlo.

    Trả về
    ------
    uplift_scores : np.ndarray float32 (n,) — τ̂(x) = E_z[y1 − y0].
    """
    # Lấy inner model để gọi predict_uplift (tránh vấn đề DataParallel)
    inner_model = model.module if isinstance(model, nn.DataParallel) else model
    inner_model.eval()

    X_t = torch.from_numpy(X).float()
    t_t = torch.from_numpy(t).float()
    y_t = torch.from_numpy(y).float()
    n   = len(X_t)
    all_uplift = []

    for start in range(0, n, batch_size):
        x_b = X_t[start: start + batch_size].to(device)
        t_b = t_t[start: start + batch_size].to(device)
        y_b = y_t[start: start + batch_size].to(device)

        # predict_uplift trả về (y0_mean, y1_mean) — Tensor (batch_n,) mỗi cái
        y0, y1 = inner_model.predict_uplift(x_b, t_b, y_b, n_samples)
        uplift_b = (y1 - y0).cpu().numpy()
        all_uplift.append(uplift_b)

    return np.concatenate(all_uplift, axis=0)    # (n,)


# ══════════════════════════════════════════════════════════════════════════════
# Đánh giá uplift metrics — AUUC, QINI, LIFT@30%
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_uplift(model: nn.Module, X: np.ndarray, t: np.ndarray,
                    y: np.ndarray, device: torch.device,
                    cfg: CEVAEConfig, split_name: str = 'test',
                    writer: SummaryWriter = None, epoch: int = 0,
                    n_samples: int = None) -> tuple:
    """
    Đánh giá CEVAE với 3 uplift metrics: AUUC, QINI, LIFT@30%.

    Sử dụng các hàm từ metrics/uplift_metrics.py (cùng metric với CDUM và DESCN).

    Tham số
    -------
    model      : CEVAE.
    X          : np.ndarray (n, 12) — features (đã chuẩn hóa nếu normalize=True).
    t          : np.ndarray (n,)    — binary treatment.
    y          : np.ndarray (n,)    — binary outcome.
    device     : torch.device.
    cfg        : CEVAEConfig.
    split_name : str — tên split để log ('val', 'test').
    writer     : SummaryWriter hoặc None.
    epoch      : int — epoch hiện tại (dùng cho trục x TensorBoard).
    n_samples  : int hoặc None — số mẫu MC; None → dùng cfg.n_samples.

    Trả về
    ------
    auuc   : float — normalized AUUC.
    qini   : float — normalized QINI coefficient.
    lift30 : float — LIFT@30%.
    """
    if n_samples is None:
        n_samples = cfg.n_samples

    uplift_scores = predict_uplift(
        model, X, t, y, device, cfg.batch_size, n_samples
    )

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
# Hàm hỗ trợ — lấy model bên trong DataParallel
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
    cfg = CEVAEConfig()
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
    # _denominators: dùng cho CDUM bucket embedding, không cần với CEVAE

    # ── [3/6] Chuẩn bị và chuẩn hóa dữ liệu ──────────────────────────────────
    log.info('\n[3/6] Preparing data for CEVAE ...')
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

    # DataLoader tập train
    train_dataset = CriteoDataset(X_train, y_train, t_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,    # tăng tốc H2D transfer cho GPU
        drop_last   = False,
    )

    # ── [4/6] Khởi tạo mô hình ───────────────────────────────────────────────
    log.info('\n[4/6] Building CEVAE model ...')
    n_features = X_train.shape[1]    # 12 features Criteo

    model = CEVAE(
        n_features   = n_features,
        z_dim        = cfg.z_dim,
        hidden_dim   = cfg.hidden_dim,
        n_hidden     = cfg.n_hidden,
        dropout_rate = cfg.dropout_rate,
    ).to(device)

    # Multi-GPU: DataParallel tự động chia batch cho tất cả GPU có sẵn.
    # Lưu ý: train_one_epoch và predict_uplift đều dùng unwrap() để gọi
    # inner model — tránh vấn đề DataParallel với scalar ELBO output.
    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log.info(f'  DataParallel: {torch.cuda.device_count()} GPU')

    n_params = sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)
    log.info(f'  Tổng tham số trainable: {n_params:,}')
    log.info(f'  z_dim={cfg.z_dim}  hidden_dim={cfg.hidden_dim}  n_hidden={cfg.n_hidden}')

    # Adam optimizer — theo bản gốc CEVAE (lr=1e-3, weight_decay nhỏ)
    optimizer = torch.optim.Adam(
        unwrap(model).parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )

    # StepLR: giảm LR 10% sau mỗi epoch (nhẹ, cho phép ELBO hội tụ chậm)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.95
    )

    # TensorBoard writer
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    tb_path = os.path.join(TENSORBOARD_DIR, MODEL_NAME)
    writer  = SummaryWriter(tb_path)
    log.info(f'  TensorBoard logs: {tb_path}')
    log.info(f'  Xem bằng: tensorboard --logdir={TENSORBOARD_DIR}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── [5/6] Huấn luyện ─────────────────────────────────────────────────────
    log.info(f'\n[5/6] Training CEVAE for {cfg.epochs} epochs ...')
    log.info(
        f'  lr={cfg.lr}  weight_decay={cfg.weight_decay}'
        f'  batch_size={cfg.batch_size}  max_grad_norm={cfg.max_grad_norm}'
    )

    global_step = [0]              # list để truyền tham chiếu mutable vào hàm
    best_qini   = -float('inf')
    best_epoch  = 0

    for epoch in range(cfg.epochs):
        log.info(f'\n── Epoch {epoch + 1}/{cfg.epochs} ──────────────────────────')

        # Một epoch huấn luyện: tối thiểu hóa −ELBO
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, writer, epoch, global_step
        )
        writer.add_scalar('train/epoch_neg_elbo', avg_loss, epoch + 1)
        log.info(f'  Train avg −ELBO : {avg_loss:.5f}')

        # Cập nhật LR scheduler
        lr_scheduler.step()
        new_lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar('train/lr', new_lr, epoch + 1)
        if cfg.verbose:
            log.info(f'  Learning rate   : {new_lr:.6f}')

        # Đánh giá trên validation set (n_samples=cfg.n_samples để nhanh)
        log.info('  Evaluating on validation set ...')
        _, qini_val, _ = evaluate_uplift(
            model, X_val, t_val, y_val,
            device, cfg, 'val', writer, epoch + 1,
            n_samples=cfg.n_samples,
        )

        # Lưu checkpoint tốt nhất theo QINI trên validation
        if qini_val > best_qini:
            best_qini  = qini_val
            best_epoch = epoch + 1
            ckpt_best  = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
            torch.save(unwrap(model).state_dict(), ckpt_best)
            log.info(f'  Checkpoint lưu : {ckpt_best}  (QINI_val={best_qini:.6f})')

    # Lưu checkpoint epoch cuối
    ckpt_final = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_final.pth')
    torch.save(unwrap(model).state_dict(), ckpt_final)
    log.info(f'\nCheckpoint cuối   : {ckpt_final}')
    log.info(f'Best val QINI      : {best_qini:.6f}  (epoch {best_epoch})')

    # ── [6/6] Đánh giá trên test set ─────────────────────────────────────────
    log.info('\n[6/6] Final evaluation on test set ...')

    # Load best checkpoint trước khi đánh giá test
    best_ckpt = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
    if os.path.exists(best_ckpt):
        log.info(f'  Loading best checkpoint: {best_ckpt}')
        state = torch.load(best_ckpt, map_location=device)
        unwrap(model).load_state_dict(state)
    else:
        log.warning(f'  Không tìm thấy {best_ckpt}, dùng model epoch cuối.')

    # Đánh giá cuối với n_samples_final (nhiều mẫu MC hơn → chính xác hơn)
    log.info(f'  Sử dụng n_samples={cfg.n_samples_final} cho đánh giá cuối.')
    auuc_test, qini_test, lift_test = evaluate_uplift(
        model, X_test, t_test, y_test,
        device, cfg, 'test', writer, cfg.epochs + 1,
        n_samples=cfg.n_samples_final,
    )

    writer.close()

    # ── In kết quả cuối ────────────────────────────────────────────────────────
    print('\n╔══════════════════════════════════════╗')
    print('║   CEVAE — Criteo Uplift Results      ║')
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
        'model'          : 'CEVAE',
        'dataset'        : 'Criteo Uplift v2.1',
        'auuc'           : float(auuc_test),
        'qini'           : float(qini_test),
        'lift30'         : float(lift_test),
        'best_val_qini'  : float(best_qini),
        'best_val_epoch' : int(best_epoch),
        'config': {
            'z_dim'          : cfg.z_dim,
            'hidden_dim'     : cfg.hidden_dim,
            'n_hidden'       : cfg.n_hidden,
            'dropout_rate'   : cfg.dropout_rate,
            'epochs'         : cfg.epochs,
            'batch_size'     : cfg.batch_size,
            'lr'             : cfg.lr,
            'weight_decay'   : cfg.weight_decay,
            'max_grad_norm'  : cfg.max_grad_norm,
            'n_samples'      : cfg.n_samples,
            'n_samples_final': cfg.n_samples_final,
            'normalize'      : cfg.normalize,
            'seed'           : cfg.seed,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_results.json')
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(results_record, fp, indent=4, ensure_ascii=False)
    log.info(f'Kết quả đã lưu: {result_path}')


if __name__ == '__main__':
    main()
