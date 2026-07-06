"""
run_euen_criteo.py — Pipeline huấn luyện và đánh giá EUEN trên Criteo Uplift v2.1
===================================================================================
Orchestrate toàn bộ pipeline:
  1. Load dataset từ file .csv.gz
  2. Split 8:1:1 (train / val / test) thông qua preprocess.data_loader
  3. Chuyển đổi Criteo pandas DataFrame → NumPy float32 → CriteoDataset
  4. Khởi tạo và huấn luyện EUENModel từ đầu (KHÔNG dùng pretrained weights)
  5. Inference trên test set → uplift scores = u_tau
  6. Đánh giá AUUC, QINI, LIFT@30% thông qua metrics.uplift_metrics

Kiến trúc EUEN (từ bản gốc XDL/TF1.x — Ke et al., ICDM 2021):
    ControlNet : input → 64 → 32 → 16 → 1  (E[Y(0)|X])
    UpliftNet  : input → 64 → 32 → 16 → 1  (τ(X) = E[Y(1)−Y(0)|X])
    Loss       : MSE(uc, y | T=0) + MSE(ut, y | T=1)
                 với ut = detach(c_logit) + u_tau

Cấu hình:
    Chỉnh sửa class EUENConfig ở phần "Cấu hình toàn cục" bên dưới.

Phụ thuộc nội bộ:
    from preprocess.data_loader import load_dataset, split_dataset
    from metrics.uplift_metrics import uplift_auc_score1, qini_auc_score1, uplift_at_k1
    from baseline.euen          import EUENModel, CriteoDataset

Multi-GPU (tùy chọn):
    - 1 GPU  : mặc định, device='cuda:0' (RTX 3060)
    - 2 GPU  : đặt cfg.multi_gpu = True → DataParallel trên tất cả GPU (RTX 4090×2)
    Lưu ý: DataParallel scatter batch theo chiều 0. Checkpoint lưu theo state
    của model bên trong (không có prefix 'module.') để tương thích khi load lại.

Môi trường yêu cầu:
    conda activate umlc_env
    Python 3.x, PyTorch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0

Cách chạy (từ thư mục gốc /home/datnghiemxuan/Documents/Criteo_CDUM/):
    conda activate umlc_env
    python -m experiment.run_euen_criteo
    # hoặc:
    python experiment/run_euen_criteo.py
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
from preprocess.data_loader import load_dataset, split_dataset
from metrics.uplift_metrics import uplift_auc_score1, qini_auc_score1, uplift_at_k1
from baseline.euen          import EUENModel, CriteoDataset


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
OUTPUT_DIR      = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/EUEN/checkpoints'
TENSORBOARD_DIR = '/home/datnghiemxuan/Documents/Criteo_CDUM/results/EUEN/runs'
MODEL_NAME      = 'euen_criteo'

FEATURE_COLS = [f'f{i}' for i in range(12)]   # f0 … f11
LABEL_COL    = 'visit'
TREAT_COL    = 'treatment'


# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EUENConfig:
    """
    Siêu tham số huấn luyện EUEN trên Criteo Uplift v2.1.

    Kiến trúc mạng
    --------------
    hc_dim : số neuron ControlNet lớp đầu tiên (mặc định 64).
    hu_dim : số neuron UpliftNet lớp đầu tiên (mặc định 64).
    use_bn : có dùng BatchNorm1d ở đầu vào model không (mặc định True).
             True → ổn định hóa training với features Criteo có scale khác nhau.

    Huấn luyện
    ----------
    lr          : learning rate (mặc định 1e-3, theo bản gốc).
    l2          : hệ số L2 weight decay trong Adam (mặc định 1e-2 = 0.01).
                  Khớp với l2_reg=0.01 trong bản gốc EUEN XDL.
    epochs      : số epoch huấn luyện (mặc định 4, theo bản gốc EUEN).
    batch_size  : kích thước mini-batch (mặc định 4096).
                  Bản gốc dùng 512 nhưng tăng lên 4096 cho Criteo scale 14M mẫu.
    num_workers : số worker DataLoader (mặc định 4; đặt 0 nếu debug).

    Khác
    ----
    log_step  : log TensorBoard sau mỗi N global step (mặc định 200).
    device    : 'auto' → tự phát hiện GPU; hoặc 'cuda:0', 'cuda:1', 'cpu'.
    multi_gpu : True → DataParallel trên tất cả GPU có sẵn (RTX 4090×2).
    seed      : random seed (mặc định 42).
    verbose   : 1 → in log chi tiết; 0 → chỉ in kết quả cuối.
    """
    # Kiến trúc mạng
    hc_dim : int  = 64
    hu_dim : int  = 64
    use_bn : bool = True

    # Huấn luyện
    lr         : float = 1e-3
    l2         : float = 1e-2
    epochs     : int   = 4
    batch_size : int   = 4096
    num_workers: int   = 4

    # Khác
    log_step  : int  = 200
    device    : str  = 'auto'
    multi_gpu : bool = False
    seed      : int  = 42
    verbose   : int  = 1


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
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# Device setup
# ══════════════════════════════════════════════════════════════════════════════

def get_device(cfg: EUENConfig) -> torch.device:
    """
    Xác định thiết bị tính toán (CPU / GPU) từ cấu hình.

    Khi cfg.device = 'auto': tự động chọn cuda:0 nếu có GPU, ngược lại CPU.

    Tham số
    -------
    cfg : EUENConfig.

    Trả về
    ------
    device : torch.device
    """
    if cfg.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    Chuyển đổi một split Criteo từ pandas sang numpy float32 cho EUEN.

    EUEN chỉ cần 3 thành phần: X (features), y (label), t (treatment).
    Không cần trường 'exposure' (chỉ dùng trong EEUEN).

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


# ══════════════════════════════════════════════════════════════════════════════
# Loss function — EUEN lift MSE
# ══════════════════════════════════════════════════════════════════════════════

def lift_mse_loss(c_logit: torch.Tensor, u_tau: torch.Tensor,
                  y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Tính EUEN lift MSE loss trên một mini-batch.

    Công thức (khớp với lift_mse_loss trong utils.py gốc,
    use_huber=False, use_group_reduce=False):

        c_logit_fix = detach(c_logit)           ← stop gradient
        uc = c_logit                             ← prediction nhóm control
        ut = c_logit_fix + u_tau                ← prediction nhóm treatment
        u  = (1−T)·uc + T·ut                   ← dự đoán chung (per sample)
        L  = mean((u − Y)²)

    Lý do stop_gradient:
        Nếu không detach c_logit, gradient từ treatment arm sẽ lan ngược
        qua ControlNet, khiến ControlNet học hỗn hợp tín hiệu từ cả hai nhóm.
        Với stop_gradient, ControlNet chỉ học E[Y(0)|X] thuần túy từ nhóm control.

    Tham số
    -------
    c_logit : Tensor (n, 1) — ControlNet raw output.
    u_tau   : Tensor (n, 1) — UpliftNet raw output (uplift score).
    y       : Tensor (n, 1) float32 — binary outcome label.
    t       : Tensor (n, 1) float32 — binary treatment indicator (0/1).

    Trả về
    ------
    loss : Tensor scalar — MSE loss để backprop.
    """
    # Stop gradient: ngăn gradient từ treatment arm lan vào ControlNet
    c_logit_fix = c_logit.detach()

    # uc: prediction cho nhóm control (có gradient về ControlNet)
    uc = c_logit

    # ut: prediction cho nhóm treatment = control baseline (frozen) + uplift
    ut = c_logit_fix + u_tau

    # Tổng hợp prediction theo treatment indicator
    u = (1.0 - t) * uc + t * ut   # (n, 1)

    # MSE loss
    return torch.mean((u - y) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Huấn luyện — một epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    cfg: EUENConfig,
                    device: torch.device,
                    writer: SummaryWriter,
                    epoch: int,
                    global_step: list) -> float:
    """
    Chạy huấn luyện qua một epoch đầy đủ.

    Cơ chế bỏ batch cuối nhỏ hơn batch_size giữ BatchNorm ổn định —
    BN với batch quá nhỏ sẽ có ước lượng mean/var không đáng tin cậy.

    Tham số
    -------
    model       : EUENModel (có thể bọc trong nn.DataParallel).
    loader      : DataLoader cho tập train.
    optimizer   : torch optimizer (Adam).
    cfg         : EUENConfig.
    device      : torch.device.
    writer      : SummaryWriter (TensorBoard).
    epoch       : chỉ số epoch hiện tại (0-indexed).
    global_step : list[int] — biến mutable đếm tổng số batch đã huấn luyện.

    Trả về
    ------
    avg_loss : float — loss trung bình của epoch.
    """
    model.train()
    running_loss = 0.0
    n_batches    = 0

    for x_b, t_b, y_b in loader:
        # Bỏ batch cuối nếu nhỏ hơn batch_size (BatchNorm unstable)
        if x_b.size(0) < cfg.batch_size:
            continue

        # Chuyển sang device và reshape nhãn thành (n, 1)
        x_b = x_b.to(device)
        t_b = t_b.to(device).unsqueeze(1)   # (n, 1)
        y_b = y_b.to(device).unsqueeze(1)   # (n, 1)

        optimizer.zero_grad()

        # Forward pass: nhận (c_logit, u_tau)
        c_logit, u_tau = model(x_b)

        # Tính lift MSE loss
        loss = lift_mse_loss(c_logit, u_tau, y_b, t_b)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches    += 1
        global_step[0] += 1

        # Log TensorBoard theo log_step
        if global_step[0] % cfg.log_step == 0:
            writer.add_scalar('train/loss', loss.item(), global_step[0])

            if cfg.verbose:
                log.info(
                    f'  epoch={epoch+1:>3d}  step={global_step[0]:>7d}'
                    f'  loss={loss.item():.6f}'
                )

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════════════
# Inference — tính uplift scores τ̂(X) = u_tau
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_uplift(model: nn.Module,
                   X: np.ndarray,
                   device: torch.device,
                   batch_size: int = 4096) -> np.ndarray:
    """
    Tính uplift scores trên một tập dữ liệu.

    Chạy inference theo từng mini-batch để tránh OOM với tập test ~1.4M mẫu.
    Mô hình được đưa về eval mode (tắt Dropout và BatchNorm train mode).

    Uplift score của EUEN = u_tau (output của UpliftNet) — đây là ước lượng
    trực tiếp của τ(X) = E[Y(1)−Y(0)|X], không cần tính hiệu giữa hai nhánh.

    Tham số
    -------
    model      : EUENModel (eval mode).
    X          : np.ndarray float32 (n, 12) — features.
    device     : torch.device.
    batch_size : int — batch size cho inference (mặc định 4096).

    Trả về
    ------
    uplift_scores : np.ndarray float32 (n,) — τ̂(X) = u_tau.
    """
    model.eval()
    X_tensor  = torch.from_numpy(X).float()
    n         = len(X_tensor)
    all_uplift = []

    for start in range(0, n, batch_size):
        x_b           = X_tensor[start: start + batch_size].to(device)
        _, u_tau      = model(x_b)
        uplift_b      = u_tau.squeeze(1).cpu().numpy()
        all_uplift.append(uplift_b)

    return np.concatenate(all_uplift, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Đánh giá uplift metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_uplift(model: nn.Module,
                    X: np.ndarray,
                    y: np.ndarray,
                    t: np.ndarray,
                    device: torch.device,
                    cfg: EUENConfig,
                    split_name: str = 'test',
                    writer: SummaryWriter = None,
                    epoch: int = 0):
    """
    Đánh giá mô hình với 3 uplift metrics: AUUC, QINI, LIFT@30%.

    Sử dụng các hàm từ metrics/uplift_metrics.py (thuật toán theo CDUM paper).

    Tham số
    -------
    model      : EUENModel.
    X          : np.ndarray (n, 12) — features.
    y          : np.ndarray (n,)    — binary outcome.
    t          : np.ndarray (n,)    — binary treatment.
    device     : torch.device.
    cfg        : EUENConfig.
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
    cfg = EUENConfig()
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
    # _denominators: giá trị max feature (chỉ dùng cho CDUM embedding), không cần với EUEN

    # ── [3/6] Chuẩn bị dữ liệu ───────────────────────────────────────────────
    log.info('\n[3/6] Preparing data for EUEN ...')
    X_train, y_train, t_train = prepare_criteo_split(x_train_df, y_train_s, t_train_s)
    X_val,   y_val,   t_val   = prepare_criteo_split(x_val_df,   y_val_s,   t_val_s)
    X_test,  y_test,  t_test  = prepare_criteo_split(x_test_df,  y_test_s,  t_test_s)

    log.info(f'  Train : n={len(X_train):,}  '
             f'treatment_rate={t_train.mean():.4f}  '
             f'positive_rate={y_train.mean():.4f}')
    log.info(f'  Val   : n={len(X_val):,}')
    log.info(f'  Test  : n={len(X_test):,}')

    # DataLoader cho tập train (shuffle=True để ngẫu nhiên hóa thứ tự batch)
    train_dataset = CriteoDataset(X_train, y_train, t_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,    # tăng tốc host→device transfer cho GPU
        drop_last   = False,   # giữ batch cuối (bị skip trong train_one_epoch nếu nhỏ)
    )

    # ── [4/6] Khởi tạo mô hình ───────────────────────────────────────────────
    log.info('\n[4/6] Building EUEN model ...')
    input_dim = X_train.shape[1]   # 12 features Criteo

    model = EUENModel(
        input_dim = input_dim,
        hc_dim    = cfg.hc_dim,
        hu_dim    = cfg.hu_dim,
        use_bn    = cfg.use_bn,
    ).to(device)

    # Multi-GPU: DataParallel tự động chia batch cho tất cả GPU có sẵn
    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log.info(f'  DataParallel: {torch.cuda.device_count()} GPU')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  Tổng tham số trainable: {n_params:,}')

    # Adam optimizer với weight_decay = L2 regularization (l2_reg=0.01 theo bản gốc)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.l2
    )

    # TensorBoard writer
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    tb_path = os.path.join(TENSORBOARD_DIR, MODEL_NAME)
    writer  = SummaryWriter(tb_path)
    log.info(f'  TensorBoard logs: {tb_path}')
    log.info(f'  Xem bằng: tensorboard --logdir={TENSORBOARD_DIR}')

    # ── [5/6] Huấn luyện ─────────────────────────────────────────────────────
    log.info(f'\n[5/6] Training EUEN for {cfg.epochs} epochs ...')
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
        log.info(f'  Train avg_loss : {avg_loss:.6f}')

        # Đánh giá trên validation set sau mỗi epoch
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

    # Lưu checkpoint của epoch cuối cùng
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
        state       = torch.load(best_ckpt, map_location=device)
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
    print('║   EUEN — Criteo Uplift Results   ║')
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
        'model'         : 'EUEN',
        'dataset'       : 'Criteo Uplift v2.1',
        'auuc'          : float(auuc_test),
        'qini'          : float(qini_test),
        'lift30'        : float(lift_test),
        'best_val_qini' : float(best_qini),
        'best_val_epoch': int(best_epoch),
        'config': {
            'hc_dim'    : cfg.hc_dim,
            'hu_dim'    : cfg.hu_dim,
            'use_bn'    : cfg.use_bn,
            'lr'        : cfg.lr,
            'l2'        : cfg.l2,
            'epochs'    : cfg.epochs,
            'batch_size': cfg.batch_size,
            'seed'      : cfg.seed,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_results.json')
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(results_record, fp, indent=4, ensure_ascii=False)
    log.info(f'Kết quả đã lưu: {result_path}')


if __name__ == '__main__':
    main()
