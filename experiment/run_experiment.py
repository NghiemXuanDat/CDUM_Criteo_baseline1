"""
run_experiment.py — Pipeline huấn luyện và đánh giá CPM (CDUM framework)
=========================================================================
Orchestrate toàn bộ pipeline:
  1. Load dataset từ file .csv.gz
  2. Split 8:1:1 (train / val / test)
  3. Build model (+ train nếu TRAIN_MODE=True)
  4. Load pre-trained weights từ file .h5
  5. Inference trên test set → uplift scores
  6. Đánh giá AUUC, QINI, LIFT@30%

Cấu hình:
  - TRAIN_MODE   : bool — True → train từ đầu; False → chỉ load weights và evaluate.
  - DATA_PATH    : str  — đường dẫn dataset .csv.gz.
  - WEIGHTS_PATH : str  — đường dẫn file weights .h5.
  - EMBEDDING_DIM: int  — chiều embedding (mặc định 32).
  - BATCH_SIZE   : int  — batch size (mặc định 4096).
  - NUM_EPOCHS   : int  — số epoch khi TRAIN_MODE=True (mặc định 10).

Phụ thuộc nội bộ:
    from preprocess.data_loader import load_dataset, split_dataset
    from CDUM.model             import UpliftModel
    from metrics.uplift_metrics import uplift_auc_score1, qini_auc_score1, uplift_at_k1

Môi trường yêu cầu:
    Python 3.x, TF 2.15.0, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0

Cách chạy (từ thư mục gốc /home/datnghiemxuan/Documents/Criteo_CDUM/):
    conda activate umlc_env
    python -m experiment.run_experiment
    # hoặc: python experiment/run_experiment.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import random

# Thêm thư mục gốc vào sys.path để import các module nội bộ
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf

# ── Nội bộ ───────────────────────────────────────────────────────────────────
from preprocess.data_loader  import load_dataset, split_dataset
from CDUM.model              import UpliftModel
from metrics.uplift_metrics  import uplift_auc_score1, qini_auc_score1, uplift_at_k1


# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình toàn cục — chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_MODE   = False   # True → train từ đầu; False → chỉ load weights và evaluate
DATA_PATH    = '/home/datnghiemxuan/Documents/Criteo_CDUM/data/criteo-uplift-v2.1.csv.gz'
WEIGHTS_PATH = '/home/datnghiemxuan/Documents/Criteo_CDUM/CDUM/epoch_criteo.h5'

EMBEDDING_DIM = 32     # số chiều embedding cho mỗi feature và treatment
BATCH_SIZE    = 4096   # batch size cho train và inference
NUM_EPOCHS    = 10     # số epoch khi TRAIN_MODE = True

FEATURE_COLS  = [f'f{i}' for i in range(12)]   # f0 … f11
LABEL_COL     = 'visit'       # nhãn nhị phân (0/1)
TREAT_COL     = 'treatment'   # nhãn treatment (0/1)


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print('\n[1/6] Loading dataset ...')
    df_all = load_dataset(DATA_PATH, FEATURE_COLS, LABEL_COL, TREAT_COL)

    # ── 2. Split 8:1:1 ────────────────────────────────────────────────────────
    print('\n[2/6] Splitting data 8:1:1 ...')
    (x_train, y_train, t_train,
     x_val,   y_val,   t_val,
     x_test,  y_test,  t_test,
     denominators) = split_dataset(df_all, FEATURE_COLS, LABEL_COL, TREAT_COL)

    # ── 3. Build / (optional) train model ─────────────────────────────────────
    print(f'\n[3/6] Building model (TRAIN_MODE={TRAIN_MODE}) ...')
    uplift_obj = UpliftModel(
        embedding_dim   = EMBEDDING_DIM,
        batch_size      = BATCH_SIZE,
        train_data_size = len(x_train),
        valid_data_size = len(x_val),
        train           = [x_train, y_train, t_train],
        test            = [x_val,   y_val,   t_val],
        features        = FEATURE_COLS,
        denominators    = denominators,
        weights_path    = WEIGHTS_PATH,
    )
    uplift_model = uplift_obj.model_train(train_mode=TRAIN_MODE, num_epochs=NUM_EPOCHS)

    # ── 4. Load pre-trained weights ───────────────────────────────────────────
    # by_name=True    : match theo tên layer thay vì theo thứ tự.
    # skip_mismatch=True: các DNN layers (expert_*, tower_*) không được lưu
    #                    đúng cách trong h5 cũ → skip và dùng random init.
    print(f'\n[4/6] Loading weights: {WEIGHTS_PATH}')
    uplift_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    print('      Weights loaded (by_name=True, skip_mismatch=True).')
    print('      NOTE: expert/tower layer weights không có trong h5 → dùng random init.')
    print('      Để có kết quả đầy đủ, cần train lại với TRAIN_MODE=True.')

    # ── 5. Inference trên test set ────────────────────────────────────────────
    # Chạy inference 2 lần:
    #   - treat_inputs  (treatment=1) → lấy treat_output (res_1)
    #   - control_inputs(treatment=0) → lấy base_output  (res_0)
    # uplift = treat_score − base_score
    print('\n[5/6] Running inference on test set ...')

    base_feats     = [x_test[col].to_numpy() for col in FEATURE_COLS]
    treat_inputs   = base_feats + [np.ones(len(x_test),  dtype=np.float32)]
    control_inputs = base_feats + [np.zeros(len(x_test), dtype=np.float32)]

    # predict() trả về [treat_output, base_output]
    # treat_mask=1 → treat_output = giá trị thực, base_output = 0 (masking)
    # treat_mask=0 → treat_output = 0,            base_output = giá trị thực
    res_1, _  = uplift_model.predict(treat_inputs,   batch_size=BATCH_SIZE, verbose=0)
    _,  res_0 = uplift_model.predict(control_inputs, batch_size=BATCH_SIZE, verbose=0)

    uplift_scores = (res_1 - res_0).reshape(-1)
    print(f'      Uplift scores: shape={uplift_scores.shape}  '
          f'mean={uplift_scores.mean():.4f}  std={uplift_scores.std():.4f}')

    # ── 6. Đánh giá ───────────────────────────────────────────────────────────
    print('\n[6/6] Evaluating ...')
    auuc = uplift_auc_score1(y_true=y_test, uplift=uplift_scores, treatment=t_test)
    qini = qini_auc_score1  (y_true=y_test, uplift=uplift_scores, treatment=t_test)
    lift = uplift_at_k1     (y_true=y_test, uplift=uplift_scores, treatment=t_test,
                             strategy='overall', k=0.3)

    print('\n╔══════════════════════════╗')
    print('║  Evaluation Results      ║')
    print('╠══════════════════════════╣')
    print(f'║  AUUC     : {auuc:>10.6f}  ║')
    print(f'║  QINI     : {qini:>10.6f}  ║')
    print(f'║  LIFT@30% : {lift:>10.6f}  ║')
    print('╚══════════════════════════╝')


if __name__ == '__main__':
    main()
