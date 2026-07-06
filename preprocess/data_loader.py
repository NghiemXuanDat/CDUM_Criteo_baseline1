"""
data_loader.py — Tiền xử lý và nạp dữ liệu Criteo Uplift v2.1
===============================================================
Chức năng:
  - load_dataset()       : đọc file .csv.gz, in thống kê cơ bản.
  - split_dataset()      : chia train/val/test theo tỷ lệ 8:1:1.
  - compute_denominators(): tính max value từng feature trên train set
                            (dùng để bucket normalize trong model).
  - make_data_generator(): generator vô hạn sinh batch (inputs, targets)
                            cho Keras model.fit().

Dataset : Criteo Uplift v2.1 (~14M rows, 12 features, binary treatment)
Columns : f0…f11 (numeric), treatment (0/1), visit (0/1)

Môi trường yêu cầu:
    pandas 2.3.3, numpy 1.26.4, scikit-learn 1.8.0
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════════════════
# Load dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_path, feature_cols, label_col, treat_col):
    """
    Đọc file CSV (hoặc .csv.gz), in thống kê cơ bản và trả về DataFrame.

    Tham số
    -------
    data_path   : str  — đường dẫn file CSV / CSV.gz.
    feature_cols: list — danh sách tên cột feature, VD ['f0', …, 'f11'].
    label_col   : str  — tên cột nhãn (VD 'visit').
    treat_col   : str  — tên cột treatment (VD 'treatment').

    Trả về
    ------
    df_all : pd.DataFrame — toàn bộ dataset.
    """
    print(f'\n[load_dataset] Đang đọc: {data_path}')
    df_all = pd.read_csv(data_path)
    print(f'               Tổng số dòng : {len(df_all):,}')
    print(f'               Các cột      : {list(df_all.columns)}')
    print(f'               Treatment ratio (1/0): {df_all[treat_col].mean():.4f}')
    print(f'               Positive rate ({label_col}): {df_all[label_col].mean():.4f}')
    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# Train / Val / Test split
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(df_all, feature_cols, label_col, treat_col,
                  test_size=0.2, val_ratio=0.5, random_state=42):
    """
    Chia dataset theo tỷ lệ 8:1:1 (train : val : test).

    Tham số
    -------
    df_all       : pd.DataFrame — toàn bộ dataset từ load_dataset().
    feature_cols : list — danh sách tên cột feature.
    label_col    : str  — tên cột nhãn.
    treat_col    : str  — tên cột treatment.
    test_size    : float — tỷ lệ dữ liệu tách ra khỏi train (mặc định 0.2 → 20%).
    val_ratio    : float — tỷ lệ chia val từ phần test_size (mặc định 0.5 → 50%).
    random_state : int  — seed để đảm bảo reproducibility.

    Trả về
    ------
    (x_train, y_train, t_train,
     x_val,   y_val,   t_val,
     x_test,  y_test,  t_test,
     denominators)

    denominators : list — giá trị max từng feature trên train set.
    """
    print(f'\n[split_dataset] Chia dữ liệu 8:1:1 ...')
    df_train, df_tmp  = train_test_split(df_all, test_size=test_size,
                                         random_state=random_state)
    df_val,   df_test = train_test_split(df_tmp, test_size=val_ratio,
                                         random_state=random_state)
    print(f'                train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}')

    denominators = compute_denominators(df_train, feature_cols)

    x_train = df_train[feature_cols]; y_train = df_train[label_col]; t_train = df_train[treat_col]
    x_val   = df_val[feature_cols];   y_val   = df_val[label_col];   t_val   = df_val[treat_col]
    x_test  = df_test[feature_cols];  y_test  = df_test[label_col];  t_test  = df_test[treat_col]

    return (x_train, y_train, t_train,
            x_val,   y_val,   t_val,
            x_test,  y_test,  t_test,
            denominators)


# ══════════════════════════════════════════════════════════════════════════════
# Feature denominators
# ══════════════════════════════════════════════════════════════════════════════

def compute_denominators(df_train, feature_cols):
    """
    Tính giá trị max từng feature trên train set để dùng cho bucket normalization.

    Chỉ dùng train set để tránh data leakage từ val/test.

    Tham số
    -------
    df_train     : pd.DataFrame — tập train.
    feature_cols : list — danh sách tên cột feature.

    Trả về
    ------
    denominators : list of float — max value theo thứ tự feature_cols.
    """
    denominators = [df_train[f].max() for f in feature_cols]
    return denominators


# ══════════════════════════════════════════════════════════════════════════════
# Keras data generator
# ══════════════════════════════════════════════════════════════════════════════

def make_data_generator(data, batch_size, features):
    """
    Generator vô hạn (infinite) sinh ra các batch (inputs, targets) cho Keras.

    Tham số
    -------
    data      : [X_df, y_series, treatment_series]
    batch_size: kích thước batch
    features  : danh sách tên cột feature

    Yields
    ------
    (input_list, [treat_label, ctrl_label])
        input_list  = [feat_0_array, …, feat_11_array, treatment_mask_array]
        treat_label = y * treatment_mask          (chỉ có giá trị khi treatment=1)
        ctrl_label  = y * (1 − treatment_mask)    (chỉ có giá trị khi treatment=0)

    Dùng với model.fit(generator, steps_per_epoch=...) để Keras tự dừng
    sau steps_per_epoch bước trong mỗi epoch.
    """
    X_df, y_s, t_s = data
    n = X_df.shape[0]
    while True:
        for offset in range(0, n, batch_size):
            X_batch = X_df.iloc[offset: offset + batch_size]
            mask    = t_s.iloc[offset: offset + batch_size].values
            label   = y_s.iloc[offset: offset + batch_size].values

            feat_arrays = [np.array(X_batch[col].tolist()) for col in features]
            feat_arrays.append(mask)   # treatment indicator là input cuối cùng
            yield feat_arrays, [label * mask, label * (1 - mask)]
