"""
uplift_metrics.py — Evaluation metrics cho Uplift Modeling
===========================================================
Hiện thực các metrics chuẩn theo paper CDUM và scikit-uplift convention:
  - qini_curve()           : tính Qini curve (sắp xếp theo uplift giảm dần).
  - uplift_curve()         : tính Uplift curve.
  - perfect_uplift_curve() : Uplift curve lý tưởng (oracle).
  - perfect_qini_curve()   : Qini curve lý tưởng (oracle).
  - uplift_auc_score1()    : normalized AUUC (Area Under Uplift Curve).
  - qini_auc_score1()      : normalized Qini coefficient.
  - uplift_at_k1()         : Uplift@k — hiệu conversion rate tại top-k.

Thuật toán giữ nguyên theo paper CDUM và scikit-uplift convention.

Môi trường yêu cầu:
    numpy 1.26.4, scikit-learn 1.8.0
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length


# ══════════════════════════════════════════════════════════════════════════════
# Qini curve
# ══════════════════════════════════════════════════════════════════════════════

def qini_curve(y_true, uplift, treatment):
    """
    Tính Qini curve.

    Sắp xếp samples theo uplift prediction giảm dần, tính lũy kế:
        curve(n) = cumTreated(n) − cumCtrl(n) × (n_treated / n_ctrl)

    Tham số
    -------
    y_true    : array-like, shape (n,) — nhãn kết quả (0/1).
    uplift    : array-like, shape (n,) — uplift prediction score.
    treatment : array-like, shape (n,) — chỉ số treatment (0 hoặc 1).

    Trả về
    ------
    (n_all, curve) — bắt đầu tại (0, 0):
        n_all : array — số mẫu được target.
        curve : array — giá trị Qini tương ứng.
    """
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = map(np.array, [y_true, uplift, treatment])

    order     = np.argsort(uplift, kind='mergesort')[::-1]
    y_true    = y_true[order];    treatment = treatment[order];  uplift = uplift[order]

    y_ctrl    = y_true.copy();   y_ctrl[treatment == 1]  = 0
    y_trmnt   = y_true.copy();   y_trmnt[treatment == 0] = 0

    # Chỉ giữ các điểm threshold khi uplift thay đổi (tránh điểm trùng)
    thresh    = np.r_[np.where(np.diff(uplift))[0], uplift.size - 1]
    n_trmnt   = stable_cumsum(treatment)[thresh]
    y_t_cum   = stable_cumsum(y_trmnt)[thresh]
    n_all     = thresh + 1
    n_ctrl    = n_all - n_trmnt
    y_c_cum   = stable_cumsum(y_ctrl)[thresh]

    curve = y_t_cum - y_c_cum * np.divide(
        n_trmnt, n_ctrl,
        out=np.zeros_like(n_trmnt, dtype=float), where=n_ctrl != 0
    )
    # Đảm bảo curve bắt đầu tại (0, 0)
    if n_all.size == 0 or curve[0] != 0 or n_all[0] != 0:
        n_all = np.r_[0, n_all];   curve = np.r_[0, curve]
    return n_all, curve


# ══════════════════════════════════════════════════════════════════════════════
# Uplift curve
# ══════════════════════════════════════════════════════════════════════════════

def uplift_curve(y_true, uplift, treatment):
    """
    Tính Uplift curve.

        curve(n) = [convRate_treated(n) − convRate_ctrl(n)] × n

    Tham số
    -------
    y_true    : array-like, shape (n,) — nhãn kết quả (0/1).
    uplift    : array-like, shape (n,) — uplift prediction score.
    treatment : array-like, shape (n,) — chỉ số treatment (0 hoặc 1).

    Trả về
    ------
    (n_all, curve) — bắt đầu tại (0, 0).
    """
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = map(np.array, [y_true, uplift, treatment])

    order     = np.argsort(uplift, kind='mergesort')[::-1]
    y_true    = y_true[order];   uplift = uplift[order];   treatment = treatment[order]

    y_ctrl    = y_true.copy();   y_ctrl[treatment == 1]  = 0
    y_trmnt   = y_true.copy();   y_trmnt[treatment == 0] = 0

    thresh    = np.r_[np.where(np.diff(uplift))[0], uplift.size - 1]
    n_trmnt   = stable_cumsum(treatment)[thresh]
    y_t_cum   = stable_cumsum(y_trmnt)[thresh]
    n_all     = thresh + 1
    n_ctrl    = n_all - n_trmnt
    y_c_cum   = stable_cumsum(y_ctrl)[thresh]

    curve = (
        np.divide(y_t_cum, n_trmnt, out=np.zeros_like(y_t_cum, dtype=float), where=n_trmnt != 0)
        - np.divide(y_c_cum, n_ctrl, out=np.zeros_like(y_c_cum, dtype=float), where=n_ctrl  != 0)
    ) * n_all

    if n_all.size == 0 or curve[0] != 0 or n_all[0] != 0:
        n_all = np.r_[0, n_all];   curve = np.r_[0, curve]
    return n_all, curve


# ══════════════════════════════════════════════════════════════════════════════
# Perfect (oracle) curves
# ══════════════════════════════════════════════════════════════════════════════

def perfect_uplift_curve(y_true, treatment):
    """
    Tính Uplift curve lý tưởng (oracle — biết trước ITE thực).

    Tham số
    -------
    y_true    : array-like, shape (n,).
    treatment : array-like, shape (n,).

    Trả về
    ------
    (n_all, curve) — xem uplift_curve() để hiểu format.
    """
    check_consistent_length(y_true, treatment)
    y_true, treatment = np.array(y_true), np.array(treatment)
    cr      = np.sum((y_true == 1) & (treatment == 0))   # control responders
    tn      = np.sum((y_true == 0) & (treatment == 1))   # treated non-responders
    summand = y_true if cr > tn else treatment
    return uplift_curve(y_true, 2 * (y_true == treatment) + summand, treatment)


def perfect_qini_curve(y_true, treatment, negative_effect=True):
    """
    Tính Qini curve lý tưởng (oracle).

    Tham số
    -------
    y_true          : array-like, shape (n,).
    treatment       : array-like, shape (n,).
    negative_effect : bool — có tính negative effect không (mặc định True).

    Trả về
    ------
    (n_all, curve) — xem qini_curve() để hiểu format.
    """
    check_consistent_length(y_true, treatment)
    y_true, treatment = np.array(y_true), np.array(treatment)
    if not isinstance(negative_effect, bool):
        raise TypeError(f'negative_effect phải là bool, nhận được {type(negative_effect)}')
    if negative_effect:
        return qini_curve(y_true, y_true * treatment - y_true * (1 - treatment), treatment)
    ratio = (
        y_true[treatment == 1].sum()
        - len(y_true[treatment == 1]) * y_true[treatment == 0].sum()
          / len(y_true[treatment == 0])
    )
    return np.array([0, ratio, len(y_true)]), np.array([0, ratio, ratio])


# ══════════════════════════════════════════════════════════════════════════════
# Normalized scores
# ══════════════════════════════════════════════════════════════════════════════

def uplift_auc_score1(y_true, uplift, treatment):
    """
    Normalized AUUC (Area Under Uplift Curve).

        AUUC = (AUC_actual − AUC_baseline) / (AUC_perfect − AUC_baseline)

    Giá trị 1.0 = hoàn hảo, 0.0 = như random.

    Tham số
    -------
    y_true    : array-like, shape (n,).
    uplift    : array-like, shape (n,).
    treatment : array-like, shape (n,).

    Trả về
    ------
    float — normalized AUUC trong khoảng [0, 1].
    """
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = map(np.array, [y_true, uplift, treatment])

    xa, ya   = uplift_curve(y_true, uplift, treatment)
    xp, yp   = perfect_uplift_curve(y_true, treatment)
    xb = np.array([0, xp[-1]]);   yb = np.array([0, yp[-1]])

    base = auc(xb, yb)
    return (auc(xa, ya) - base) / (auc(xp, yp) - base)


def qini_auc_score1(y_true, uplift, treatment, negative_effect=True):
    """
    Normalized Qini coefficient.

        Qini = (AUC_actual − AUC_baseline) / (AUC_perfect − AUC_baseline)

    Tham số
    -------
    y_true          : array-like, shape (n,).
    uplift          : array-like, shape (n,).
    treatment       : array-like, shape (n,).
    negative_effect : bool — truyền vào perfect_qini_curve() (mặc định True).

    Trả về
    ------
    float — normalized Qini coefficient.
    """
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = map(np.array, [y_true, uplift, treatment])

    xa, ya   = qini_curve(y_true, uplift, treatment)
    xp, yp   = perfect_qini_curve(y_true, treatment, negative_effect)
    xb = np.array([0, xp[-1]]);   yb = np.array([0, yp[-1]])

    base    = auc(xb, yb)
    perfect = auc(xp, yp) - base
    actual  = auc(xa, ya) - base

    print(f'  [Qini debug]  perfect={perfect:.6f}  baseline={base:.6f}  actual={actual:.6f}')
    return actual / perfect


# ══════════════════════════════════════════════════════════════════════════════
# Uplift@k
# ══════════════════════════════════════════════════════════════════════════════

def uplift_at_k1(y_true, uplift, treatment, strategy, k=0.3):
    """
    Uplift@k: tính hiệu conversion rate giữa top-k treated và top-k control.

    Tham số
    -------
    y_true    : array-like, shape (n,).
    uplift    : array-like, shape (n,).
    treatment : array-like, shape (n,).
    strategy  : str  — 'overall' hoặc 'by_group'.
                'overall'  → top-k từ toàn bộ dataset (treat + control gộp chung).
                'by_group' → top-k riêng từng nhóm treatment / control.
    k         : float (0,1) → tỷ lệ; int > 0 → số lượng tuyệt đối.

    Trả về
    ------
    float — uplift@k score.
    """
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = map(np.array, [y_true, uplift, treatment])

    if strategy not in ('overall', 'by_group'):
        raise ValueError(f"strategy phải là 'overall' hoặc 'by_group', nhận được '{strategy}'")

    n      = len(y_true)
    order  = np.argsort(uplift, kind='mergesort')[::-1]
    _, cnt = np.unique(treatment, return_counts=True)
    n_ctrl, n_trmnt = cnt[0], cnt[1]
    kt = np.asarray(k).dtype.kind

    if (kt == 'i' and not (0 < k < n)) or (kt == 'f' and not (0.0 < k < 1.0)):
        raise ValueError(f'k={k} nằm ngoài khoảng hợp lệ với n={n}')
    if kt not in ('i', 'f'):
        raise ValueError(f'k phải là int hoặc float, nhận dtype kind={kt}')

    if strategy == 'overall':
        top = int(n * k) if kt == 'f' else k
        yt  = y_true[order][:top];   tt = treatment[order][:top]
        return yt[tt == 1].mean() - yt[tt == 0].mean()

    # by_group: lấy top-k riêng từng nhóm
    nc = int(n_ctrl  * k) if kt == 'f' else k
    nt = int(n_trmnt * k) if kt == 'f' else k
    sc = y_true[order][treatment[order] == 0][:nc].mean()
    st = y_true[order][treatment[order] == 1][:nt].mean()
    return st - sc
