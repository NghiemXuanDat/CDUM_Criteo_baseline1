"""
criteo.py — CPM (Coarse-grained Preference Modeling) module của CDUM
=======================================================================
Paper : "Enhancing Online Video Recommendation via a Coarse-to-fine
         Dynamic Uplift Modeling Framework", RecSys 2025 (Kuaishou/Tsinghua)
Dataset: Criteo Uplift v2.1  (~14M rows, 12 features, binary treatment)
Task   : Uplift modeling — ước lượng ITE (Individual Treatment Effect):
             uplift = P(visit | treatment=1) − P(visit | treatment=0)
Model  : MMoE (3 shared experts, 2 task gates) với indicator/guidance
         treatment embeddings + 2 task towers (treat / base).
Metrics: AUUC, QINI, LIFT@30%

Môi trường đã kiểm tra:
    Python  3.x
    TF      2.15.0  (tensorflow-gpu hoặc tensorflow)
    Keras   2.15.0  (tích hợp trong TF 2.15, KHÔNG cài standalone keras3)
    scikit  1.8.0
    pandas  2.3.3
    numpy   1.26.4
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import random
import datetime

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length

# Tất cả Keras imports đều dùng tensorflow.keras để tránh xung đột
# với keras standalone (keras 3.x) nếu vô tình được cài song song.
# Trong TF 2.15 thì tensorflow.keras == keras 2.15 (cùng một package).
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotNormal, Zeros
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Concatenate,
    Dense, Dropout, Flatten, Input, Lambda, Layer,
)
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ── Cấu hình toàn cục — chỉnh sửa tại đây nếu cần ───────────────────────────
TRAIN_MODE   = False   # True → train từ đầu; False → chỉ load weights và evaluate
DATA_PATH    = '/home/datnghiemxuan/Documents/Criteo_CDUM/data/criteo-uplift-v2.1.csv.gz'
WEIGHTS_PATH = '/home/datnghiemxuan/Documents/Criteo_CDUM/CDUM/epoch_criteo.h5'

EMBEDDING_DIM = 32     # số chiều embedding cho mỗi feature và treatment
BATCH_SIZE    = 4096   # batch size cho train và inference
NUM_EPOCHS    = 10     # số epoch khi TRAIN_MODE = True

FEATURE_COLS  = [f'f{i}' for i in range(12)]   # f0 … f11 (12 đặc trưng số)
LABEL_COL     = 'visit'       # nhãn nhị phân (0/1) — target label
TREAT_COL     = 'treatment'   # nhãn nhị phân (0/1) — nhóm điều trị / kiểm soát

# Python 2/3 compat cho str check
try:
    unicode
except NameError:
    unicode = str


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════════════

def activation_layer(activation):
    """
    Trả về một Keras Layer tương ứng với tên activation hoặc Layer subclass.
    Hỗ trợ tất cả chuỗi Keras hợp lệ (relu, sigmoid, softplus, …).
    """
    if isinstance(activation, (str, unicode)):
        return Activation(activation)
    elif isinstance(activation, type) and issubclass(activation, Layer):
        return activation()
    raise ValueError(
        "Invalid activation '%s'. Dùng chuỗi Keras hoặc một Layer subclass." % activation
    )


class DNN(Layer):
    """
    Stacked fully-connected layers:  Linear → [BN] → Activation → Dropout.

    Dùng cho: expert networks, gate networks, tower networks trong MMoE.

    Tham số
    -------
    hidden_units      : tuple số neuron từng lớp, VD (128, 64). Nếu () → identity.
    activation        : chuỗi Keras, mặc định 'relu'.
    l2_reg            : hệ số L2 regularization trên weight.
    dropout_rate      : tỷ lệ dropout [0, 1).
    use_bn            : có dùng BatchNormalization không.
    output_activation : activation của lớp cuối (overrides activation).
    seed              : seed cho initializer và dropout.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0,
                 dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units      = tuple(hidden_units)
        self.activation        = activation
        self.l2_reg            = l2_reg
        self.dropout_rate      = dropout_rate
        self.use_bn            = use_bn
        self.output_activation = output_activation
        self.seed              = seed
        super().__init__(**kwargs)

    def build(self, input_shape):
        sizes = [int(input_shape[-1])] + list(self.hidden_units)

        # Khởi tạo weight và bias cho mỗi lớp Linear
        self.kernels = [
            self.add_weight(
                name=f'kernel{i}',
                shape=(sizes[i], sizes[i + 1]),
                initializer=GlorotNormal(seed=self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True,
            ) for i in range(len(self.hidden_units))
        ]
        self.bias = [
            self.add_weight(
                name=f'bias{i}',
                shape=(self.hidden_units[i],),
                initializer=Zeros(),
                trainable=True,
            ) for i in range(len(self.hidden_units))
        ]

        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in self.hidden_units]

        self.dropout_layers    = [
            Dropout(self.dropout_rate, seed=self.seed + i)
            for i in range(len(self.hidden_units))
        ]
        self.activation_layers = [
            activation_layer(self.activation) for _ in self.hidden_units
        ]
        # Ghi đè activation lớp cuối nếu output_activation được chỉ định
        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for i in range(len(self.hidden_units)):
            # Linear transform: W·x + b  (dùng tensordot + bias_add thay vì Dense
            # để tương thích với input shape nhiều chiều)
            x = tf.nn.bias_add(
                tf.tensordot(x, self.kernels[i], axes=(-1, 0)), self.bias[i]
            )
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)
            try:
                x = self.activation_layers[i](x, training=training)
            except TypeError:
                # Một số custom activation không nhận tham số training
                x = self.activation_layers[i](x)
            x = self.dropout_layers[i](x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        if self.hidden_units:
            return tuple(input_shape[:-1]) + (self.hidden_units[-1],)
        return tuple(input_shape)

    def get_config(self):
        cfg = {
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'dropout_rate': self.dropout_rate,
            'output_activation': self.output_activation,
            'seed': self.seed,
        }
        return {**super().get_config(), **cfg}


def make_data_generator(data, batch_size, features):
    """
    Generator vô hạn (infinite) sinh ra các batch (inputs, targets) cho Keras.

    Tham số
    -------
    data     : [X_df, y_series, treatment_series]
    batch_size: kích thước batch
    features : danh sách tên cột feature

    Yields
    ------
    (input_list, [treat_label, ctrl_label])
        input_list  = [feat_0_array, …, feat_11_array, treatment_mask_array]
        treat_label = y * treatment_mask          (chỉ có giá trị khi treatment=1)
        ctrl_label  = y * (1 − treatment_mask)    (chỉ có giá trị khi treatment=0)

    Ghi chú: dùng với model.fit(generator, steps_per_epoch=...) để keras tự
    dừng sau steps_per_epoch bước.
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


# ══════════════════════════════════════════════════════════════════════════════
# Uplift Model — CPM (Coarse-grained Preference Modeling)
# ══════════════════════════════════════════════════════════════════════════════

class UpliftModel:
    """
    Hiện thực module CPM trong CDUM framework.

    Kiến trúc tổng quan
    ───────────────────────────────────────────────────────────────────────
    [f0_input … f11_input]  →  Embedding  →  Concat  →  Flatten
                                                               ↓
    [treatment_input]  →  Embedding  →  Flatten  →  _treatment_mlp
                                                    ↙           ↘
                                            indicator_emb   guide_emb
                                                                  ↓
                              mlp_inputs  ──────────►  MMoE (3 experts, 2 gates)
                                                                  ↓
                              indicator_emb  ──►  Tower × treat  (masked by treatment)
                              indicator_emb  ──►  Tower × base   (masked by 1-treatment)

    Output: [treat_output * treat_mask,  base_output * (1 − treat_mask)]

    Tại inference:
        treat_mask = 1  →  treat_output có giá trị,  base_output = 0
        treat_mask = 0  →  base_output có giá trị,   treat_output = 0
    """

    def __init__(self, embedding_dim, batch_size, train_data_size,
                 valid_data_size, train, test, features, denominators):
        self.embedding_dim   = embedding_dim
        self.batch_size      = batch_size
        self.train_data_size = train_data_size
        self.valid_data_size = valid_data_size
        self.train           = train         # [X_train_df, y_train, t_train]
        self.test            = test          # [X_val_df,   y_val,   t_val]
        self.features        = features      # danh sách tên cột feature
        self.denominators    = denominators  # max value của mỗi feature (từ train set)

    # ── Treatment refinement ──────────────────────────────────────────────
    def _treatment_mlp(self, treat_flat, name_prefix):
        """
        MLP 2 lớp để biến treatment embedding thành indicator/guidance embedding.
        - Guidance embedding: hướng dẫn expert selection (gate input).
        - Indicator embedding: mask nhân với tower output (indicative role).
        """
        # Tên layer PHẢI khớp với tên trong file epoch_criteo.h5
        h = Dense(64, activation='relu', name=f'{name_prefix}_hidden1')(treat_flat)
        return Dense(32, activation='sigmoid', name=f'{name_prefix}_out')(h)

    # ── MMoE + Tower ──────────────────────────────────────────────────────
    def _mmoe_towers(self, mlp_inputs, task_names, masks, indicator_emb, gate_emb):
        """
        Multi-gate Mixture-of-Experts (MMoE) với 3 shared experts và 2 task towers.

        Tham số
        -------
        mlp_inputs   : (B, d) — feature embedding đã flatten
        task_names   : ['treat', 'base']
        masks        : [treat_mask, 1−treat_mask] — áp lên output của mỗi tower
        indicator_emb: (B, 32) — nhân element-wise vào tower output
        gate_emb     : (B, 32) — input cho gate networks
        """
        NUM_EXPERTS = 3

        # --- Shared expert networks: 3 DNN(128→64) song song ---
        expert_outs = [
            DNN((128, 64), name=f'expert_{i}')(mlp_inputs)
            for i in range(NUM_EXPERTS)
        ]
        # Stack experts: list of (B, 64)  →  (B, 3, 64)
        expert_stack = Lambda(
            lambda x: tf.stack(x, axis=1), name='expert_stack'
        )(expert_outs)

        # --- Per-task gating: mỗi task có gate riêng ---
        mmoe_outs = []
        for task_name in task_names:
            # gate_emb → (identity DNN) → softmax(num_experts)
            gate_h = DNN((), name=f'gate_{task_name}')(gate_emb)   # identity khi ()
            gate_w = Dense(
                NUM_EXPERTS, use_bias=False, activation='softmax',
                name=f'gate_softmax_{task_name}'
            )(gate_h)                                               # (B, 3)
            gate_w = Lambda(
                lambda x: tf.expand_dims(x, axis=-1),
                name=f'gate_expand_{task_name}'
            )(gate_w)                                               # (B, 3, 1)

            # Weighted sum over experts: (B,3,64) * (B,3,1) → sum axis=1 → (B,64)
            expert_mix = Lambda(
                lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
                name=f'gate_mul_expert_{task_name}'
            )([expert_stack, gate_w])
            mmoe_outs.append(expert_mix)

        # --- Task towers: DNN(32) + indicator mask + softplus output ---
        task_outs = []
        for task_name, mix, mask in zip(task_names, mmoe_outs, masks):
            tower  = DNN((32,), name=f'tower_{task_name}')(mix)
            # indicator_emb hoạt động như learnable gate trên tower output
            tower  = indicator_emb * tower
            output = Dense(1, activation='softplus',
                           name=f'{task_name}_task_out')(tower)
            # Masking: treat tower → 0 khi sample là control (và ngược lại)
            task_outs.append(
                tf.keras.layers.multiply([output, mask], name=f'{task_name}_task')
            )
        return task_outs

    # ── Xây dựng Keras functional model ──────────────────────────────────
    def build_model(self):
        feat_inputs = []
        feat_embeds = []

        # Feature encoder: mỗi feature số → bucket index → embedding
        for idx, feat in enumerate(self.features):
            inp   = Input(shape=(1,), name=f'{feat}_input')
            denom = max(1e-8, self.denominators[idx])
            dim   = 100   # số buckets

            # Chuẩn hóa về [0, dim] → ép kiểu int → clamp [0, dim]
            bucket = tf.minimum(
                tf.maximum(tf.cast(inp / denom * dim, tf.int32), 0), dim
            )
            emb = layers.Embedding(
                input_dim=dim + 1, output_dim=self.embedding_dim,
                mask_zero=True, name=f'{feat}_embedding'
            )(bucket)                          # shape: (B, 1, embedding_dim)
            feat_inputs.append(inp)
            feat_embeds.append(emb)

        # Treatment input (0 hoặc 1) → embedding (dim×4 để mang nhiều thông tin hơn)
        treat_inp = Input(shape=(1,), name='exp_name_input')
        treat_emb = layers.Embedding(
            input_dim=2, output_dim=self.embedding_dim * 4,
            mask_zero=True, name='treat_embedding'
        )(treat_inp)                           # shape: (B, 1, embedding_dim*4)
        feat_inputs.append(treat_inp)          # treatment là input cuối cùng

        # Nối tất cả feature embeddings → flatten thành vector
        concat_emb = Concatenate(axis=1)(feat_embeds)  # (B, n_feat, emb_dim)
        mlp_inputs = Flatten()(concat_emb)             # (B, n_feat * emb_dim)

        # Treatment refinement → indicator + guidance embedding
        treat_flat    = Flatten()(treat_emb)           # (B, emb_dim*4)
        indicator_emb = self._treatment_mlp(treat_flat, 'indicator_emb')  # (B, 32)
        guide_emb     = self._treatment_mlp(treat_flat, 'guide_emb')      # (B, 32)

        # MMoE: treat tower dùng treat_mask, base tower dùng (1 - treat_mask)
        task_outputs = self._mmoe_towers(
            mlp_inputs,
            task_names    = ['treat', 'base'],
            masks         = [treat_inp, tf.ones_like(treat_inp) - treat_inp],
            indicator_emb = indicator_emb,
            gate_emb      = guide_emb,
        )

        model = Model(inputs=feat_inputs, outputs=task_outputs)
        model.summary()
        return model

    # ── Cấu hình callbacks và loss ────────────────────────────────────────
    def _model_conf(self):
        """Khởi tạo loss weights, metrics, checkpoint callback, LR scheduler."""
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        ckpt_dir = os.path.join(
            os.path.dirname(WEIGHTS_PATH),
            f'uplift_model_{ts}_emb{self.embedding_dim}_bs{self.batch_size}'
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        # Trọng số loss cho 2 tasks (treat và base)
        self.loss_weights = {'treat_task': 1.0, 'base_task': 1.0}
        self.metrics_dict = {
            'treat_task': [MeanSquaredError()],
            'base_task':  [MeanSquaredError()],
        }
        # Lưu weights sau mỗi epoch
        self.checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'epoch_{epoch:02d}.h5'),
            save_weights_only=True,
            save_freq='epoch',
        )
        # Giảm learning rate khi val_loss không cải thiện sau 2 epoch
        self.reduce_lr_cb = ReduceLROnPlateau(
            monitor='val_loss', factor=0.6, patience=2,
            min_lr=1e-6, verbose=1,
        )

    # ── GPU setup + build / train ─────────────────────────────────────────
    def model_train(self):
        """
        Thiết lập GPU memory growth, build model, compile, và train nếu TRAIN_MODE=True.
        Trả về model đã build (chưa load weights, chỉ compile xong).
        """
        # Bật memory growth để tránh TF chiếm hết VRAM ngay từ đầu
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f'[GPU setup warning] {e}')

        # MirroredStrategy tự động phân phối sang multi-GPU nếu có
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            self._model_conf()
            model = self.build_model()
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss={'treat_task': 'huber', 'base_task': 'huber'},
                loss_weights=self.loss_weights,
                metrics=self.metrics_dict,
            )

            if TRAIN_MODE:
                train_gen = make_data_generator(self.train, self.batch_size, self.features)
                valid_gen = make_data_generator(self.test,  self.batch_size, self.features)
                steps_train = self.train_data_size // self.batch_size
                steps_valid = self.valid_data_size // self.batch_size
                print(f'Training: {steps_train} steps/epoch | Validation: {steps_valid} steps/epoch')
                model.fit(
                    train_gen,
                    epochs           = NUM_EPOCHS,
                    steps_per_epoch  = steps_train,
                    validation_data  = valid_gen,
                    validation_steps = steps_valid,
                    verbose          = 1,
                    callbacks        = [self.checkpoint_cb, self.reduce_lr_cb],
                )

        return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics: Uplift curve, Qini curve, AUUC, QINI, LIFT@k
# Thuật toán giữ nguyên theo paper CDUM và scikit-uplift convention.
# ══════════════════════════════════════════════════════════════════════════════

def qini_curve(y_true, uplift, treatment):
    """
    Tính Qini curve.

    Sắp xếp samples theo uplift prediction giảm dần, tính lũy kế:
        curve(n) = cumTreated(n) − cumCtrl(n) × (n_treated / n_ctrl)

    Trả về: (num_targeted_array, qini_values_array) bắt đầu tại (0, 0).
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


def uplift_curve(y_true, uplift, treatment):
    """
    Tính Uplift curve.

    curve(n) = [convRate_treated(n) − convRate_ctrl(n)] × n

    Trả về: (num_targeted_array, uplift_values_array) bắt đầu tại (0, 0).
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


def perfect_uplift_curve(y_true, treatment):
    """Tính Uplift curve lý tưởng (oracle — biết trước ITE thực)."""
    check_consistent_length(y_true, treatment)
    y_true, treatment = np.array(y_true), np.array(treatment)
    cr      = np.sum((y_true == 1) & (treatment == 0))   # control responders
    tn      = np.sum((y_true == 0) & (treatment == 1))   # treated non-responders
    summand = y_true if cr > tn else treatment
    # Perfect uplift score = 2*(y==t) + summand
    return uplift_curve(y_true, 2 * (y_true == treatment) + summand, treatment)


def perfect_qini_curve(y_true, treatment, negative_effect=True):
    """Tính Qini curve lý tưởng (oracle)."""
    check_consistent_length(y_true, treatment)
    y_true, treatment = np.array(y_true), np.array(treatment)
    if not isinstance(negative_effect, bool):
        raise TypeError(f'negative_effect phải là bool, nhận được {type(negative_effect)}')
    if negative_effect:
        # Oracle score: y*t (responders treated) − y*(1−t) (responders in control)
        return qini_curve(y_true, y_true * treatment - y_true * (1 - treatment), treatment)
    # Không tính negative effect: đường lý tưởng đơn giản hơn
    ratio = (
        y_true[treatment == 1].sum()
        - len(y_true[treatment == 1]) * y_true[treatment == 0].sum()
          / len(y_true[treatment == 0])
    )
    return np.array([0, ratio, len(y_true)]), np.array([0, ratio, ratio])


def uplift_auc_score1(y_true, uplift, treatment):
    """
    Normalized AUUC (Area Under Uplift Curve).

    AUUC = (AUC_actual − AUC_baseline) / (AUC_perfect − AUC_baseline)
    Giá trị 1.0 = hoàn hảo, 0.0 = như random.
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


def uplift_at_k1(y_true, uplift, treatment, strategy, k=0.3):
    """
    Uplift@k: tính hiệu conversion rate giữa top-k treated và top-k control.

    strategy='overall' : top-k từ toàn bộ dataset (cả treatment + control gộp chung)
    strategy='by_group': top-k riêng từng nhóm treatment / control
    k                  : float (0,1) → tỷ lệ; int > 0 → số lượng tuyệt đối
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


# ══════════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Load toàn bộ dataset từ file .csv.gz ───────────────────────────────────
print(f'\n[1/6] Loading dataset: {DATA_PATH}')
df_all = pd.read_csv(DATA_PATH)
print(f'      Total rows : {len(df_all):,}')
print(f'      Columns    : {list(df_all.columns)}')
print(f'      Treatment ratio (1/0): {df_all[TREAT_COL].mean():.4f}')
print(f'      Positive rate (visit): {df_all[LABEL_COL].mean():.4f}')

# ── 2. Split 8:1:1 (train : val : test) với random_state cố định ─────────────
print('\n[2/6] Splitting data 8:1:1 ...')
df_train, df_tmp  = train_test_split(df_all, test_size=0.2, random_state=42)
df_val,   df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)
print(f'      train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}')

# Tính max value từng feature trên train set để chuẩn hóa bucket (không dùng val/test)
denominators = [df_train[f].max() for f in FEATURE_COLS]

x_train = df_train[FEATURE_COLS];  y_train = df_train[LABEL_COL];  t_train = df_train[TREAT_COL]
x_val   = df_val[FEATURE_COLS];    y_val   = df_val[LABEL_COL];    t_val   = df_val[TREAT_COL]
x_test  = df_test[FEATURE_COLS];   y_test  = df_test[LABEL_COL];   t_test  = df_test[TREAT_COL]

# ── 3. Khởi tạo model và build / train ────────────────────────────────────────
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
)
uplift_model = uplift_obj.model_train()

# ── 4. Load pre-trained weights ───────────────────────────────────────────────
# Dùng by_name=True để match theo tên layer thay vì theo thứ tự.
# Dùng skip_mismatch=True vì file h5 được save bằng TF phiên bản cũ —
# các DNN layers (expert_*, tower_*) không được lưu đúng cách trong h5 đó,
# nên skip chúng và load những layers có sẵn.
print(f'\n[4/6] Loading weights: {WEIGHTS_PATH}')
uplift_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
print('      Weights loaded (by_name=True, skip_mismatch=True).')
print('      NOTE: expert/tower layer weights không có trong h5 → dùng random init.')
print('      Để có kết quả đầy đủ, cần train lại với TRAIN_MODE=True.')

# ── 5. Inference trên test set ────────────────────────────────────────────────
# Dự đoán uplift = treat_score − base_score bằng cách chạy inference 2 lần:
#   - treat_inputs  : tất cả samples với treatment=1  → lấy treat_output (res_1)
#   - control_inputs: tất cả samples với treatment=0  → lấy base_output  (res_0)
print('\n[5/6] Running inference on test set ...')

base_feats      = [x_test[col].to_numpy() for col in FEATURE_COLS]
treat_inputs    = base_feats + [np.ones(len(x_test), dtype=np.float32)]
control_inputs  = base_feats + [np.zeros(len(x_test), dtype=np.float32)]

# predict trả về [treat_output, base_output]
# treat_mask=1: treat_output = giá trị thực, base_output = 0 (do masking)
# treat_mask=0: treat_output = 0,             base_output = giá trị thực
res_1, _  = uplift_model.predict(treat_inputs,   batch_size=BATCH_SIZE, verbose=0)
_,  res_0 = uplift_model.predict(control_inputs, batch_size=BATCH_SIZE, verbose=0)

uplift_scores = (res_1 - res_0).reshape(-1)
print(f'      Uplift scores: shape={uplift_scores.shape}  '
      f'mean={uplift_scores.mean():.4f}  std={uplift_scores.std():.4f}')

# ── 6. Đánh giá với AUUC, QINI, LIFT@30% ─────────────────────────────────────
print('\n[6/6] Evaluating ...')
auuc = uplift_auc_score1(y_true=y_test, uplift=uplift_scores, treatment=t_test)
qini = qini_auc_score1 (y_true=y_test, uplift=uplift_scores, treatment=t_test)
lift = uplift_at_k1    (y_true=y_test, uplift=uplift_scores, treatment=t_test,
                        strategy='overall', k=0.3)

print('\n╔══════════════════════════╗')
print('║  Evaluation Results      ║')
print('╠══════════════════════════╣')
print(f'║  AUUC     : {auuc:>10.6f}  ║')
print(f'║  QINI     : {qini:>10.6f}  ║')
print(f'║  LIFT@30% : {lift:>10.6f}  ║')
print('╚══════════════════════════╝')
