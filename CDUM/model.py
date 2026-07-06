"""
model.py — UpliftModel (CPM module) của CDUM framework
=======================================================
Hiện thực kiến trúc MMoE với treatment embeddings cho uplift modeling.

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

Phụ thuộc nội bộ:
    from CDUM.layers import DNN

Môi trường yêu cầu:
    TF 2.15.0  (tensorflow.keras, KHÔNG dùng standalone keras3)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import datetime

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Concatenate, Dense, Flatten, Input, Lambda,
)
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Model

# ── Nội bộ ───────────────────────────────────────────────────────────────────
from CDUM.layers import DNN


# ══════════════════════════════════════════════════════════════════════════════
# UpliftModel — CPM (Coarse-grained Preference Modeling)
# ══════════════════════════════════════════════════════════════════════════════

class UpliftModel:
    """
    Hiện thực module CPM trong CDUM framework.

    Tham số khởi tạo
    ----------------
    embedding_dim   : int  — số chiều embedding cho mỗi feature (mặc định 32).
    batch_size      : int  — batch size cho train/inference.
    train_data_size : int  — số mẫu trong tập train (để tính steps_per_epoch).
    valid_data_size : int  — số mẫu trong tập validation.
    train           : list — [X_train_df, y_train_series, t_train_series].
    test            : list — [X_val_df, y_val_series, t_val_series].
    features        : list — danh sách tên cột feature (VD ['f0', …, 'f11']).
    denominators    : list — giá trị max từng feature trên train set (để bucket).
    weights_path    : str  — đường dẫn đến file .h5 chứa pre-trained weights.
    """

    def __init__(self, embedding_dim, batch_size, train_data_size,
                 valid_data_size, train, test, features, denominators,
                 weights_path):
        self.embedding_dim   = embedding_dim
        self.batch_size      = batch_size
        self.train_data_size = train_data_size
        self.valid_data_size = valid_data_size
        self.train           = train
        self.test            = test
        self.features        = features
        self.denominators    = denominators
        self.weights_path    = weights_path

    # ── Treatment refinement ──────────────────────────────────────────────
    def _treatment_mlp(self, treat_flat, name_prefix):
        """
        MLP 2 lớp biến treatment embedding thành indicator/guidance embedding.
        - Guidance embedding : hướng dẫn expert selection (gate input).
        - Indicator embedding: nhân element-wise với tower output.

        Tên layer PHẢI khớp với tên trong file epoch_criteo.h5.
        """
        h = Dense(64, activation='relu', name=f'{name_prefix}_hidden1')(treat_flat)
        return Dense(32, activation='sigmoid', name=f'{name_prefix}_out')(h)

    # ── MMoE + Tower ──────────────────────────────────────────────────────
    def _mmoe_towers(self, mlp_inputs, task_names, masks, indicator_emb, gate_emb):
        """
        Multi-gate Mixture-of-Experts (MMoE) với 3 shared experts, 2 task towers.

        Tham số
        -------
        mlp_inputs   : (B, d)   — feature embedding đã flatten.
        task_names   : list     — ['treat', 'base'].
        masks        : list     — [treat_mask, 1−treat_mask].
        indicator_emb: (B, 32)  — nhân element-wise vào tower output.
        gate_emb     : (B, 32)  — input cho gate networks.
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
            gate_h = DNN((), name=f'gate_{task_name}')(gate_emb)   # identity khi ()
            gate_w = Dense(
                NUM_EXPERTS, use_bias=False, activation='softmax',
                name=f'gate_softmax_{task_name}'
            )(gate_h)                                               # (B, 3)
            gate_w = Lambda(
                lambda x: tf.expand_dims(x, axis=-1),
                name=f'gate_expand_{task_name}'
            )(gate_w)                                               # (B, 3, 1)

            # Weighted sum: (B,3,64) * (B,3,1) → sum axis=1 → (B,64)
            expert_mix = Lambda(
                lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
                name=f'gate_mul_expert_{task_name}'
            )([expert_stack, gate_w])
            mmoe_outs.append(expert_mix)

        # --- Task towers: DNN(32) + indicator mask + softplus output ---
        task_outs = []
        for task_name, mix, mask in zip(task_names, mmoe_outs, masks):
            tower  = DNN((32,), name=f'tower_{task_name}')(mix)
            tower  = indicator_emb * tower
            output = Dense(1, activation='softplus',
                           name=f'{task_name}_task_out')(tower)
            # Masking: treat tower → 0 khi sample là control (và ngược lại)
            task_outs.append(
                tf.keras.layers.multiply([output, mask], name=f'{task_name}_task')
            )
        return task_outs

    # ── Build Keras functional model ──────────────────────────────────────
    def build_model(self):
        """
        Xây dựng và trả về Keras functional model chưa compile.
        """
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
            )(bucket)                          # (B, 1, embedding_dim)
            feat_inputs.append(inp)
            feat_embeds.append(emb)

        # Treatment input (0 hoặc 1) → embedding (dim×4)
        treat_inp = Input(shape=(1,), name='exp_name_input')
        treat_emb = layers.Embedding(
            input_dim=2, output_dim=self.embedding_dim * 4,
            mask_zero=True, name='treat_embedding'
        )(treat_inp)                           # (B, 1, embedding_dim*4)
        feat_inputs.append(treat_inp)          # treatment là input cuối cùng

        # Nối tất cả feature embeddings → flatten
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

    # ── Callbacks & loss config ───────────────────────────────────────────
    def _model_conf(self):
        """Khởi tạo loss weights, metrics, checkpoint callback, LR scheduler."""
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        ckpt_dir = os.path.join(
            os.path.dirname(self.weights_path),
            f'uplift_model_{ts}_emb{self.embedding_dim}_bs{self.batch_size}'
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        self.loss_weights = {'treat_task': 1.0, 'base_task': 1.0}
        self.metrics_dict = {
            'treat_task': [MeanSquaredError()],
            'base_task':  [MeanSquaredError()],
        }
        self.checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'epoch_{epoch:02d}.h5'),
            save_weights_only=True,
            save_freq='epoch',
        )
        self.reduce_lr_cb = ReduceLROnPlateau(
            monitor='val_loss', factor=0.6, patience=2,
            min_lr=1e-6, verbose=1,
        )

    # ── GPU setup + build / (optional) train ─────────────────────────────
    def model_train(self, train_mode=False, num_epochs=10):
        """
        Thiết lập GPU memory growth, build model, compile.
        Nếu train_mode=True thì train bằng generator, ngược lại chỉ build + compile.

        Tham số
        -------
        train_mode : bool — True → train từ đầu; False → chỉ build để load weights.
        num_epochs : int  — số epoch khi train_mode=True.

        Trả về
        ------
        model : tf.keras.Model đã compile (chưa load weights).
        """
        from preprocess.data_loader import make_data_generator

        # Bật memory growth để tránh TF chiếm hết VRAM ngay từ đầu
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f'[GPU setup warning] {e}')

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

            if train_mode:
                train_gen   = make_data_generator(self.train, self.batch_size, self.features)
                valid_gen   = make_data_generator(self.test,  self.batch_size, self.features)
                steps_train = self.train_data_size // self.batch_size
                steps_valid = self.valid_data_size // self.batch_size
                print(f'Training: {steps_train} steps/epoch | Validation: {steps_valid} steps/epoch')
                model.fit(
                    train_gen,
                    epochs           = num_epochs,
                    steps_per_epoch  = steps_train,
                    validation_data  = valid_gen,
                    validation_steps = steps_valid,
                    verbose          = 1,
                    callbacks        = [self.checkpoint_cb, self.reduce_lr_cb],
                )

        return model
