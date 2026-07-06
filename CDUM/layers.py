"""
layers.py — Custom Keras layers cho CPM module của CDUM
========================================================
Chứa các building blocks dùng chung trong kiến trúc MMoE:
  - activation_layer(): factory trả về Keras activation Layer
  - DNN: stacked fully-connected layer (Linear → BN → Activation → Dropout)

Môi trường yêu cầu:
    TF 2.15.0  (tensorflow.keras, KHÔNG dùng standalone keras3)
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf

from tensorflow.keras.initializers import GlorotNormal, Zeros
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Dropout, Layer,
)
from tensorflow.keras.regularizers import l2

# Python 2/3 compat cho str check
try:
    unicode
except NameError:
    unicode = str


# ══════════════════════════════════════════════════════════════════════════════
# Activation factory
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


# ══════════════════════════════════════════════════════════════════════════════
# DNN: Stacked fully-connected layers
# ══════════════════════════════════════════════════════════════════════════════

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
            # Linear transform: W·x + b  (tensordot + bias_add để tương thích
            # với input shape nhiều chiều)
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
