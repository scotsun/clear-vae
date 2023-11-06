"""Models."""

import tensorflow as tf
import keras
from keras import layers
from keras import Sequential
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.regularizers import L1
from keras.callbacks import EarlyStopping


class TransformerBlock(layers.Layer):
    """Transformer block."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """Init."""
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        """Call."""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ASDTransformer(keras.Model):
    """Model the diagnoses code in terms of sequence data format."""

    def __init__(
        self,
        vocab_sizes: list[int],
        maxlens: list[int],
        embed_dim: int,
        ff_dim: int,
        struct_dim: int = 0,
        **kwargs
    ):
        """Init."""
        super().__init__(**kwargs)
