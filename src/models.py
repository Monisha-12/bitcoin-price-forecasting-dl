import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    SimpleRNN,
    LSTM,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SimpleRNN, LSTM

# ================= CNN =================
def build_cnn_model(input_shape, output_size):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),

        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(output_size)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


# ================= RNN =================
def build_rnn_model(input_shape, output_size):
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ================= LSTM =================
def build_lstm_model(input_shape, output_size):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ================= Transformer =================
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_transformer_model(input_shape, output_size):
    inputs = tf.keras.Input(shape=input_shape)

    x = transformer_block(inputs, head_size=64, num_heads=2, ff_dim=64)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(output_size)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model