import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    SimpleRNN,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Input,
)
from tensorflow.keras.models import Model


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


def build_rnn_model(input_shape, output_size):
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_lstm_model(input_shape, output_size):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)

    return x + res


def build_transformer_model(input_shape, output_size):
    inputs = Input(shape=input_shape)

    x = transformer_encoder(
        inputs,
        head_size=32,
        num_heads=2,
        ff_dim=64,
        dropout=0.2
    )

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_size)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model