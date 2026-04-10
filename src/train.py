import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_scaler,
    epochs=50,
    batch_size=32
):
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

    history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

    y_pred = model.predict(X_test, verbose=0)

    y_test_actual = target_scaler.inverse_transform(y_test)
    y_pred_actual = target_scaler.inverse_transform(y_pred)

    y_test_flat = y_test_actual.flatten()
    y_pred_flat = y_pred_actual.flatten()

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    mape = calculate_mape(y_test_flat, y_pred_flat)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    return history, y_test_actual, y_pred_actual, metrics


def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, num_points=100):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:num_points], label="Actual Price")
    plt.plot(y_pred[:num_points], label="Predicted Price")
    plt.title("Actual vs Predicted Bitcoin Price (CNN - 1 Day Forecast)")
    plt.xlabel("Test Samples")
    plt.ylabel("Bitcoin Price")
    plt.legend()
    plt.grid(True)
    plt.show()