from src.preprocessing import (
    set_seed,
    load_and_clean_data,
    time_split_dataframe,
    scale_features_and_target,
    create_sequences,
)
from src.models import build_cnn_model, build_rnn_model, build_lstm_model
from src.train import train_and_evaluate_model, plot_loss_curve, plot_actual_vs_predicted


def get_model(model_name, input_shape, output_size):
    if model_name == "cnn":
        return build_cnn_model(input_shape, output_size)
    elif model_name == "rnn":
        return build_rnn_model(input_shape, output_size)
    elif model_name == "lstm":
        return build_lstm_model(input_shape, output_size)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def main():
    set_seed(42)

    model_name = "lstm"   # change this to "rnn" or "lstm"
    file_path = "data/raw/bitcoin.csv"
    feature_cols = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    target_col = "Price"
    window_size = 60
    horizon = 7

    print("Loading dataset...")
    df = load_and_clean_data(file_path)

    print("Splitting data...")
    train_df, test_df = time_split_dataframe(df, train_ratio=0.8)

    print("Scaling features and target separately...")
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_features_and_target(
        train_df, test_df, feature_cols, target_col
    )

    print("Generating sequences...")
    X_train, y_train = create_sequences(
        X_train_scaled,
        y_train_scaled,
        window_size=window_size,
        horizon=horizon,
    )

    X_test, y_test = create_sequences(
        X_test_scaled,
        y_test_scaled,
        window_size=window_size,
        horizon=horizon,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print(f"\nBuilding {model_name.upper()} model...")
    model = get_model(
        model_name=model_name,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_size=horizon
    )
    model.summary()

    print(f"\nTraining {model_name.upper()} model...")
    history, y_test_actual, y_pred_actual, metrics = train_and_evaluate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_scaler=target_scaler,
        epochs=30,
        batch_size=32
    )

    print(f"\n{model_name.upper()} 1-Day Forecast Metrics:")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAPE : {metrics['MAPE']:.4f}%")

    plot_loss_curve(history)
    plot_actual_vs_predicted(y_test_actual.flatten(), y_pred_actual.flatten(), num_points=100)


if __name__ == "__main__":
    main()