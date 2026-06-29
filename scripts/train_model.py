import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from keras import layers, models, optimizers

from data_loader import NPZSteeringDataset, SteeringBatchGenerator, split_indices


def build_nvidia_style_model(input_shape=(160, 320, 3)):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Lambda(lambda x: x / 127.5 - 1.0),
            layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
            layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
            layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(1),
        ]
    )
    return model


def maybe_start_mlflow(args, run_name):
    try:
        import mlflow
    except Exception:
        return None

    tracking_uri = args.mlflow_tracking_uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment)
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "npz_path": str(args.data_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_split": args.train_split,
            "seed": args.seed,
            "model_type": "nvidia-style",
        }
    )
    return run


def _to_json_safe_history(history_dict):
    """Convert numpy scalar values in Keras history to plain Python floats."""
    safe = {}
    for key, values in history_dict.items():
        safe[key] = [float(v) for v in values]
    return safe


def main():
    parser = argparse.ArgumentParser(description="Train steering model from training_dataset NPZ")
    parser.add_argument(
        "--data-path",
        type=str,
        default="training_dataset/udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
        help="Path to NPZ file with observations/actions",
    )
    parser.add_argument("--output-dir", type=str, default="training_dataset/models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-experiment", type=str, default="asdve-training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NPZSteeringDataset(args.data_path)
    train_idx, val_idx = split_indices(len(dataset), train_split=args.train_split, seed=args.seed)

    train_gen = SteeringBatchGenerator(
        observations=dataset.observations,
        steering=dataset.steering,
        indices=train_idx,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_gen = SteeringBatchGenerator(
        observations=dataset.observations,
        steering=dataset.steering,
        indices=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_nvidia_style_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    run_name = f"train-{Path(args.data_path).stem}"
    mlflow_run = maybe_start_mlflow(args, run_name=run_name)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    model_name = f"dave2_retrain_{Path(args.data_path).stem}.h5"
    model_path = output_dir / model_name
    model.save(str(model_path))

    history_path = output_dir / f"history_{Path(args.data_path).stem}.json"
    history_json = _to_json_safe_history(history.history)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f)

    final_metrics = {
        "train_loss": float(history.history["loss"][-1]),
        "train_mae": float(history.history["mae"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "val_mae": float(history.history["val_mae"][-1]),
    }

    print("Training complete")
    print(f"Model: {model_path}")
    print(f"History: {history_path}")
    print(f"Metrics: {final_metrics}")

    if mlflow_run is not None:
        import mlflow

        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(history_path), artifact_path="training")
        mlflow.log_param("train_samples", int(len(train_idx)))
        mlflow.log_param("val_samples", int(len(val_idx)))
        mlflow.end_run()


if __name__ == "__main__":
    main()
