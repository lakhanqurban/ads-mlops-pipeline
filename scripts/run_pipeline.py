import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd, cwd):
    print("\n[STEP]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({proc.returncode}): {' '.join(cmd)}")


def infer_model_path(output_dir, data_path):
    data_stem = Path(data_path).stem
    return Path(output_dir) / f"dave2_retrain_{data_stem}.h5"


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate train -> optional register -> optional validate for ASDVE"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="training_dataset/udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz",
    )
    parser.add_argument("--output-dir", type=str, default="training_dataset/models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--with-mlflow", action="store_true", default=False)
    parser.add_argument("--mlflow-experiment", type=str, default="asdve-training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="")

    parser.add_argument("--register", action="store_true", default=False)
    parser.add_argument("--model-name", type=str, default="asdve-steering-model")
    parser.add_argument("--stage", type=str, default="Staging")

    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--min-success-rate", type=float, default=0.70)
    parser.add_argument("--max-mean-cte", type=float, default=1.0)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = Path(__file__).resolve().parent

    train_cmd = [
        sys.executable,
        str(scripts_dir / "train_model.py"),
        "--data-path",
        args.data_path,
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--train-split",
        str(args.train_split),
        "--seed",
        str(args.seed),
    ]

    if args.with_mlflow:
        train_cmd.extend(["--mlflow-experiment", args.mlflow_experiment])
        if args.mlflow_tracking_uri:
            train_cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])

    run_step(train_cmd, cwd=repo_root)

    model_path = infer_model_path(args.output_dir, args.data_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model not found: {model_path}")

    if args.register:
        register_cmd = [
            sys.executable,
            str(scripts_dir / "register_model.py"),
            "--model-path",
            str(model_path),
            "--model-name",
            args.model_name,
            "--stage",
            args.stage,
        ]
        if args.mlflow_tracking_uri:
            register_cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
        run_step(register_cmd, cwd=repo_root)

    if args.validate:
        validate_cmd = [
            sys.executable,
            str(scripts_dir / "validate_model.py"),
            "--model-path",
            str(model_path),
            "--num-episodes",
            str(args.num_episodes),
            "--timeout-sec",
            str(args.timeout_sec),
            "--min-success-rate",
            str(args.min_success_rate),
            "--max-mean-cte",
            str(args.max_mean_cte),
        ]
        run_step(validate_cmd, cwd=repo_root)

    print("\nPipeline complete")
    print(f"Trained model: {model_path}")


if __name__ == "__main__":
    main()
