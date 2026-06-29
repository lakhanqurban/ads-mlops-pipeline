import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def latest_file(pattern):
    files = [str(path) for path in Path.cwd().glob(pattern)]
    if not files:
        return None
    return max(files, key=lambda p: Path(p).stat().st_mtime)


def latest_matching_file(patterns):
    for pattern in patterns:
        found = latest_file(pattern)
        if found is not None:
            return found
    return None


def run_validation(model_path, num_episodes, timeout_sec):
    cmd = [
        "python3",
        "main.py",
        "--model-path",
        str(model_path),
        "--num-episodes",
        str(num_episodes),
        "--disable-real-time-stl",
        "--stl-live-view",
        "terminal",
    ]

    print("Running:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )

    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Validation run failed with code {proc.returncode}")


def compute_gate_metrics(output_csv):
    df = pd.read_csv(output_csv)
    if "road_id" in df.columns:
        df = df[df["road_id"].astype(str) != "TOTAL_SIMULATION_TIME"]

    success_rate = float(df["success"].mean()) if "success" in df.columns and len(df) else 0.0
    mean_max_cte = float(df["max_cte"].mean()) if "max_cte" in df.columns and len(df) else float("nan")
    mean_max_speed = float(df["max_speed"].mean()) if "max_speed" in df.columns and len(df) else float("nan")

    return {
        "rows": int(len(df)),
        "success_rate": success_rate,
        "mean_max_cte": mean_max_cte,
        "mean_max_speed": mean_max_speed,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a trained model by running main.py")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--min-success-rate", type=float, default=0.70)
    parser.add_argument("--max-mean-cte", type=float, default=1.0)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    run_validation(model_path, args.num_episodes, args.timeout_sec)

    output_csv = latest_matching_file(
        [
            "stl_monitor/logs/chauffeur_all/output_*.csv",
            "stl_monitor/logs/**/chauffeur_all/output_*.csv",
            "STL_Monitor_for_ADS_Behavior/logs/chauffeur_all/output_*.csv",
            "STL_Monitor_for_ADS_Behavior/logs/**/chauffeur_all/output_*.csv",
        ]
    )
    if output_csv is None:
        raise FileNotFoundError(
            "No simulation output CSV found under stl_monitor/logs/chauffeur_all or STL_Monitor_for_ADS_Behavior/logs/chauffeur_all"
        )

    metrics = compute_gate_metrics(output_csv)

    pass_success = metrics["success_rate"] >= args.min_success_rate
    pass_cte = metrics["mean_max_cte"] <= args.max_mean_cte if not pd.isna(metrics["mean_max_cte"]) else False
    passed = pass_success and pass_cte

    print("Validation summary")
    print(f"- output_csv: {output_csv}")
    print(f"- rows: {metrics['rows']}")
    print(f"- success_rate: {metrics['success_rate']:.4f} (threshold >= {args.min_success_rate:.4f})")
    print(f"- mean_max_cte: {metrics['mean_max_cte']:.4f} (threshold <= {args.max_mean_cte:.4f})")
    print(f"- result: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
