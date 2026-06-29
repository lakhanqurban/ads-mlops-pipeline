import argparse
import hashlib
from pathlib import Path


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Register trained .h5 model to MLflow Model Registry")
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--model-name", default="asdve-steering-model", type=str)
    parser.add_argument("--run-id", default="", type=str, help="Optional MLflow run_id to register from runs:/")
    parser.add_argument("--artifact-path", default="model", type=str)
    parser.add_argument("--stage", default="Staging", type=str)
    parser.add_argument("--mlflow-tracking-uri", default="", type=str)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        raise RuntimeError(
            "MLflow is required for model registration. Install with: pip install mlflow"
        ) from exc

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    client = MlflowClient()

    model_hash = compute_sha256(model_path)

    if args.run_id:
        model_uri = f"runs:/{args.run_id}/{args.artifact_path}"
    else:
        model_uri = str(model_path.resolve())

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    version = result.version

    client.set_model_version_tag(args.model_name, version, "sha256", model_hash)
    client.set_model_version_tag(args.model_name, version, "source_model_path", str(model_path.resolve()))

    if args.stage:
        client.transition_model_version_stage(
            name=args.model_name,
            version=version,
            stage=args.stage,
            archive_existing_versions=False,
        )

    print(f"Registered {args.model_name} v{version}")
    print(f"sha256={model_hash}")
    print(f"stage={args.stage}")


if __name__ == "__main__":
    main()
