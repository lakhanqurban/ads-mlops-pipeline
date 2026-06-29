import argparse
from pathlib import Path

import numpy as np


def _load_required_arrays(npz_path):
    archive = np.load(npz_path, allow_pickle=True)

    if "observations" not in archive or "actions" not in archive:
        raise KeyError(
            f"{npz_path} must contain 'observations' and 'actions' arrays. "
            f"Found keys: {list(archive.keys())}"
        )

    observations = archive["observations"]
    actions = archive["actions"]

    if len(observations) != len(actions):
        raise ValueError(
            f"Length mismatch in {npz_path}: observations={len(observations)} actions={len(actions)}"
        )

    optional = {}
    for key in ("road_ids", "episode_indices", "timestamps"):
        if key in archive:
            values = archive[key]
            if len(values) == len(observations):
                optional[key] = values

    return observations, actions, optional


def _sample_indices(total, max_samples, seed):
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False))


def main():
    parser = argparse.ArgumentParser(description="Merge two NPZ datasets with optional subsampling")
    parser.add_argument("--base-npz", required=True, help="Existing/base NPZ path")
    parser.add_argument("--new-npz", required=True, help="Newly collected NPZ path")
    parser.add_argument("--output-npz", required=True, help="Output merged NPZ path")
    parser.add_argument(
        "--max-base-samples",
        type=int,
        default=0,
        help="Maximum samples to keep from base NPZ (0 means keep all)",
    )
    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help="Maximum samples to keep from new NPZ (0 means keep all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    base_path = Path(args.base_npz)
    new_path = Path(args.new_npz)
    out_path = Path(args.output_npz)

    if not base_path.exists():
        raise FileNotFoundError(f"Base NPZ not found: {base_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New NPZ not found: {new_path}")

    base_obs, base_act, base_opt = _load_required_arrays(base_path)
    new_obs, new_act, new_opt = _load_required_arrays(new_path)

    base_idx = _sample_indices(len(base_obs), args.max_base_samples, args.seed)
    new_idx = _sample_indices(len(new_obs), args.max_new_samples, args.seed + 1)

    merged_obs = np.concatenate([base_obs[base_idx], new_obs[new_idx]], axis=0)
    merged_act = np.concatenate([base_act[base_idx], new_act[new_idx]], axis=0)

    merged_payload = {
        "observations": merged_obs,
        "actions": merged_act,
    }

    # Keep optional arrays only if present in both datasets.
    common_optional = set(base_opt.keys()).intersection(set(new_opt.keys()))
    for key in sorted(common_optional):
        merged_payload[key] = np.concatenate([base_opt[key][base_idx], new_opt[key][new_idx]], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **merged_payload)

    print("Merged NPZ created")
    print(f"- output: {out_path}")
    print(f"- base kept: {len(base_idx)} / {len(base_obs)}")
    print(f"- new kept: {len(new_idx)} / {len(new_obs)}")
    print(f"- total samples: {len(merged_obs)}")
    print(f"- observation shape: {merged_obs.shape}")
    print(f"- action shape: {merged_act.shape}")


if __name__ == "__main__":
    main()
