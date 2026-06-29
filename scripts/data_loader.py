import math
from pathlib import Path

import cv2
import numpy as np
from keras.utils import Sequence


class NPZSteeringDataset:
    """Load NPZ telemetry datasets and expose steering-training arrays."""

    def __init__(self, npz_path):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        self._archive = np.load(self.npz_path, allow_pickle=True)
        self.observations = self._require_key("observations")
        self.actions = self._require_key("actions")

        self.steering = self._extract_steering(self.actions)
        if len(self.observations) != len(self.steering):
            raise ValueError(
                "Observations/actions length mismatch: "
                f"{len(self.observations)} vs {len(self.steering)}"
            )

    def _require_key(self, key):
        if key not in self._archive:
            available = ", ".join(sorted(self._archive.files))
            raise KeyError(f"Missing key '{key}' in {self.npz_path}. Available: {available}")
        return self._archive[key]

    @staticmethod
    def _extract_steering(actions):
        # Common shapes:
        # - (N, 2): [steering, throttle]
        # - (N, 1, 2): [ [steering, throttle] ]
        # - (N, 1): [steering]
        # - (N,): [steering]
        arr = np.asarray(actions)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        if arr.ndim == 2:
            if arr.shape[1] >= 1:
                return arr[:, 0].astype(np.float32)
        if arr.ndim == 3:
            # Prefer first channel first value, fallback to squeeze rules.
            if arr.shape[1] >= 1 and arr.shape[2] >= 1:
                return arr[:, 0, 0].astype(np.float32)
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 1:
            raise ValueError(f"Unsupported actions shape: {arr.shape}")
        return squeezed.astype(np.float32)

    def __len__(self):
        return len(self.observations)


class SteeringBatchGenerator(Sequence):
    """Keras Sequence that preprocesses camera frames on demand."""

    def __init__(
        self,
        observations,
        steering,
        indices,
        batch_size=32,
        shuffle=True,
        target_height=160,
        target_width=320,
    ):
        self.observations = observations
        self.steering = steering
        self.indices = np.array(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self._order = np.arange(len(self.indices))
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        order_slice = self._order[start:end]
        batch_indices = self.indices[order_slice]

        x_batch = []
        y_batch = []
        for idx in batch_indices:
            frame = self.observations[idx]
            steering = self.steering[idx]
            x_batch.append(self._preprocess_frame(frame))
            y_batch.append(np.float32(steering))

        return np.asarray(x_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._order)

    def _preprocess_frame(self, frame):
        # Current repo models use YUV input with shape 160x320.
        # Some datasets (e.g. fake) have reduced height, so crop only when safe.
        img = np.asarray(frame)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected frame shape: {img.shape}")

        if img.shape[0] > 100:
            # Remove sky and hood similar to existing project preprocessing.
            img = img[60:-25, :, :]

        img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img


def split_indices(num_samples, train_split=0.8, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    cut = int(num_samples * float(train_split))
    return indices[:cut], indices[cut:]
