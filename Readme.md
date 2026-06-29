# 🚗 Autonomous Driving Model Training & Validation Pipeline

An end-to-end **MLOps pipeline** for training, validating, registering, and iteratively improving autonomous driving steering models using the Udacity Self-Driving Car Simulator.

The pipeline implements a **self-improving loop**: each trained model is used to collect new driving data, which trains the next model version — creating a continuous improvement cycle grounded in pass/fail simulator validation gates.

> ⚠️ **Scope Note:** Validation is performed inside the Udacity Self-Driving Car Simulator — a controlled, single-track environment. Results reflect performance within this simulated context and do not constitute real-world safety guarantees.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Iterative Retraining Loop](#iterative-retraining-loop)
- [Pipeline Workflow](#pipeline-workflow)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Pipeline Stages](#pipeline-stages)
- [Running the Pipeline](#running-the-pipeline)
- [All Parameters](#all-parameters)
- [Pipeline Outputs](#pipeline-outputs)
- [Model Versioning & Results](#model-versioning--results)
- [ASDVE Integration](#asdve-integration)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Overview

Traditional deep learning projects treat model training as a one-time task. This project implements a full **MLOps lifecycle** where every trained model feeds back into the next iteration of data collection and retraining.

The pipeline automates:

- Dataset preparation and merging
- NVIDIA DAVE-2 style steering model training
- Offline performance evaluation
- Threshold-based simulator validation (pass/fail gate)
- Optional MLflow experiment tracking and model registration
- Iterative model versioning (`v1 → v2 → ...`)

Each component is modular — stages can be run individually or orchestrated end-to-end via `run_pipeline.py`.

---

## Project Objectives

- 📂 Build an incrementally expanding training dataset from simulator-collected `.npz` files
- 🧠 Train and retrain steering prediction models across versioned iterations
- 📊 Evaluate models with objective pass/fail thresholds (success rate, CTE)
- 🔁 Use each validated model to collect the next generation of training data
- 📦 Track experiments and register models via MLflow
- ✅ Integrate validated models into the ASDVE safety verification framework

---

## Iterative Retraining Loop

This is the core concept of the pipeline. Rather than training once, each model version drives the simulator to generate new data, which is merged with prior data to train the next version.

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
         dave2_base (pretrained)                         │
                    │                                     │
                    ▼                                     │
         Collect NPZ data via simulator                  │
                    │                                     │
                    ▼                                     │
         Merge with existing dataset                     │
                    │                                     │
                    ▼                                     │
         Train dave2_retrain_v1                          │
                    │                                     │
                    ▼                                     │
      ┌─── Simulator Validation (pass/fail) ───┐         │
      │                                        │         │
    PASS                                      FAIL       │
      │                                        │         │
      ▼                                        ▼         │
  Register v1                         Investigate &      │
  Use v1 to collect new data          retrain            │
      │                                                  │
      ▼                                                  │
  Train dave2_retrain_v2 ────────────────────────────────┘
```

**Pass criteria (configurable):**
- Minimum success rate: `0.70` (70% of episodes completed)
- Maximum mean Cross-Track Error: `1.0`

> ⚠️ **Data Quality Note:** A model that passes at 70% success rate is still failing 30% of episodes. NPZ data collected during failed episodes should be reviewed or filtered before merging into the next training set, to avoid propagating poor driving behavior into future versions.

---

## Pipeline Workflow

Each run of `run_pipeline.py` executes the following sequence, with optional stages controlled by flags:

```
         Raw Driving Data (.npz)
                  │
                  ▼
       Dataset Preparation & Merging
                  │
                  ▼
       Steering Model Training
       (NVIDIA DAVE-2 architecture)
                  │
                  ▼
       Offline Evaluation
       (Loss, MAE on held-out split)
                  │
                  ▼
       [Optional] MLflow Experiment Logging  ◄── --with-mlflow
                  │
                  ▼
       [Optional] Model Registration         ◄── --register
       (MLflow Model Registry)
                  │
                  ▼
       [Optional] Simulator Validation       ◄── --validate
       (Pass/Fail gate on success rate & CTE)
                  │
                  ▼
       Verified Model → ASDVE Framework
```

All three optional stages are **disabled by default** and must be explicitly enabled.

---

## Repository Structure

```
training_dataset/
│
├── merged/
│   ├── merged_dataset.npz
│   └── ...
│
├── models/
│   ├── dave2_retrain_<dataset_name>.h5
│   ├── history_<dataset_name>.json
│   └── ...
│
├── scripts/
│   ├── merge_npz.py          # Dataset merging
│   ├── train_model.py        # Model training
│   ├── validate_model.py     # Simulator validation
│   ├── register_model.py     # MLflow model registration
│   └── run_pipeline.py       # Full pipeline orchestrator
│
└── ...
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- tqdm
- MLflow *(optional, required for `--with-mlflow` and `--register`)*

Install required packages:

```bash
pip install tensorflow numpy pandas scikit-learn tqdm
```

For MLflow support:

```bash
pip install mlflow
```

---

## Pipeline Stages

### 1. Dataset Preparation

Training data is stored as compressed NumPy (`.npz`) files containing synchronized driving data collected from the simulator.

Each file typically contains:

- Front camera images
- Steering angle commands
- Optional episode metadata

### 2. Dataset Merging

New datasets collected by the latest model version are merged with the existing baseline dataset to grow the training set incrementally.

This stage supports:

- Merging multiple `.npz` files
- Dataset balancing
- Random subsampling
- Controlled dataset growth

```
  Baseline Dataset (collected by v_base)
            +
  New Dataset (collected by dave2_retrain_v1)
            │
            ▼
      Merged Dataset → train dave2_retrain_v2
```

### 3. Model Training

Trains an NVIDIA DAVE-2 end-to-end steering prediction network.

Training includes:

- Data preprocessing
- Train/validation split (default: 80/20)
- Batch generation
- Adam optimizer
- Per-epoch validation monitoring

Artifacts saved:

- Trained model: `dave2_retrain_<dataset_name>.h5`
- Training history: `history_<dataset_name>.json`
- Final evaluation metrics

### 4. Offline Evaluation

The held-out validation split is used to compute:

- Training Loss
- Validation Loss
- Mean Absolute Error (MAE)

These metrics indicate convergence and in-distribution generalization. They are necessary but **not sufficient** — a model can have low MAE and still fail in the simulator. Simulator validation is the authoritative gate.

### 5. MLflow Experiment Tracking *(Optional)*

When `--with-mlflow` is passed, all training parameters and metrics are logged to an MLflow tracking server for experiment comparison across model versions.

### 6. Model Registration *(Optional)*

When `--register` is passed, the trained model is pushed to the MLflow Model Registry under a configurable name and stage (default: `Staging`).

### 7. Simulator Validation *(Optional — Pass/Fail Gate)*

When `--validate` is passed, the trained model is loaded into the Udacity simulator and evaluated across multiple episodes.

**This is a hard gate.** The pipeline will exit with a non-zero return code if the model does not meet the configured thresholds:

| Metric | Default Threshold | Flag |
|---|---|---|
| Success Rate | ≥ 70% | `--min-success-rate` |
| Mean Cross-Track Error | ≤ 1.0 | `--max-mean-cte` |

Collected statistics per run:

- Success Rate (episodes completed without failure)
- Mean Cross-Track Error (CTE)
- Episode Completion percentage
- Driving stability indicators

### 8. ASDVE Verification *(External Framework)*

Models that pass simulator validation can be submitted to the **ASDVE (Autonomous Systems Data Verification & Evaluation)** framework for deeper safety analysis using Signal Temporal Logic (STL) runtime monitoring. This stage is external to this pipeline and is documented separately.

---

## Running the Pipeline

### Merge datasets

```bash
python training_dataset/scripts/merge_npz.py \
    --base-npz baseline_dataset.npz \
    --new-npz collected_dataset.npz \
    --output-npz merged_dataset.npz
```

### Train only

```bash
python training_dataset/scripts/run_pipeline.py \
    --data-path training_dataset/merged/merged_dataset.npz \
    --epochs 20 \
    --batch-size 32
```

### Train + validate (pass/fail gate)

```bash
python training_dataset/scripts/run_pipeline.py \
    --data-path training_dataset/merged/merged_dataset.npz \
    --epochs 20 \
    --batch-size 32 \
    --validate \
    --num-episodes 10 \
    --min-success-rate 0.70 \
    --max-mean-cte 1.0
```

### Train + MLflow tracking + register + validate

```bash
python training_dataset/scripts/run_pipeline.py \
    --data-path training_dataset/merged/merged_dataset.npz \
    --epochs 20 \
    --batch-size 32 \
    --with-mlflow \
    --mlflow-experiment asdve-training \
    --mlflow-tracking-uri http://localhost:5000 \
    --register \
    --model-name asdve-steering-model \
    --stage Staging \
    --validate \
    --num-episodes 10 \
    --min-success-rate 0.70 \
    --max-mean-cte 1.0
```

---

## All Parameters

### Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `--data-path` | Path to training `.npz` dataset | Required |
| `--output-dir` | Directory to save model artifacts | `training_dataset/models` |
| `--epochs` | Number of training epochs | `20` |
| `--batch-size` | Mini-batch size | `32` |
| `--learning-rate` | Adam optimizer learning rate | `1e-4` |
| `--train-split` | Train/validation split ratio | `0.8` |
| `--seed` | Random seed for reproducibility | `42` |

### MLflow Parameters *(optional)*

| Parameter | Description | Default |
|---|---|---|
| `--with-mlflow` | Enable MLflow experiment logging | `False` |
| `--mlflow-experiment` | MLflow experiment name | `asdve-training` |
| `--mlflow-tracking-uri` | MLflow tracking server URI | *(local)* |

### Model Registration Parameters *(optional)*

| Parameter | Description | Default |
|---|---|---|
| `--register` | Register model to MLflow Model Registry | `False` |
| `--model-name` | Registered model name | `asdve-steering-model` |
| `--stage` | Registry stage | `Staging` |

### Validation Parameters *(optional)*

| Parameter | Description | Default |
|---|---|---|
| `--validate` | Run simulator validation after training | `False` |
| `--num-episodes` | Number of simulator episodes to run | `10` |
| `--timeout-sec` | Max seconds before validation timeout | `7200` |
| `--min-success-rate` | Minimum required success rate (pass threshold) | `0.70` |
| `--max-mean-cte` | Maximum allowed mean Cross-Track Error (pass threshold) | `1.0` |

---

## Pipeline Outputs

After a successful run, the following artifacts are generated:

```
training_dataset/models/
├── dave2_retrain_<dataset_name>.h5    # Trained Keras model
├── history_<dataset_name>.json        # Per-epoch loss and MAE
└── metrics                            # Final evaluation summary
```

Model naming is automatically derived from the input dataset filename, enabling traceability between training data and model version.

---

## Model Versioning & Results

The project follows an explicit versioning scheme tied to the iterative retraining loop:

| Version | Training Data Source | Status |
|---|---|---|
| `dave2_base` | Original simulator autopilot data | Baseline |
| `dave2_retrain_v1` | Data collected by `dave2_base` | ✅ Trained — results under review |
| `dave2_retrain_v2` | Data collected by `dave2_retrain_v1` | 🔄 Planned |

Comparative pass/fail results between `dave2_base` and `dave2_retrain_v1` will be published here upon completion of evaluation.

---

## ASDVE Integration

This pipeline is one component of the broader **ASDVE (Autonomous Systems Data Verification & Evaluation)** framework. Validated models are submitted to ASDVE for:

- Signal Temporal Logic (STL) runtime safety monitoring
- Robustness evaluation under challenging scenarios
- Prioritized testing on edge-case road conditions
- Formal runtime verification

ASDVE documentation and integration details are maintained separately.

---

## Known Limitations

- **Simulator scope:** The Udacity simulator is a single-track, obstacle-free environment. Validation results do not generalize to real-world driving or more complex simulated environments.
- **Metric sufficiency:** Low MAE does not guarantee simulator success. The simulator validation gate is the authoritative evaluation.
- **Data loop quality:** Data collected by a model that passes at 70% success rate includes failed episodes. Merging unfiltered data into the next training set may propagate poor driving behavior. Episode-level filtering before merging is recommended.
- **Distribution narrowing:** As model quality improves, collected data becomes less diverse. Future iterations should consider deliberate edge-case data injection to maintain training distribution coverage.

---

## Future Work

- Automated episode-level NPZ filtering based on CTE and success criteria before dataset merging
- Drift detection between model versions
- CI/CD integration for automated retraining triggers
- Expanded simulator scenarios for more diverse validation
- Formal ASDVE integration documentation

---

## Summary

This pipeline implements a complete, iterative MLOps workflow for autonomous driving:

- ✅ Versioned model training (`v1 → v2 → ...`)
- ✅ Pass/fail simulator validation gate
- ✅ MLflow experiment tracking and model registration
- ✅ Modular, individually executable pipeline stages
- ✅ Iterative self-improving data collection loop
- ✅ ASDVE safety verification integration

Each model version drives the simulator to generate training data for the next — turning model improvement into a reproducible, traceable, and continuously validated cycle.
