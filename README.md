# NASA C-MAPSS Predictive Maintenance — RUL System

An end-to-end Remaining Useful Life (RUL) prediction system built around NASA's C-MAPSS turbofan engine simulation data.

This project started as an attempt to build a deep-learning model for RUL prediction. It gradually became a larger engineering problem: preserving temporal structure, preventing engine-level leakage, defining a consistent target, validating the data pipeline, and putting a statistical safety boundary around model inference.

The current baseline is **EXP-001**, a deliberately simple LSTM model. The project is being developed as both a technical portfolio project and an engineering case study documenting the decisions, failures, and trade-offs behind the system.

> **Current milestone:** EXP-001 end-to-end baseline completed. Deeper error diagnosis and model improvements are intentionally treated as the next phase.

## Why this project?

RUL prediction is easy to describe and much harder to make trustworthy.

A turbofan engine does not produce independent tabular samples. Each observation belongs to an engine, a point in its lifecycle, and an operating regime. A model can still produce a numerical prediction when the data reaching it is malformed or unlike the data it learned from.

That led to the central engineering question behind this project:

> **How do you build a machine-learning system you can trust before you look at the final prediction?**

The project therefore focuses on the complete path from raw telemetry to inference rather than treating the neural network as the entire system.

## System overview

```text
NASA C-MAPSS FD001–FD004
          │
          ▼
     Mage ETL pipeline
          │
          ├── engine identity
          ├── temporal integrity
          ├── RUL construction
          └── condition-aware normalization
          │
          ▼
      Processed data
          │
          ▼
   Engine-level split
    (train / validation)
          │
          ▼
     30-cycle windows
          │
          ▼
       EXP-001
   1-layer LSTM baseline
          │
          ▼
       RUL prediction
          │
          ▼
   FastAPI inference service
          │
          ├── input validation
          └── statistical safety valve
          │
          ▼
       React / Vite UI
```

## EXP-001 baseline

The first model is intentionally simple so that later experiments have a meaningful control model.

The model receives a 30-cycle sequence containing 24 input features per timestep and predicts a single RUL value.

```text
Input
[batch, 30, 24]
    │
    ▼
LSTM
1 layer
hidden size = 128
    │
    ▼
Last timestep
[batch, 128]
    │
    ▼
Linear 128 → 64
    │
    ▼
ReLU
    │
    ▼
Linear 64 → 1
    │
    ▼
Predicted RUL
```

The baseline was trained with the RUL target capped at **130 cycles**. The same target convention was used for training, validation, and the FD004 evaluation.

## Data pipeline

The processed dataset combines FD001 through FD004 into a common representation while preserving engine identity and temporal ordering.

Important data contracts include engine-level integrity, consecutive cycle ordering, valid RUL values, non-negative RUL, zero RUL at an engine's final cycle, absence of duplicate engine-cycle pairs, valid sequence lengths, and the presence of all four CMAPSS subsets.

The model uses 30-cycle sliding windows. Window creation is performed per engine so that a sequence cannot cross from one engine into another.

### Operating conditions and normalization

The first three telemetry values are used to derive the operating-condition key used by the pipeline. The project identified six unique operating-condition groups in the processed data and uses condition-specific statistics for sensor normalization.

Sensor values are standardized using Z-score normalization relative to the relevant condition rather than treating all operating regimes as one statistical population.

## Train / validation split

The split is performed at the **engine level**, not at the window level.

This is important because neighbouring 30-cycle windows from the same engine overlap heavily. Randomly splitting those windows could place closely related histories on both sides of the validation boundary.

The current split contains:

| Split | Engines |
|---|---:|
| Training | 567 |
| Validation | 142 |
| Total | 709 |

## EXP-001 evaluation results

The current baseline was evaluated in `notebooks/testing.ipynb` against the FD004 test set (`test_FD004.txt`) and its corresponding RUL ground truth (`RUL_FD004.txt`). The notebook builds a normalized tensor from the final 30 cycles available for each test engine, loads the trained checkpoint, and compares the predicted RUL values with the ground-truth targets.

The evaluation covered **237 of 248 test engines**. **11 engines were skipped** because they contained fewer than 30 cycles required to form the model input window. The resulting evaluation tensor has shape **237 × 30 × 24**.

The primary evaluation uses the same **130-cycle target cap** used throughout EXP-001: 

| Metric | Result |
|---|---:|
| MAE | 10.73 cycles |
| RMSE | 14.83 cycles |
| R² | 0.887 |
| Mean signed error | 0.14 cycles |
| Median absolute error | 7.54 cycles |
| 90th percentile absolute error | 23.15 cycles |
| NASA score | 1059.42 |

An additional **diagnostic-only** evaluation was performed against the uncapped FD004 targets. It produced an MAE of **17.16 cycles** and an RMSE of **24.16 cycles**. These values are not the primary EXP-001 result because the model was trained and evaluated under the 130-cycle capped target convention.

## Safety valve

The model is not treated as the only authority over whether an input is valid.

A Mahalanobis-distance based statistical gate is used as an inference-time safety mechanism. The reference statistics are condition-specific so that anomalous observations are compared against the distribution associated with their operating regime.

The intended inference path is:

```text
incoming 30-cycle window
          │
          ▼
preprocessing / normalization
          │
          ▼
condition-specific statistical check
          │
       ┌──┴──┐
       │     │
     normal  OOD
       │     │
       ▼     ▼
     LSTM   reject
       │
       ▼
     RUL
```

The safety-valve implementation is still considered an area for further work. In particular, the aggregation of statistical distances across a 30-step window and the calibration/justification of the rejection threshold need deeper validation.

## Serving

The project includes a FastAPI inference service and a React/Vite client.

The service validates the incoming window, applies the same stored preprocessing statistics used during model development, runs the safety check, and invokes the PyTorch model only for accepted requests.

Docker is used to package the inference service.

The current service is an engineering implementation for the portfolio project; production-scale load testing and further deployment hardening are part of future work.

## Testing

The project uses `pytest` to test the data and ML contracts rather than relying only on the final model metric.

The test suite covers areas including ETL integrity, engine boundaries, temporal cycles, RUL validity, window construction, target alignment, normalization behaviour, model output shape, training-step behaviour, and inference properties.

The test suite was intentionally expanded after an early window-boundary test failed for the wrong reason. That incident changed the testing philosophy from checking arbitrary expected values to checking properties that must remain true throughout the pipeline.

## Repository structure

```text
.
├── .vscode/
├── cmapss-ui/
├── docs/
│   ├── experiments/
│   └── journal/
├── notebooks/
│   ├── inference.ipynb
│   └── testing.ipynb
├── src/
│   ├── data/
│   ├── serve/
│   ├── training/
│   └── __init__.py
├── tests/
├── .gitignore
└── requirements_prod.txt
```

### Directory roles

`src/data/` contains the data preparation, normalization, dataset, and loading logic.

`src/training/` contains model training logic.

`src/serve/` contains the FastAPI inference service and inference-time safety logic.

`tests/` contains the automated test suite for the pipeline and model contracts.

`docs/experiments/` records controlled ML experiments such as EXP-001.

`docs/journal/` contains the chronological engineering journal behind the project and the eventual portfolio article.

`notebooks/` contains exploratory and verification notebooks rather than the primary production path.

`cmapss-ui/` contains the React/Vite frontend used to interact with the inference API.

## Running the project

### Python environment

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements_prod.txt
```

### Run the tests

```bash
pytest -q
```

### Run the FastAPI service

From the repository root:

```bash
uvicorn src.serve.app:app --reload
```

The exact runtime configuration may depend on the local MLflow/model artifact setup.

### Run the frontend

```bash
cd cmapss-ui
npm install
npm run dev
```

The frontend expects the API endpoint configured in its client-side API configuration.

## Current evaluation

The current baseline has been evaluated on the FD004 test data using the corresponding RUL ground truth.

The latest established evaluation result is approximately:

| Metric | Result |
|---|---:|
| R² | 0.889 |
| MAE | 9.97 cycles |

These numbers are useful as a baseline, not as evidence that the model is finished. The next investigation is to determine where the remaining errors come from and whether they are associated with RUL range, operating condition, or specific engines.

## Current limitations and future work

EXP-001 deliberately stops short of being the final model.

The next phase is focused on understanding the baseline before changing its architecture. Planned work includes deeper error analysis by RUL range and operating condition, investigation of high-error engines, individual-engine trajectory analysis, controlled model experiments, explainability, and more rigorous calibration of the OOD safety mechanism.

The goal is not to add complexity simply because a more complicated architecture is available. Each future experiment should be motivated by an observed weakness in EXP-001.

## Engineering journal

The project is being documented as an engineering journal rather than only as a final implementation report.

The journal records the decisions and failures that shaped the system: the shift from ETL-as-cleaning to ETL-as-contract, engine-level splitting, the 130-cycle target cap, the window-boundary testing mistake, the suspiciously fast training run, the first inference surprises, the safety-valve design, and the FD004 evaluation.

This documentation is intended to accompany the final portfolio article and show not just what was built, but how the engineering decisions evolved.

## Project status

**EXP-001 — End-to-end baseline:** Completed

**Data / ETL / Dataset / DataLoader:** Baseline implementation established

**FastAPI + Docker + UI:** Working inference path established

**Safety valve:** Implemented, further statistical validation pending

**FD004 evaluation:** Completed for the current baseline

**EXP-002 model improvements:** Not started / future work

**XAI:** Future work

**Production-scale deployment and monitoring:** Future work

---


