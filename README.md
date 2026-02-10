# CGM Imputation Stress-Testing Framework

**Stationarity Bias · Scenarios A / B / C · Slurm + Optuna + PyPOTS**

> **Paper:** *The Stationarity Bias: Stratified Stress-Testing for Time-Series Imputation in Regulated Dynamical Systems*
> Amirreza Dolatpour Fathkouhi, Alireza Namazi, Heman Shakeri
> KDD 2026, Jeju, Korea
> Code: [github.com/Shakeri-Lab/Glucose-Imputation](https://github.com/Shakeri-Lab/Glucose-Imputation)

---

## About This Document

This document serves as the **official, paper-faithful specification** of the experimental protocol described in the accompanying publication. It is intended for reviewers, collaborators, and future maintainers who need to understand, reproduce, or extend the reported experiments.

Every configuration, parameter, and pipeline step documented here maps directly to the paper — there are no undocumented behaviors, implicit assumptions, or deviations from the published methodology.

All models are implemented using **PyPOTS**, a unified library for time-series imputation, and hyperparameter optimization is conducted with **Optuna**.

---

## 1. Core Contributions

The paper formalizes three findings with broad implications beyond CGM:

1. **The Stationarity Bias.** Uniform random masking on signals with a dominant stationary regime systematically inflates the apparent performance of simple imputation methods.
2. **The RMSE Mirage.** Low pointwise error (RMSE) masks the destruction of signal morphology during critical transients. DTW reveals divergence up to 3× larger than RMSE suggests.
3. **Regime-Conditional Model Selection.** Linear interpolation is optimal during stationary intervals; deep learning (SAITS) is essential during transients for morphological fidelity, clinical accuracy, and safety-relevant bias.

---

## 2. Experimental Regimes (Strict Separation)

The paper defines **two non-overlapping regimes** that serve different purposes and must never be mixed.

| Regime             | Purpose                        | Used When                   |
| ------------------ | ------------------------------ | --------------------------- |
| **Mixed**          | Robust representation learning | Training only               |
| **Scenario A/B/C** | Stress-test evaluation         | Validation and testing only |

---

## 3. Mixed Training Regime

Mixed is a **training-only** regime. It exposes models to ecologically valid missingness patterns derived from clinical trial data (DCLP3, DCLP5) and optimizes a real-missingness loss.

> **Note:** Mixed is never used for evaluation.

**Configuration:**

```yaml
type: Mixed
missing_enabled: True

miss_config:
  is_real_mask: True
```

| Aspect         | Detail                        |
| -------------- | ----------------------------- |
| Usage          | Training only                 |
| Missingness    | Real (trial-derived)          |
| Loss           | Real-missingness loss (paper) |
| Scenario A/B/C | Not applied                   |

---

## 4. Scenario-Based Evaluation (A / B / C)

Evaluation is performed under **explicit stress-test scenarios** that isolate distinct dynamical regimes. Each scenario targets a specific clinical context where imputation quality has direct safety implications.

### Scenario Definitions

| Scenario | Regime          | What Is Masked                             | Masking Strategy                               |
| -------- | --------------- | ------------------------------------------ | ---------------------------------------------- |
| **A**    | Stationary      | Homeostatic gaps (euglycemic stability)    | 10%, 20%, 30% of sequence length               |
| **B**    | Transient       | Post-prandial peaks                        | 3.5–4 hr windows centered on 1, 2, or 3 peaks |
| **C**    | Safety-critical | Hypoglycemic events during TCR activation  | 1 hr window centered on glucose < 70 mg/dL     |

**Configuration:**

```yaml
missing_enabled: True

miss_config:
  type: 'A' | 'B' | 'C'
  is_real_mask: False
```

---

### 4.1 Protocol A — Homeostatic State Evaluation

Evaluates reconstruction fidelity during stable intervals. Candidate 30-minute windows must satisfy **all five criteria simultaneously**:

1. **Euglycemia:** All glucose values in 70–140 mg/dL.
2. **Gradient stability:** ≥ 85% of time points satisfy |∇gₜ| < 0.6 mg/dL/min.
3. **No exogenous inputs:** No meal or bolus events within the window (Σcₜ = 0).
4. **Washout period:** No meal or bolus events in the 60 minutes preceding the window.
5. **Low variability:** Glucose range (gₘₐₓ − gₘᵢₙ) < 25 mg/dL.

Masking is allocated into full 30-minute segments, with any residual applied as a partial segment. Experiments are run at masking ratios of **10%, 20%, and 30%** of total sequence length.

### 4.2 Protocol B — Post-Prandial Peak Evaluation

Evaluates reconstruction of critical glycemic excursions:

1. Meals separated by < 1 hour are aggregated into single events.
2. A 4-hour post-prandial window is scanned to identify each peak.
3. A masking window of **random duration 3.5–4 hours** is centered on the identified peak.
4. Masking windows must not cover pre-meal time steps or exceed total daily duration.

Experiments vary the **number of masked peaks: 1, 2, or 3**.

### 4.3 Protocol C — Temporal Control Reset for Hypoglycemia

Evaluates imputation during insulin pump adjustments precipitated by impending hypoglycemia:

- Uses the **TCR-Simulation dataset** exclusively (no models are trained on this data).
- A **1-hour window** is masked, centered on the hypoglycemic event (glucose < 70 mg/dL) observed during TCR activation.
- TCR activation: 2.5 hours post-meal, basal rate reduced to 5% of nominal for 4 hours.

---

## 5. Datasets

All datasets use CGM measurements, timestamps, meal, basal, and bolus insulin at **5-minute intervals**. Data is split at the patient level (70% train / 10% validation / 20% test) to assess generalization on unseen subjects.

### Data Acquisition and Processing

The DCLP3, DCLP5, and PEDAP clinical datasets can be downloaded from the Jaeb Center for Health Research:
**https://public.jaeb.org/datasets/diabetes**

| Dataset        | Processing Pipeline                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| DCLP3, DCLP5   | Processed using the [DIAX repository](https://github.com/Center-for-Diabetes-Technology/DIAX)               |
| PEDAP          | Processed using the `process_pedap.ipynb` notebook included in this repository                               |

The UVA/Padova Simulation and TCR-Simulation datasets will be available for download soon.

All optimized hyperparameters and trained model checkpoints are available at:
**https://myuva-my.sharepoint.com/:f:/g/personal/aww9gh_virginia_edu/IgCoABI3SmJzQrGXDZKyl0G1AZuLXYe2nO05kiXvYIguzUQ?e=Of08bF**

After downloading, simply set `saving_path` in your configuration to the downloaded directory and run the evaluation pipeline directly.

### 5.1 PEDAP (Pediatric Artificial Pancreas)

102 children aged 2–6 years, 13-week clinical trial.

| Variant             | Episode Split  | Missing CGM Handling  | Usage         |
| ------------------- | -------------- | --------------------- | ------------- |
| **Raw PEDAP**       | Gaps > 240 min | Retained as missing   | Training only |
| **Processed PEDAP** | Gaps > 30 min  | Linearly interpolated | Testing       |

Auxiliary variables (meal, bolus, basal) are zero-filled in both variants. Models trained on Raw PEDAP are tested on Processed PEDAP.

### 5.2 UVA/Padova Simulation

100 adults, 3 meals/day (Breakfast 06:00–11:00, Lunch 11:00–13:00, Dinner 18:00–20:00). Meal times have random variability (σ = 20 min), meal sizes range 0.84–1.44 g/kg with 15% size variability. No exercise. Hypoglycemia treatment triggered below 70 mg/dL.

### 5.3 TCR-Simulation

Follows the UVA/Padova protocol with temporal variability removed. Induces hypoglycemia via meal size overestimation (+20% to +30%). One meal per day triggers TCR activation 2.5 hours post-meal for 4 hours (basal rate × 0.05). **Used exclusively for Protocol C evaluation** — 20% of patients designated as the test set.

---

## 6. Models Evaluated

### Deep Learning

| Model     | Category                        | Reference             |
| --------- | ------------------------------- | --------------------- |
| SAITS     | Dual-stage Transformer          | Du et al., 2023       |
| GPT4TS    | Frozen GPT-2 adaptation         | Zhou et al., 2023     |
| FreTS     | Frequency-domain MLP            | Yi et al., 2023       |
| TSLANet   | Spectral + convolution blocks   | Eldele et al., 2024   |
| TimeMixer | Multi-scale decomposable mixing | Wang et al., 2024     |
| SCINet    | Recursive downsampling          | Liu et al., 2022      |
| TEFN      | Evidence theory + fuzzy masses  | Zhan et al., 2025     |
| TOTEM     | VQ-VAE tokenization             | Talukder et al., 2024 |

### Baselines

Mean, Median, LOCF (Last Observation Carried Forward), and Linear Interpolation (Lerp).

---

## 7. Metrics

| Metric    | What It Measures                      | Why It Matters                                     |
| --------- | ------------------------------------- | -------------------------------------------------- |
| **RMSE**  | Pointwise reconstruction error        | Standard benchmark; susceptible to the RMSE Mirage |
| **MARD**  | Mean absolute relative difference (%) | Clinical gold standard for CGM accuracy            |
| **DTW**   | Temporal/morphological similarity     | Captures shape destruction invisible to RMSE       |
| **Bias**  | Systematic over-/under-estimation     | Directly affects closed-loop controller calibration|
| **EmpSE** | Empirical standard error              | Variability of residuals                           |

---

## 8. Real-Missingness Modeling

Mixed training relies on trial-derived missingness statistics estimated from DCLP3 and DCLP5. The stochastic model has three components:

1. **Hourly onset probability** P_start(h): fraction of valid monitoring days on which a gap begins at hour h.
2. **Single-point dropout classification:** probability π_short that a gap is a transient 5-minute dropout.
3. **Sustained gap duration model:** mixture of Exponential + Gaussian (centered near 120-min sensor warm-up) + uniform component, estimated separately for Day (h ∈ [6, 24)) and Night (h ∈ [0, 6)) regimes. Maximum duration capped at 240 minutes.

```yaml
miss_config:
  csv_list:
    - RawData/dclp3_cgm_plus_features.csv
    - RawData/dclp5_cgm_plus_features.csv
  valid_threshold: 0.5
```

| Parameter         | Meaning                                                             |
| ----------------- | ------------------------------------------------------------------- |
| `csv_list`        | Clinical-trial feature files used to estimate real-missingness loss |
| `valid_threshold` | Minimum observed/expected ratio (O_obs / 288) for valid days       |

---

## 9. Scenario Parameters

```yaml
miss_config:
  protocol_mask_ratio: 0.3
  num_meal_hide: 3
  peak_prominence_mgdl: 40
  peak_distance_minutes: 120
```

| Parameter               | Scenario | Meaning                                     |
| ----------------------- | -------- | ------------------------------------------- |
| `protocol_mask_ratio`   | A        | Fraction of sequence masked (0.1, 0.2, 0.3) |
| `num_meal_hide`         | B        | Number of post-prandial peaks masked (1–3)  |
| `peak_prominence_mgdl`  | B        | Minimum peak prominence (mg/dL)             |
| `peak_distance_minutes` | B        | Minimum separation between peaks (minutes)  |

---

## 10. Implementation Details

All models are implemented using the **PyPOTS** library. The following settings are **fixed across all experiments** and must not be changed.

```yaml
seq_len: 288              # 24 hours at 5-min intervals
stride_train: 57
stride_test: 128
batch_size: 32
max_day_length: 288
early_stopping_patience: 10
max_epochs: 100
random_seed: 7
```

---

## 11. Hyperparameter Optimization (Optuna)

Optimization is conducted using Optuna over **40 trials per model**. The best-performing configuration is selected for final evaluation.

```bash
--ParamRangeDir param_range.json
```

Defines model-specific hyperparameter ranges and ensures Optuna samples only valid, paper-approved configurations.

> **Note:** The number of trials is defined **in Slurm**, not in `config.yml`.

---

## 12. Reproduction Pipeline

### Step 1 — Optuna Training (Mixed)

```bash
NUM_TRIALS=40

python hyperparameter_engine.py \
  --model_name MODEL_NAME \
  --ParamRangeDir param_range.json \
  --config-path config.yml \
  --NTrials $NUM_TRIALS
```

### Step 2 — Train Best Model

```bash
python hyperparameter_engine.py \
  --model_name MODEL_NAME \
  --ParamRangeDir param_range.json \
  --config-path config.yml \
  --NTrials $NUM_TRIALS \
  --train_best
```

### Step 3 — Scenario Evaluation (A / B / C)

```bash
python hyperparameter_engine.py \
  --model_name MODEL_NAME \
  --config-path config.yml \
  --is_evaluate
```

Run this step **once per scenario**.

---

## 13. Slurm Execution Strategy

| Stage                 | Slurm Pattern             |
| --------------------- | ------------------------- |
| Training / Train-Best | One model per array task  |
| Evaluation            | Separate job per scenario |

This guarantees isolation and reproducibility.

---

## 14. Required Paths and Dataset Control

```yaml
data_path: /path/to/your/data/
saving_path: /path/to/output/root/
```

| Field         | Meaning                                 |
| ------------- | --------------------------------------- |
| `data_path`   | Dataset used for training or evaluation |
| `saving_path` | Root directory for all outputs          |

### PEDAP Switch

```yaml
is_pedap: True | False
```

When `True`, the pipeline uses the **Raw PEDAP** and **Processed PEDAP** datasets. When `False`, it uses the **Simulation** datasets (UVA/Padova, TCR-Simulation).

---

## 15. Output Structure

All outputs — training artifacts, Optuna logs, and scenario evaluation results — are stored under a single directory:

**https://myuva-my.sharepoint.com/:f:/g/personal/aww9gh_virginia_edu/IgCoABI3SmJzQrGXDZKyl0G1AZuLXYe2nO05kiXvYIguzUQ?e=Of08bF**

```
Mixed/
```

---

## 16. Reproducibility Guarantees

The following are held constant across all experiments:

- Fixed temporal context (seq_len = 288, stride = 128 for test)
- Fixed missingness definitions per scenario
- Fixed Optuna budget (40 trials per model)
- Scenario isolation (no cross-contamination between A/B/C)
- Identical compute resources
- Fixed random seed (7)

If you have any questions or issues running the code, don't hesitate to reach out: **aww9gh@virginia.edu**

---

## Reference

Amirreza Dolatpour Fathkouhi, Alireza Namazi, and Heman Shakeri. 2026. *The Stationarity Bias: Stratified Stress-Testing for Time-Series Imputation in Regulated Dynamical Systems.* In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26). ACM, Jeju, Korea.