# 🧬 Missing Data Simulation Module

## Overview

This module simulates **realistic sensor dropout patterns** to stress-test glucose imputation algorithms.
It moves beyond simple *random dropout* by implementing **biologically aware masking protocols**.

The system is designed to replicate two real-world CGM failure modes:

* **Homeostatic Loss**: Gaps during stable, low-variability periods (e.g., sleep).
* **Critical Event Loss**: Gaps that obscure post-prandial (post-meal) glucose peaks.

The simulation operates on a **per-day basis**, preserving the original ground truth while generating a companion `cgm_simulated` column containing masked values.

---

## 1. Protocol A: Homeostatic Masking

### *“The Stability Test”*

This protocol targets regions of physiological stability.
It assumes that sensor compression artifacts or benign signal losses are most likely to occur when the patient is sedentary (e.g., sleeping) or fasting.

### ⚙️ Parameters

* **Target Missingness:** 10% – 15% of total daily data
* **Window Duration:** 1.0 – 2.0 hours (randomized per gap)
* **Stability Threshold:** Gradient `< 2.0` mg/dL per minute

---

### 📝 Execution Logic

#### Gradient Calculation

The algorithm computes the first-order difference of the glucose signal to estimate the rate of change at every timestamp:

```python
gradients = np.gradient(cgm)
is_stable = gradients < 2.0
```

This produces a binary *stability map*.

---

#### Candidate Selection (Detailed)

The algorithm iterates over every possible start time `t` in the day.
A window starting at `t` is considered **valid** only if **both** conditions hold:

1. **Zero Carbohydrate Intake**
   Ensures no meal event is masked:

   ```python
   np.sum(window_meal) == 0
   ```

2. **High Stability Ratio**
   At least 80% of points in the window must satisfy the stability constraint:

   ```python
   np.mean(window_stable) > 0.8
   ```

This allows for minor noise while enforcing a flat physiological trend.

---

#### Randomized Application

1. All valid start indices are collected into `valid_starts`
2. The list is randomly shuffled
3. Masking windows are applied sequentially
4. The process stops once cumulative missingness reaches **10–15%**

---

## 2. Protocol B: Hidden Peak Masking

### *“The Stress Test”*

This protocol simulates a **worst-case failure mode** where data loss coincides with high variability.
It explicitly targets meal events, forcing the model to reconstruct glucose excursions **without observing the peak**.

### ⚙️ Parameters

* **Target Missingness:** 20% – 30% of total daily data
* **Window Duration:** 2.5 – 3.0 hours (randomized per gap)
* **Search Radius:** 2 hours post-meal (24 points)

---

### 📝 Execution Logic

#### Meal Identification

The algorithm scans the meal signal to locate carbohydrate intake events:

```python
meal_idx = np.where(meal > 0)
```

---

#### Peak Detection (Region Selection)

For each detected meal at index `meal_idx`:

1. Define a 2-hour search window immediately post-meal
2. Identify the index of the maximum glucose value:

   ```python
   peak_idx = meal_idx + np.argmax(cgm[meal_idx : meal_idx + 24])
   ```
3. This index is designated the **True Peak Candidate**

---

#### Window Centering & Boundary Calculation

Unlike Protocol A, masking here is **deterministic relative to the peak**.

* A random window length is sampled (e.g., 3 hours)
* The mask is centered on the peak:

  ```python
  start_idx = max(0, peak_idx - window_len // 2)
  end_idx   = min(T, start_idx + window_len)
  ```

**Result:**
The mask removes:

* Pre-peak rise
* Peak apex
* Post-peak recovery

---

#### Priority Masking

1. Peak candidates are shuffled
2. Masks are applied sequentially
3. Overlapping masks are **allowed and merged**
4. Process stops once **20–30% missingness** is reached

---

## 3. Mixed Experiment Mode

### *“The Real-World Scenario”*

To prevent overfitting to a single missingness pattern, the **Mixed Mode** introduces stochastic variability by combining Protocol A and Protocol B.

### 🎲 Decision Probability (Per Day)

| Probability | Strategy         | Description                         |
| ----------- | ---------------- | ----------------------------------- |
| 20%         | Protocol A Only  | Minor, stable dropouts (“Good Day”) |
| 30%         | Protocol B Only  | Severe peak loss (“Bad Sensor Day”) |
| 50%         | Combined (A + B) | Maximum difficulty                  |

---

## Data Output Structure

The pipeline returns a modified `DataFrame` with the following fields:

* **`cgm`**
  Ground truth glucose signal (used for loss/error calculation)

* **`cgm_simulated`**
  Input feature with `NaN` values injected according to masking protocols

* **`drop_mask`**
  Boolean array indicating missing indices
  (`True` = data missing)

---

*This framework intentionally decouples biological realism from model assumptions, ensuring that imputation performance reflects clinically meaningful stress conditions rather than artificial randomness.*
