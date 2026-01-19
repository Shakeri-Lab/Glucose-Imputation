# 🧠 Algorithmic Methodology: Realistic Missingness as a Generative Process

## 🎯 Motivation

Missing data in real-world time-series is **not random**. It follows repeatable patterns driven by physiology, behavior, and hardware constraints. Treating missingness as MCAR hides this structure and produces misleading evaluations.

This methodology models missingness as a **learnable stochastic process** and then **re-samples it algorithmically**.

> 🧩 Goal: generate missingness that behaves like reality, not like noise.

---

## 🧱 Core Decomposition

Missingness is represented as a sequence of **gap events**. Each gap is defined by three independent components:

1. ⏰ **When** the gap starts
2. 🧬 **What type** of gap it is (atomic vs sustained)
3. 📏 **How long** it lasts

Formally:

P(missingness) = P(start time) · P(gap type) · P(duration | type)

This separation keeps the model interpretable and modular.

---

## 🧠 Modeling Assumptions

To remain simple and stable, the model assumes:

* ⌚ Fixed temporal grid (Δt = 5 minutes)
* 🔗 Missingness appears as contiguous gaps
* 🧩 Timing and duration are conditionally independent
* 👥 Population-level statistics are shared

These assumptions are explicit and can be relaxed later.

---

## 🔬 Stage I — Learning Missingness from Data

### 🧭 Step 1: Temporal Alignment

All signals are projected onto a complete, fixed-resolution timeline. After this step, missingness is a **binary process** over time rather than an artifact of irregular sampling.

---

### 🧹 Step 2: Structural Day Validation

Days with extremely low data coverage are removed. These days usually represent non-wear or logging failure rather than genuine sensor gaps.

✔ Only days exceeding a minimum coverage threshold are retained.

---

### 🧾 Step 3: Gap Event Representation

Missingness is summarized as a set of gap events:

G = {(t₁, ℓ₁), (t₂, ℓ₂), …}

where:

* tᵢ = gap start time
* ℓᵢ = gap duration

This converts raw binary sequences into a **marked point process**.

---

### ⏰ Step 4: Learning *When* Gaps Start

We estimate the probability that a gap begins at each hour of the day:

P(start | hour = h)

This captures circadian structure such as nighttime compression losses or daytime activity artifacts.

📊 No parametric assumptions—only empirical counting.

---

### 📏 Step 5: Learning *How Long* Gaps Last

Empirically, gap durations fall into two regimes.

#### ⚡ Atomic Gaps

Single-interval gaps (ℓ = Δt) are modeled explicitly using a Bernoulli probability.

These represent brief transmission glitches.

#### 🧩 Sustained Gaps

Longer gaps are modeled using a mixture distribution:

f(ℓ) = w₁·Exp(ℓ) + w₂·Gauss(ℓ) + w₃·Uniform(ℓ)

Interpretation:

* Exp → short outages
* Gauss → structured physiological gaps
* Uniform → rare long-tail events

---

## 🧪 Stage II — Generating New Missingness

### 🎯 Step 6: Gap Triggering

For each hour in the target signal:

* Draw a Bernoulli trial using P(start | hour)
* If successful, initiate a gap

---

### 🎲 Step 7: Duration Sampling

If a gap is triggered:

* With probability P(ℓ = Δt), generate an atomic gap
* Otherwise, sample ℓ from the mixture distribution

Durations are clipped to plausible bounds.

---

### 🧩 Step 8: Mask Realization

Each sampled gap is instantiated as a contiguous missing segment on the temporal grid. The output is a binary missingness mask.

---

## 🧠 Algorithm Summary (Pseudocode)

```
Algorithm LearnAndGenerateMissingness
Input: Time-series dataset D
Output: Missingness mask M

1. Align all signals to fixed grid
2. Remove low-coverage days
3. Extract gap events G = {(tᵢ, ℓᵢ)}
4. Estimate P(start | hour)
5. Estimate P(ℓ = Δt)
6. Fit mixture model for ℓ > Δt

7. For each hour h in new signal:
      if Bernoulli(P(start | h)):
          if Bernoulli(P(ℓ = Δt)):
              ℓ ← Δt
          else:
              ℓ ← Sample from mixture
          Apply gap of length ℓ

Return M
```

---

## 🧭 Conceptual Diagram

```
Raw Signal
   │
   ▼
[ Temporal Alignment ]
   │
   ▼
[ Gap Extraction ]
   │
   ▼
[ Learn Start-Time PMF ]
[ Learn Duration Model ]
   │
   ▼
[ Gap Trigger + Duration Sampler ]
   │
   ▼
Generated Missingness Mask
```

---

## 🌱 Why This Design Works

* 🧩 Clear separation of concerns
* 📊 Fully data-driven
* 🔍 Interpretable at every stage
* 🧪 Produces deployment-relevant missingness

---

## 🧠 Guiding Principle

Missingness is **behavior**, not noise. Modeling it explicitly leads to fairer benchmarks and more robust algorithms.
