## 📊 Evaluation Metrics

Metrics are computed strictly on imputed gaps to evaluate the model's generative performance.

### 1. Clarke Error Grid Analysis
We evaluate clinical safety using the **Clarke Error Grid**, stratifying results by physiological **risk**.
* **Zone A (Clinically Accurate)
* **Zone B (Benign Errors)
* **Zone C (Overcorrection)
* **Zone D (Failure to Detect)
* **Zone E (Erroneous Treatment)

### 2. Statistical Accuracy Metrics
* **MSE (Mean Squared Error):** $\frac{1}{N} \sum (\hat{y} - y)^2$. Penalizes large outliers.
* **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Error magnitude in mg/dL, sensitive to outliers.
* **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |\hat{y} - y|$. Average linear error in mg/dL.
* **Bias:** $\frac{1}{N} \sum (\hat{y} - y)$. Indicates systematic over-prediction (+) or under-prediction (-).
* **emp_SE (Empirical Standard Error):** Sample standard deviation of residuals; measures prediction precision (noise).

### 3. Domain-Specific Metrics
* **MARD (Mean Absolute Relative Difference):** The standard CGM accuracy metric.
  $$\text{MARD} = \frac{1}{N} \sum \left| \frac{\hat{y} - y}{y + \epsilon} \right| \times 100$$

### 4. Shape & Temporal Metrics
* **DTW (Dynamic Time Warping):** Measures the similarity of **shape** between predicted and actual curves, allowing for non-linear time alignment. It is calculated **gap-wise** (averaged across missing segments) to validate that the model recovers correct physiological trends (e.g., recovery rates) even if slight time-shifts occur.