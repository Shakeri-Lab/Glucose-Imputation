Algorithmic Specification: Realistic Missingness Generator

1. Introduction

This approach models missingness as a Structured Stochastic Process with two distinct dependencies:

Temporal Dependency (Circadian Rhythm): The probability of a gap occurring is non-stationary; it varies significantly by time of day (e.g., compression artifacts during sleep vs. signal loss during commutes).

Duration Dependency (Heavy-Tailed Distribution): The length of a gap follows a complex, multi-modal distribution. Most gaps are short (single packet loss), but some are systematic (restarts) or long-term (sensor failure).

The system operates in two distinct phases: Parameter Estimation (Learning) and Stochastic Synthesis (Generating).

2. Phase I: Statistical Parameter Estimation

The objective of this phase is to derive a probabilistic model $P(Gap | t, \text{history})$ from historical raw data.

A. Temporal Grid Standardization

Before statistical analysis, irregular sensor streams must be mapped to a rigorous discrete time domain.

Algorithm: The timeline is discretized into fixed bins (e.g., $\Delta t = 5$ mins) aligned to absolute timestamps (00:00, 00:05, ...).

Mechanism: A "Left-Join" operation is performed against a theoretical complete time grid.

Logic: Any bin in the theoretical grid without a matching valid sensor reading is flagged as a "Gap". This conversion is crucial because it transforms "implicit" missingness (the absence of a database row) into "explicit" missingness (the presence of a NaN state), enabling the calculation of gap durations.

B. Signal-to-Noise Filtering ("Ghost Days")

A critical preprocessing step separates behavioral artifacts (user non-compliance) from technical artifacts (sensor failure).

Concept: A "Ghost Day" is defined as a 24-hour period where the user barely wore the device (e.g., they took it off for 20 hours). If included in the analysis, these days would introduce massive gaps (1000+ minutes) that are not representative of sensor performance.

Algorithm: We calculate the Data Density ($D$) for each 24-hour window.

$$D = \frac{\text{Observed Points}}{\text{Maximum Theoretical Points}}$$

Thresholding: If $D < \text{Threshold}$ (e.g., 0.5), the entire 24-hour window is excluded from the learning set. This ensures the derived distribution represents the conditional probability of failure given that the device is being worn.

C. Hybrid Distribution Modeling

The core innovation is the modeling of gap durations. Empirical analysis reveals that gap durations follow a Bi-Modal, Heavy-Tailed Distribution that cannot be fitted by standard Gaussian or Exponential curves alone. The algorithm employs a Two-Stage Hybrid Model:

Stage 1: The Discrete "Micro-Gap" Event (Point Mass)

The most frequent error in digital transmission is a single missed packet (e.g., one 5-minute void). This behaves as a discrete event rather than part of a continuous curve.

$$P_{micro} = \frac{N_{single}}{N_{total}}$$

Interpretation: This separates the "glitches" from the "outages."

Stage 2: The "Macro-Gap" Mixture Model (Continuous Tail)

For gaps longer than a single packet ($t > 5$), the probability density function (PDF) is approximated using a Gaussian-Exponential Mixture Model with a noise floor. This composite function $f(t)$ captures three distinct physical failure modes:

$$f(t) = \underbrace{A e^{-kt}}_{\text{Random Drops}} + \underbrace{B e^{-\frac{(t-\mu)^2}{2\sigma^2}}}_{\text{Systemic Reboots}} + \underbrace{C}_{\text{Outliers}}$$

Memoryless Failures (Exponential Component):

Physics: Random, independent connection drops (e.g., Bluetooth interference, distance from receiver). These follow a Poisson process, resulting in exponentially decaying durations.

Dominance: Short to medium gaps (10–45 mins).

Systemic Failures (Gaussian Component):

Physics: Predictable duration events. For example, a sensor warm-up cycle or a firmware crash-and-reboot sequence often takes a fixed amount of time (e.g., exactly 2 hours). This creates a "hump" in the distribution centered around mean $\mu$.

Dominance: Medium to long gaps (~120 mins).

Background Noise (Uniform Component):

Physics: Random, long-tail outliers (e.g., sensor peeling off, battery death). This ensures the model can generate rare, very long gaps.

Optimization: The parameters $(A, k, B, \mu, \sigma, C)$ are optimized using Non-linear Least Squares (Levenberg-Marquardt algorithm) to minimize the residual between the mixture equation and the empirical histogram of observed macro-gaps.

3. Phase II: Stochastic Mask Synthesis

The generator uses the learned parameters to synthesize new boolean masks via a Monte Carlo simulation.

A. Non-Homogeneous Poisson Process (The "When")

Instead of initiating gaps uniformly (a Homogeneous Poisson Process), the algorithm creates a Non-Homogeneous process where the arrival rate $\lambda(t)$ depends on the time of day.

Input: An hourly probability vector $H = [p_0, p_1, ..., p_{23}]$, derived from the frequency of gap starts in the training data.

Process:

Iterate through every hour $h$ in the target timeframe.

Perform a Bernoulli trial with probability $p_h$.

Offset Injection: If successful, a random minute offset $m \in [0, 59]$ is selected uniformly. The gap start time becomes $T_{start} = \text{Hour}_h + m$.

B. Hierarchical Duration Sampling (The "How Long")

Once a start time $T_{start}$ is established, the duration $\tau$ is determined via a hierarchical sampling tree:

Micro vs. Macro Decision:

Draw random number $r_1 \sim U(0,1)$.

If $r_1 < P_{micro}$, then $\tau = 5 \text{ mins}$.

Mixture Component Selection (If Macro):

To sample from the complex PDF $f(t)$, we first calculate the Area Under the Curve (AUC) (or "Probability Mass") for each component:

Weight Exponential ($w_e$) $\approx \int A e^{-kt} dt$

Weight Gaussian ($w_g$) $\approx \int B \mathcal{N}(\mu, \sigma) dt$

Weight Uniform ($w_u$) $\approx \int C dt$

Normalize weights so $\sum w = 1$.

Draw random number $r_2 \sim U(0,1)$ to select a component based on these weights.

Final Value Generation:

If Exponential selected: Generate $\tau \sim \text{Exp}(1/k)$.

If Gaussian selected: Generate $\tau \sim \mathcal{N}(\mu, \sigma)$.

If Uniform selected: Generate $\tau \sim U(10, 240)$.

C. Vectorized Projection & Collision Handling

The abstract tuples $(T_{start}, \tau)$ are mapped onto the boolean mask array.

Vectorization: The start time and duration are converted into array indices $[i_{start}, i_{end}]$.

Collision Handling (Boolean OR):
The algorithm initializes a False array. For each generated gap, it sets the slice mask[start:end] = True.

Significance: If a new gap is generated while a previous gap is still active (an overlapping collision), the Boolean OR operation naturally merges them. This mimics real-world scenarios where multiple failure causes (e.g., interference followed by a reboot) can overlap, resulting in a single longer observed gap.