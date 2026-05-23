# CS224R First Milestone Plan Prompt

## Project Working Title

**Frequency-Gated Action Chunking for Generative Robot Policies**

Alternative title if we focus only on robomimic:

**Adaptive Frequency-Gated Action Chunking on robomimic Manipulation Tasks**

---

## One-Sentence Project Pitch

We study whether action chunks in generative robot policies should always use full-frequency raw actions, or whether low-frequency plans plus adaptively gated high-frequency corrections can improve the smoothness–precision tradeoff in manipulation.

---

## Current Milestone Goal

For the first milestone, we do **not** need to finish the full policy training or rollout evaluation.  
The goal is to complete a minimal but convincing experiment showing that frequency decomposition of action chunks is meaningful.

The milestone should demonstrate:

1. We can load a real robot imitation dataset.
2. We can construct action chunks.
3. We can decompose each action chunk into low-frequency and high-frequency components.
4. We can measure whether low-frequency actions capture smooth motion and whether high-frequency components become important near gripper/contact-like transitions.
5. We have a concrete next step toward training a generative policy with adaptive frequency gating.

---

## Recommended Dataset for Milestone

### Primary MVP Dataset

Use:

```text
robomimic Lift-PH low_dim
```

Reason:

- It is simple and reliable.
- It matches the original proposal’s robomimic setup.
- It avoids the engineering risk of Unitree G1 / humanoid data loading.
- It still lets us test temporal action representation without changing action semantics.
- The original action labels can be used directly.

### Optional Extension Dataset

If Lift is too simple or high-frequency effects are weak, add:

```text
robomimic Can-PH low_dim
```

Can has more alignment/contact complexity and may show stronger high-frequency effects.

### Do Not Use for Milestone

Avoid for the milestone:

```text
Square
Transport
Unitree full-body robot_q_desired
full WBC / humanoid joint targets
image-based policy
real robot rollout
```

These are too risky for the first milestone.

---

## Core Research Question

Standard action chunking predicts the raw future action sequence:

\[
A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
\]

However, manipulation actions may have different temporal structure:

- Low-frequency components may capture smooth reaching and lifting.
- High-frequency components may capture grasp/contact corrections.
- Always using full-frequency actions may increase jitter.
- Always using only low-frequency actions may lose precision.

We ask:

> Can an adaptive frequency-gated action representation selectively use high-frequency corrections only when needed?

---

## Method Overview

### Step 1: Construct action chunks

For each robomimic trajectory, take native dataset actions:

\[
a_0, a_1, \ldots, a_T
\]

Construct chunks:

\[
A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
\in \mathbb{R}^{H \times d}
\]

Recommended milestone values:

```text
H = 16
stride = 1
K_freq ∈ {2, 4, 8, 12, 16}
K_exec = 4 for boundary-jerk simulation
```

Do not cross episode boundaries when constructing chunks.

---

### Step 2: Apply DCT along the time dimension

For each chunk:

\[
Z_t = \mathrm{DCT}(A_t)
\]

where \(Z_t \in \mathbb{R}^{H \times d}\).

Split:

\[
Z_{\text{low}} = Z_{0:K}
\]

\[
Z_{\text{high}} = Z_{K:H}
\]

---

### Step 3: Reconstruct low-frequency plan

Set high-frequency coefficients to zero:

\[
A_{\text{low}} = \mathrm{IDCT}([Z_{\text{low}}, 0])
\]

This is the smooth low-frequency action plan.

---

### Step 4: Define high-frequency residual / correction

There are two equivalent views.

#### View A: time-domain residual

\[
R_{\text{high}} = A - A_{\text{low}}
\]

#### View B: frequency-domain high-frequency coefficients

\[
Z_{\text{high}} = Z_{K:H}
\]

For the final method, prefer the frequency-domain view:

\[
\hat{A}
=
\mathrm{IDCT}([\hat{Z}_{\text{low}}, \alpha \hat{Z}_{\text{high}}])
\]

where \(\alpha \in [0,1]\) controls how much high-frequency signal is used.

---

## Adaptive Gate Idea

Instead of fixing \(\alpha\), learn or estimate:

\[
\alpha = g(o_t)
\]

where \(o_t\) is the current observation.

Intuition:

```text
alpha ≈ 0: smooth reaching, low-frequency is enough
alpha ≈ 1: grasp/contact transition, high-frequency correction is needed
```

For milestone, do not train the full learned gate yet.  
Instead, compute an oracle / pseudo-label gate:

\[
\alpha^*
=
\arg\min_{\alpha \in \{0, 0.25, 0.5, 0.75, 1\}}
\left[
\|\mathrm{IDCT}([Z_{\text{low}}, \alpha Z_{\text{high}}]) - A\|^2
+
\lambda \alpha
\right]
\]

This tests whether different chunks need different amounts of high-frequency signal.

Recommended:

```text
lambda = small penalty, e.g. 1e-3 to 1e-2 after action normalization
alphas = [0, 0.25, 0.5, 0.75, 1]
```

---

## Milestone Experiments

### Experiment 1: Frequency decomposition diagnostic

Use robomimic Lift-PH low_dim.

For all action chunks:

1. Compute DCT.
2. Reconstruct low-frequency versions for \(K = 2, 4, 8, 12, 16\).
3. Compute reconstruction MSE.
4. Compute action smoothness.
5. Compute high-frequency energy ratio.
6. Compare high-frequency energy near vs away from gripper transitions.

This experiment is enough for the first milestone.

---

### Experiment 2: Pseudo-label adaptive gate analysis

For each chunk:

1. Compute \(\alpha^*\) using the oracle objective above.
2. Define gripper/contact-like transition events using the gripper action dimension.
3. Compare \(\alpha^*\) near gripper transitions vs away from gripper transitions.

Expected result:

```text
alpha* should be higher near gripper/contact-like transitions
alpha* should be lower during smooth reaching
```

If this happens, it supports the adaptive frequency-gating hypothesis.

---

### Optional Experiment 3: Small generative policy run

If time permits, train one small policy:

```text
Policy: Flow Matching or Diffusion Policy
Task: Lift-PH low_dim
Target: raw action chunks OR frequency coefficients
```

Minimum reportable result:

```text
training loss
validation decoded MSE
one qualitative or rollout result if available
```

Do not make this mandatory for the milestone.

---

## Metrics

### 1. Reconstruction MSE

\[
\mathrm{MSE}_{rec}(K)
=
\|A - \mathrm{IDCT}([Z_{0:K},0])\|^2
\]

Purpose:

Shows how much of the action chunk can be represented by low-frequency coefficients.

Expected:

```text
MSE decreases as K increases.
If K=4 or K=8 already gives low MSE, actions have strong low-frequency structure.
```

---

### 2. Smoothness

\[
S(A)
=
\frac{1}{H-1}
\sum_{i=0}^{H-2}
\|a_{i+1} - a_i\|_2^2
\]

Purpose:

Measures action jitter.

Expected:

```text
Low-frequency reconstructions should be smoother than raw actions.
```

---

### 3. High-Frequency Energy Ratio

\[
E_{\text{high}}(K)
=
\frac{
\sum_{k=K}^{H-1}\|Z_k\|^2
}{
\sum_{k=0}^{H-1}\|Z_k\|^2
}
\]

Purpose:

Measures how much action information lives in high-frequency components.

---

### 4. Gripper Transition Event

Use the gripper dimension of robomimic actions.

Define:

\[
|\Delta a_{\text{gripper}}| > \tau
\]

where \(\tau\) can be the 90th or 95th percentile of gripper action differences.

Compare:

```text
E_high near gripper transitions
E_high away from gripper transitions
```

Expected:

```text
High-frequency energy may increase near gripper/contact-like events.
```

---

### 5. Oracle Gate Value

For each chunk, compute:

\[
\alpha^*
\]

Compare:

```text
alpha* near gripper transitions
alpha* away from gripper transitions
```

Expected:

```text
alpha* should be larger near gripper/contact-like transitions.
```

---

### 6. Boundary Jerk

For predicted or reconstructed chunks, simulate receding-horizon execution.

\[
J_{\text{boundary}}
=
\|\hat{a}^{new}_{t} - \hat{a}^{old}_{t-1}\|_2
\]

Purpose:

Measures discontinuity between consecutive action chunks.

Expected:

```text
Low-frequency chunks should reduce boundary jerk.
Full-frequency chunks may be more accurate but jerkier.
Adaptive gate may provide a tradeoff.
```

---

## Expected Milestone Figure

Use one figure with two panels.

### Panel A

```text
x-axis: retained frequency K
y-axis: reconstruction MSE
curves/bars: Lift-PH actions
```

Shows whether low-frequency coefficients reconstruct most action information.

### Panel B

Either:

```text
high-frequency energy near gripper transitions vs away from transitions
```

or:

```text
oracle alpha* near gripper transitions vs away from transitions
```

This directly supports the adaptive high-frequency-gating story.

---

## Milestone Report Outline

The milestone report should be one page.

### Title

**Adaptive Frequency-Gated Action Chunking for Generative Robot Policies**

### 1. Objective

Write:

> Our original project studied how action representation affects generative robot policies. We pivot from spatial action representations such as joint vs end-effector control to temporal frequency representations of native action chunks. This avoids controller relabeling while preserving the core research question: how does action representation affect learning dynamics, smoothness, and precision?

### 2. Experiment Conducted

Write:

> We used robomimic Lift-PH low_dim and extracted native action chunks of horizon \(H=16\). We applied DCT along the temporal dimension, reconstructed low-frequency versions using \(K \in \{2,4,8,12,16\}\), and measured reconstruction error, smoothness, high-frequency energy, and gripper-transition-conditioned high-frequency usage.

### 3. Initial Findings

Fill this based on results:

Option A, if results support hypothesis:

> Preliminary results show that low-frequency coefficients capture most smooth action variation, while high-frequency energy / oracle gate values increase near gripper transitions. This supports our hypothesis that high-frequency actions are not uniformly needed but should be used selectively.

Option B, if results do not support hypothesis:

> Preliminary results show weaker separation between low- and high-frequency components than expected. This suggests either gripper transitions are an imperfect proxy for contact, or the chosen task is too simple. We will test Can-PH and evaluate boundary jerk / rollout behavior next.

### 4. Change from Proposal

Write:

> The overall objective remains action representation in generative robot policies. We changed the representation studied from joint-vs-EE controller semantics to frequency-domain action chunk representations because controller relabeling in robomimic would confound the comparison. The new setup preserves native actions and modifies only their representation.

### 5. Next Steps

Write:

> Next, we will train a generative action-chunk policy using raw action chunks, low-frequency DCT chunks, and adaptive frequency-gated DCT chunks. We will compare decoded MSE, smoothness, boundary jerk, rollout success, and gate usage near gripper transitions. If Lift is too simple, we will add Can-PH.

### 6. AI Tools Disclosure

Write:

> We used ChatGPT for project planning and writing assistance. Dataset loading, DCT implementation, experiments, analysis, and conclusions were developed and verified independently by the team.

---

## Concrete To-Do List

### Before Milestone

```text
[ ] Download robomimic Lift-PH low_dim.
[ ] Load hdf5 file and inspect action dimensions.
[ ] Identify gripper action dimension.
[ ] Construct H=16 action chunks without crossing episode boundaries.
[ ] Normalize actions using training-set statistics.
[ ] Implement DCT and IDCT.
[ ] Compute low-frequency reconstruction for K = 2, 4, 8, 12, 16.
[ ] Compute reconstruction MSE vs K.
[ ] Compute smoothness vs K.
[ ] Compute high-frequency energy ratio.
[ ] Define gripper transition events.
[ ] Compare high-frequency energy near vs away from gripper transitions.
[ ] Compute oracle alpha* for adaptive gate.
[ ] Make one 2-panel figure.
[ ] Write one-page milestone report.
```

---

## Minimal Python Pseudocode

```python
import h5py
import numpy as np
from scipy.fft import dct, idct

H = 16
K_VALUES = [2, 4, 8, 12, 16]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def dct_time(x):
    return dct(x, axis=0, norm="ortho")

def idct_time(z):
    return idct(z, axis=0, norm="ortho")

def low_reconstruct(A, K):
    Z = dct_time(A)
    Z_low = np.zeros_like(Z)
    Z_low[:K] = Z[:K]
    A_low = idct_time(Z_low)
    return A_low, Z

def high_energy_ratio(Z, K):
    high = np.sum(Z[K:] ** 2)
    total = np.sum(Z ** 2) + 1e-8
    return high / total

def oracle_alpha(A, Z, K, lambda_alpha=1e-3):
    best_alpha = None
    best_score = float("inf")
    for alpha in ALPHAS:
        Z_hat = np.zeros_like(Z)
        Z_hat[:K] = Z[:K]
        Z_hat[K:] = alpha * Z[K:]
        A_hat = idct_time(Z_hat)
        mse = np.mean((A_hat - A) ** 2)
        score = mse + lambda_alpha * alpha
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, best_score
```

---

## Suggested Final Claim

If the milestone results are positive, the final project can claim:

> Low-frequency coefficients capture most smooth manipulation motion, while high-frequency coefficients become more useful near gripper/contact-like transitions. This motivates an adaptive frequency-gated action representation that uses high-frequency corrections selectively rather than always using full-frequency raw action chunks.

---

## Fallback Plan

If Lift-PH does not show meaningful high-frequency effects:

```text
1. Add Can-PH.
2. Use boundary jerk instead of gripper-transition energy as the primary diagnostic.
3. Treat Lift as a sanity check and Can as the main task.
4. Keep the final method unchanged.
```

If DCT results are weak:

```text
1. Report that fixed frequency bases are less useful than expected on Lift.
2. Analyze whether action normalization or task simplicity caused the issue.
3. Try PCA action chunks as a data-adaptive alternative.
4. Keep the project framed as action representation analysis.
```
