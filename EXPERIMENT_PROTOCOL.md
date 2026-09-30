# Correctness Experiment Protocol

## 1. Reproducibility Protocol

All research experiments must adhere to the following reproducibility invariants:

1. **Immutable Configuration**:
   Every experiment run must record an immutable YAML/JSON configuration containing:
   - `experiment_id` (e.g. `EXP-001-baseline-factuality`)
   - `model` (name, revision, temperature, top_p, seed)
   - `dataset` (name, split sizes, version hash)
   - `verifier` (verifier_id, parameters, threshold)
   - `statistical_parameters` ($\alpha$, confidence level $1 - \delta$, calibration sample count)
   - `code_commit` (git SHA)
2. **Three-Way Data Split**:
   - **Calibration Set ($D_{\text{cal}}$)**: Used strictly to compute conformal quantiles, risk thresholds, and temperature scaling.
   - **Validation Set ($D_{\text{val}}$)**: Used to tune hyperparameters, stopping rules, and verifier weights.
   - **Test Set ($D_{\text{test}}$)**: Evaluated only once per finalized method. Never used during exploratory tuning.
3. **Multi-Seed Evaluation**:
   Key benchmarks must be evaluated across at least 5 distinct random seeds ($S \in \{42, 101, 2024, 777, 999\}$). Report mean, standard deviation, and 95% confidence intervals.
4. **Failure Conditions & Red Status**:
   An experiment is marked **RED** if:
   - Empirical coverage on test data falls systematically below $1 - \alpha$ under the stated assumptions.
   - False Certification Rate (FCR) exceeds certified risk bounds.
   - Run metadata is insufficient to reconstruct the identical result.

## 2. Standard Metric Definitions

- **Empirical Coverage**:
  $$\text{Coverage} = \frac{1}{|D_{\text{test}}|} \sum_{i=1}^{|D_{\text{test}}|} \mathbb{I}(Y_i \in C(X_i))$$
- **Selective Risk**:
  $$\text{Risk}_{\text{sel}} = \frac{\sum_{i=1}^n L(Y_i, \hat{Y}_i) \cdot \mathbb{I}(\text{Accept}_i)}{\sum_{i=1}^n \mathbb{I}(\text{Accept}_i)}$$
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **False Certification Rate (FCR)**:
  $$\text{FCR} = P(\text{Certified} \mid \text{Output is Incorrect})$$
