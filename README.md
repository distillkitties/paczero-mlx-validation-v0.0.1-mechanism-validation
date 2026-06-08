# PAC-Zero/ZPL MLX Mechanism Validation Artifact (v0.0.1)

This archive is a compact **mechanism-validation artifact** for a PAC-Zero/ZPL-style release rule in Apple MLX, using `mlx-community/SmolLM-135M-4bit` as a small test model.

It is intentionally narrow. It validates the MLX implementation structure, transcript-audit behavior, and negative-control checks. It does **not** claim paper-scale utility reproduction, OPT-1.3B/6.7B reproduction, generated SQuAD EM/F1, differential privacy, or paper-level task performance.

## What this v0.0.1 artifact claims

This release demonstrates that the PAC-Zero/ZPL mechanism can be expressed and audited in MLX at smoke scale:

- `M = 126` candidate subsets.
- Balanced membership: each example appears in `M/2 = 63` candidate subsets.
- Per-sample two-point zeroth-order finite differences.
- Sign-quantized subset aggregation.
- ZPL-style release audit: unanimous releases are subset-independent; disagreement releases are independent RNG signs and never index `S*`.
- Rank-8 / alpha-16 LoRA adapters on `q_proj` and `v_proj` targets.
- SST-2 and SQuAD data paths at small scale.
- Negative control showing the audit catches the forbidden `S*`-dependent disagreement release.

## What this v0.0.1 artifact does not claim

This release does **not** claim:

- full PAC-Zero paper reproduction;
- paper-scale OPT-1.3B / OPT-6.7B reproduction;
- paper-scale `1000/500/1000` data or 1000 ZPL steps;
- paper-level SST-2 or SQuAD utility;
- generated SQuAD EM/F1;
- differential privacy.

The included utility checks are smoke-scale preservation checks against a frozen baseline. They are not evidence of paper-level utility reproduction.

## Quick check from the unpacked archive

From the archive root:

```bash
bash run.sh quick
```

This command does not require MLX. It uses any available Python 3.9+ to:

1. compile the Python scripts;
2. run the negative-control audit;
3. rebuild the aggregate report from the included result JSON files;
4. print a final human-readable summary.

Expected high-level outcome:

```text
Overall success: PASS
Negative control: PASS
Privacy/audit checks: PASS
```

## Full local MLX rerun

On Apple Silicon macOS, you can rerun the small MLX validation:

```bash
bash run.sh install
bash run.sh local-all
```

`run.sh install` first tries the available system `python3` with user-site packages. If that cannot install/import MLX, use a separate Python 3.11+ environment.

The default full rerun is intentionally small:

```text
model = mlx-community/SmolLM-135M-4bit
steps = 30
train/dev/eval = 8/8/32
M = 126
LoRA = rank 8 / alpha 16
targets = q_proj + v_proj
layers = all
```

## GitHub Actions rerun

The included workflow is:

```text
.github/workflows/paczero-smollm-validation.yml
```

Use GitHub's **Actions** tab and run the PACZero SmolLM validation workflow with the default inputs for the release-scale smoke run.

## Main preserved outputs

Aggregate JSON:

```text
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_aggregate_results.json
```

Human-readable aggregate report:

```text
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_report.md
```

Negative-control result:

```text
benchmark-results/paczero-smollm-validation-aggregate/zpl_negative_control_results.json
```

Task-level smoke validation outputs:

```text
benchmark-results/paczero-smollm-validation/smollm-135m-4bit-sst2/smollm_validation_results.json
benchmark-results/paczero-smollm-validation/smollm-135m-4bit-squad/smollm_validation_results.json
```

Non-private utility-control outputs:

```text
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-sst2/nonprivate_utility_control_results.json
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-squad/nonprivate_utility_control_results.json
```

## Suggested citation title

```text
PAC-Zero/ZPL MLX Mechanism Validation Artifact (v0.0.1)
```

## Recommended wording

Use this wording when describing the release:

> This artifact validates a PAC-Zero/ZPL-style release mechanism in MLX at smoke scale. It demonstrates balanced `M=126` candidate subsets, strict transcript auditing, q/v LoRA wiring, two-point zeroth-order finite-difference plumbing, and a negative control that catches an `S*`-dependent disagreement release. It does not claim paper-scale utility reproduction.
