# overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512

## Training

287 steps, 58.9 s/step, peak 4.58 GiB torch / 7.73 GiB device.

| steps | mean reward | zero-var (crps_log) | zero-var (binary) | accuracy ≤1% | median rel. σ | 1σ cov | parse fail | tokens |
|---|---|---|---|---|---|---|---|---|
| 1–50 | -0.118 | 0.5% | 48.0% | 48.2% | 0.074% | 22.6% | 0.2% | 152 |
| 51–100 | -0.154 | 0.5% | 51.0% | 40.6% | 0.074% | 14.7% | 0.1% | 184 |
| 101–150 | -0.235 | 0.0% | 55.0% | 43.6% | 0.078% | 20.8% | 0.1% | 182 |
| 151–200 | -0.185 | 0.5% | 55.5% | 40.9% | 0.077% | 17.6% | 0.4% | 191 |
| 201–250 | -0.211 | 1.5% | 54.5% | 41.6% | 0.063% | 17.5% | 0.3% | 177 |
| 251–287 | -0.129 | 2.0% | 49.3% | 44.8% | 0.072% | 17.6% | 0.3% | 179 |

## Curriculum

- Unlock log (observation #, cell): [[56, 'dynamics/3'], [136, 'dynamics/4'], [180, 'kinematics/3'], [256, 'kinematics/4']]
- Final unlocked cells: dynamics/1, dynamics/2, dynamics/3, dynamics/4, energy/1, energy/2, kinematics/1, kinematics/2, kinematics/3, kinematics/4
- Final EMA pass rate per cell: dynamics/1 0.85, dynamics/2 0.59, dynamics/3 0.56, dynamics/4 0.24, energy/1 0.41, energy/2 0.16, kinematics/1 0.65, kinematics/2 0.34, kinematics/3 0.18, kinematics/4 0.24
- Groups trained per cell: dynamics/1 95, dynamics/2 141, dynamics/3 116, dynamics/4 106, energy/1 142, energy/2 70, kinematics/1 146, kinematics/2 156, kinematics/3 105, kinematics/4 71

## Final eval (frozen eval sets, identical settings)

| split | model | acc ≤1% (σ-free) | ≤2% | ≤5% | ≤10% | strict ≤1% | parse fail | 1σ cov | 2σ cov | median rel. σ | mean reward (crps_log) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id | raw base | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | – | – | – | -5.000 | 484 |
| id | SFT only | 38.0% | 44.5% | 60.5% | 67.5% | 38.0% | 0.0% | 15.0% | 20.5% | 0.050% | -0.224 | 196 |
| id | SFT + RL | 40.0% | 46.5% | 60.5% | 71.0% | 40.0% | 0.0% | 15.5% | 20.0% | 0.042% | -0.177 | 195 |
| depth_extrapolation | raw base | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | – | – | – | -5.000 | 511 |
| depth_extrapolation | SFT only | 31.0% | 40.0% | 51.0% | 60.0% | 31.0% | 0.0% | 11.5% | 16.0% | 0.081% | -0.215 | 317 |
| depth_extrapolation | SFT + RL | 31.0% | 39.5% | 51.0% | 59.5% | 31.0% | 0.0% | 10.0% | 15.0% | 0.064% | -0.220 | 316 |

## Paired tests (same problems, greedy)

| comparison | split | strict ≤1%: before → after (only-after / only-before), McNemar p | σ-free ≤1%, McNemar p | parsed, McNemar p | reward Δ, Wilcoxon p |
|---|---|---|---|---|---|
| rl vs sft | depth_extrapolation | 62 → 62 (4/4), p = 1 | 62 → 62, p = 1 | 200 → 200, p = 1 | -0.005, p = 0.373 |
| rl vs sft | id | 76 → 80 (4/0), p = 0.125 | 76 → 80, p = 0.125 | 200 → 200, p = 1 | +0.046, p = 0.0175 |
| rl vs base | depth_extrapolation | 0 → 62 (62/0), p = 4.34e-19 | 0 → 62, p = 4.34e-19 | 0 → 200, p = 1.24e-60 | +4.780, p = 2.09e-34 |
| rl vs base | id | 0 → 80 (80/0), p = 1.65e-24 | 0 → 80, p = 1.65e-24 | 0 → 200, p = 1.24e-60 | +4.823, p = 2.09e-34 |
| sft vs base | depth_extrapolation | 0 → 62 (62/0), p = 4.34e-19 | 0 → 62, p = 4.34e-19 | 0 → 200, p = 1.24e-60 | +4.785, p = 1.44e-34 |
| sft vs base | id | 0 → 76 (76/0), p = 2.65e-23 | 0 → 76, p = 2.65e-23 | 0 → 200, p = 1.24e-60 | +4.776, p = 2.09e-34 |
