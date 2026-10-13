# overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512

## Training

| steps | mean reward | zero-var (crps_log) | zero-var (binary) | accuracy ≤1% | median rel. σ | 1σ cov | parse fail | tokens |
|---|---|---|---|---|---|---|---|---|
| 1–50 | -0.118 | 0.5% | 48.0% | 48.2% | 0.074% | 22.6% | 0.2% | 152 |
| 51–100 | -0.154 | 0.5% | 51.0% | 40.6% | 0.074% | 14.7% | 0.1% | 184 |
| 101–150 | -0.235 | 0.0% | 55.0% | 43.6% | 0.078% | 20.8% | 0.1% | 182 |
| 151–200 | -0.185 | 0.5% | 55.5% | 40.9% | 0.077% | 17.6% | 0.4% | 191 |
| 201–250 | -0.211 | 1.5% | 54.5% | 41.6% | 0.063% | 17.5% | 0.3% | 177 |

## Curriculum

- Unlock log (observation #, cell): [[56, 'dynamics/3'], [136, 'dynamics/4'], [180, 'kinematics/3'], [256, 'kinematics/4']]
- Final unlocked cells: dynamics/1, dynamics/2, dynamics/3, dynamics/4, energy/1, energy/2, kinematics/1, kinematics/2, kinematics/3, kinematics/4
- Final EMA pass rate per cell: dynamics/1 0.62, dynamics/2 0.72, dynamics/3 0.54, dynamics/4 0.23, energy/1 0.46, energy/2 0.17, kinematics/1 0.59, kinematics/2 0.38, kinematics/3 0.16, kinematics/4 0.18
- Groups trained per cell: dynamics/1 80, dynamics/2 121, dynamics/3 101, dynamics/4 92, energy/1 124, energy/2 58, kinematics/1 137, kinematics/2 136, kinematics/3 89, kinematics/4 62

## Final eval (frozen eval sets, identical settings)

| split | model | acc ≤1% (σ-free) | ≤2% | ≤5% | ≤10% | strict ≤1% | parse fail | 1σ cov | 2σ cov | median rel. σ | mean reward (crps_log) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Paired tests (same problems, greedy)

(paired tests missing)
