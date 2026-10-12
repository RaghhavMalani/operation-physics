# overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512

## Training

| steps | mean reward | zero-var (crps_log) | zero-var (binary) | accuracy ≤1% | median rel. σ | 1σ cov | parse fail | tokens |
|---|---|---|---|---|---|---|---|---|
| 1–50 | -0.118 | 0.5% | 48.0% | 48.2% | 0.074% | 22.6% | 0.2% | 152 |
| 51–100 | -0.154 | 0.5% | 51.0% | 40.6% | 0.074% | 14.7% | 0.1% | 184 |
| 101–150 | -0.235 | 0.0% | 55.0% | 43.6% | 0.078% | 20.8% | 0.1% | 182 |
| 151–200 | -0.185 | 0.5% | 55.5% | 40.9% | 0.077% | 17.6% | 0.4% | 191 |

## Curriculum

- Unlock log (observation #, cell): [[56, 'dynamics/3'], [136, 'dynamics/4'], [180, 'kinematics/3'], [256, 'kinematics/4']]
- Final unlocked cells: dynamics/1, dynamics/2, dynamics/3, dynamics/4, energy/1, energy/2, kinematics/1, kinematics/2, kinematics/3, kinematics/4
- Final EMA pass rate per cell: dynamics/1 0.68, dynamics/2 0.67, dynamics/3 0.37, dynamics/4 0.23, energy/1 0.69, energy/2 0.11, kinematics/1 0.74, kinematics/2 0.24, kinematics/3 0.22, kinematics/4 0.21
- Groups trained per cell: dynamics/1 59, dynamics/2 103, dynamics/3 83, dynamics/4 71, energy/1 97, energy/2 42, kinematics/1 111, kinematics/2 112, kinematics/3 69, kinematics/4 53

## Final eval (frozen eval sets, identical settings)

| split | model | acc ≤1% (σ-free) | ≤2% | ≤5% | ≤10% | strict ≤1% | parse fail | 1σ cov | 2σ cov | median rel. σ | mean reward (crps_log) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Paired tests (same problems, greedy)

(paired tests missing)
