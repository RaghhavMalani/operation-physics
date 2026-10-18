# D69 sensitivity: families N and E with A5 re-fitted on 4-bit SFT completions

FINDINGS #16's verdicts (bf16-fitted A5) stand; this is a sensitivity rerun. Holm-supported = the RL σ beats the A5 σ on that test.

| family / seed | test | RL | A5 (bf16 fit) | A5 (4-bit fit) | supported, bf16 fit | supported, 4-bit fit | changed |
|---|---|---|---|---|---|---|---|
| N seed0 | N1_cov1_error | 0.2652 | 0.0048 | 0.0277 | False | False | no |
| N seed0 | N2_rho | 0.0769 | 0.6764 | 0.6522 | False | False | no |
| N seed0 | N3_crps_log | -0.3163 | -0.2885 | -0.2890 | False | False | no |
| N seed1 | N1_cov1_error | 0.2702 | 0.0023 | 0.0227 | False | False | no |
| N seed1 | N2_rho | 0.0340 | 0.6736 | 0.6507 | False | False | no |
| N seed1 | N3_crps_log | -0.3171 | -0.2895 | -0.2898 | False | False | no |
| N seed2 | N1_cov1_error | 0.2577 | 0.0027 | 0.0302 | False | False | no |
| N seed2 | N2_rho | 0.0712 | 0.6444 | 0.6372 | False | False | no |
| N seed2 | N3_crps_log | -0.3065 | -0.2828 | -0.2824 | False | False | no |
| E | E1a-N (Δcrps_log, RL − A5) | – | -0.0258 | -0.0255 | False | False | no |
| E | E1b-N (Δcrps_log, RL − A5) | – | -0.0259 | -0.0257 | False | False | no |

Changed verdicts: none.
