# Merge check (post-hoc diagnostic for D65; CPU, bf16, E1a final adapter)

3 E1a eval prefixes whose bf16 and 4-bit greedy σ suffixes differ.

- greedy suffix, unmerged == merged: 1 / 3
- merged (CPU) == the bf16 eval's suffix: 1 / 3
- unmerged bf16 == the 4-bit eval's suffix: 3 / 3

| key | bf16 eval | 4-bit eval | bf16 unmerged | bf16 merged | max |Δp| merged vs unmerged | top-5 after ± (base / unmerged) |
|---|---|---|---|---|---|---|
| ['kinematics', 1, 1000000000] | ' 0.0003 min</answer>' | ' 0.001 min</answer>' | ' 0.001 min</answer>' | ' 0.0003 min</answer>' | 0.0000 | ' ' 1.00, ' <' 0.00, ' .' 0.00, ' same' 0.00, ' zero' 0.00 / ' ' 1.00, ' .' 0.00, ' <' 0.00, ' same' 0.00, ' etc' 0.00 |
| ['kinematics', 2, 1000000001] | ' 0.0003 km</answer>' | ' 0.0002 km</answer>' | ' 0.0002 km</answer>' | ' 0.0001 km</answer>' | 0.0000 | ' ' 1.00, ' <' 0.00, ' -' 0.00, ' same' 0.00, ' .' 0.00 / ' ' 1.00, ' <' 0.00, ' .' 0.00, ' same' 0.00, ' -' 0.00 |
| ['kinematics', 3, 1000000002] | ' 10 ms</answer>' | ' 100 ms</answer>' | ' 100 ms</answer>' | ' 100 ms</answer>' | 0.0000 | ' ' 1.00, ' -' 0.00, ' <' 0.00, ' same' 0.00, ' not' 0.00 / ' ' 1.00, ' -' 0.00, ' same' 0.00, ' <' 0.00, ' (-' 0.00 |
