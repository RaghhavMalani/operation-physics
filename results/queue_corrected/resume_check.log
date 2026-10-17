# D66 GPU resume check

**Verdict: PASS**

- Run A exit 0; run B killed during step 3 (checkpoints then: ['checkpoint-2']); resume exit 0; 445 s in all.
- 1. logs: steps completed 4, resumed at 2, tape rows per step {1: 32, 2: 32, 3: 32, 4: 32}, replays True, resume event True → True
- 2. curriculum cells A vs B, steps 1–4 (step 1–2 outcomes equal: True) → True
  - A: [['energy/2', 'dynamics/2', 'dynamics/1', 'dynamics/1'], ['kinematics/1', 'kinematics/2', 'energy/2', 'kinematics/1'], ['energy/2', 'kinematics/2', 'kinematics/1', 'dynamics/1'], ['kinematics/2', 'dynamics/1', 'kinematics/1', 'dynamics/2']]
  - B: [['energy/2', 'dynamics/2', 'dynamics/1', 'dynamics/1'], ['kinematics/1', 'kinematics/2', 'energy/2', 'kinematics/1'], ['energy/2', 'kinematics/2', 'kinematics/1', 'dynamics/1'], ['kinematics/2', 'dynamics/1', 'kinematics/1', 'dynamics/2']]
- 3. adapter reloaded: ‖B4 − A4‖ = 0.003262, ‖B4 − A2‖ = 0.008914 (‖A4 − A2‖ = 0.009046) → True
- repeat-sampler chunks equal at checkpoint-4: True
