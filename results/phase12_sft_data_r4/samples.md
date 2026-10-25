# R4-1 SFT data (D113): examples

## seed 60000000: kinematics / depth 1

Problem:

```
Consider a cart on a straight, horizontal track. Phase 1: the cart has initial speed u and speeds up with constant acceleration a1 during a time t1; it travels s1 and ends the phase with speed v. Phase 2: the cart keeps the constant speed v for a time t2 and travels s2. Phase 3: the cart slows down with constant deceleration of magnitude b and stops after a time t3, having travelled s3 in this phase. For the entire motion, d_tot is the total distance, t_tot the total time, and v_avg the average speed.

Given:
- t1 = 11.341 s ± 5% (the duration of phase 1).
- The length of the cart is len_cart = 841.76 mm.
- The braking time in phase 3, t3, equals 0.33562 min ± 2%.
- t_tot = 0.93315 min ± 5% (the total duration of the trip).

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

What is the duration of phase 2, t2? Give your answer in min. Show your calculation first; no intermediate values need to be listed in <steps>.
```

**explicit** (masked: [])

```
SI values: t1 = 11.341 s; t3 = 0.33562 min = 20.14 s; t_tot = 0.93315 min = 55.99 s.
Not needed: len_cart.
t2 = -t1 - t3 + t_tot = -11.34 - 20.14 + 55.99 = 24.51 s
In min: t2 = 0.4085 min
S_t1 = -t1/t2 = -11.34/24.51 = -0.463
S_t3 = -t3/t2 = -20.14/24.51 = -0.822
S_t_tot = t_tot/t2 = 55.99/24.51 = 2.28
<steps>
sigma_rel = sqrt((-0.463*0.05)^2 + (-0.822*0.02)^2 + (2.28*0.05)^2) = 0.12
</steps>
<answer>0.4085 ± 0.049 min</answer>
<confidence>10%</confidence>
```

**supervised** (masked: [])

```
SI values: t1 = 11.341 s; t3 = 0.33562 min = 20.14 s; t_tot = 0.93315 min = 55.99 s.
Not needed: len_cart.
t2 = -t1 - t3 + t_tot = -11.34 - 20.14 + 55.99 = 24.51 s
In min: t2 = 0.4085 min
<steps>
</steps>
<answer>0.4085 ± 0.048 min</answer>
<confidence>10%</confidence>
```

## seed 60000375: energy / depth 2

Problem:

```
A block of mass m is released from rest at height h on a frictionless ramp. At the top, its gravitational potential energy relative to the bottom of the ramp is U. It slides down and reaches the bottom with kinetic energy K1 and speed v1. It then crosses a rough horizontal patch of length L where the coefficient of kinetic friction is mu; friction does work of magnitude W_f on the block, which leaves the patch with kinetic energy K2 and speed v2. Finally, on a frictionless floor, the block runs into a light horizontal spring with spring constant k and compresses it by a maximum distance x, at which moment the spring force on the block has magnitude F_s.

Given:
- m = 15.958 kg (the mass of the block).
- It is known that K2 = 0.10100 kJ ± 2%, where K2 is the kinetic energy of the block after the rough patch.
- It is known that v1 = 6.6539 m/s, where v1 is the speed of the block at the bottom of the ramp.
- The coefficient of kinetic friction on the rough patch, mu, equals 0.12930 ± 5%.
- Use g = 9.81 m/s^2.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

What is the length of the rough patch, L? Give your answer in km. In <steps>, report the intermediate values v2.
```

**explicit** (masked: [])

```
SI values: K2 = 0.10100 kJ = 101 J; m = 15.958 kg; mu = 0.12930; v1 = 6.6539 m/s.
v2 = sqrt(2)*sqrt(K2)/sqrt(m) = sqrt(2)*sqrt(101)/sqrt(15.96) = 3.558 m/s
L = (v1**2 - v2**2)/(2*g*mu) = (-3.558**2 + 6.654**2)/(2*0.1293*9.81) = 12.46 m
In km: L = 0.01246 km
S_K2 = -v2**2/(2*L*g*mu) = -3.558**2/(2*0.1293*12.46*9.81) = -0.4
S_mu = -1
<steps>
v2 = 3.558 m/s
sigma_rel = sqrt((-0.4*0.02)^2 + (-1*0.05)^2) = 0.051
</steps>
<answer>0.01246 ± 0.00064 km</answer>
<confidence>90%</confidence>
```

**supervised** (masked: [])

```
SI values: K2 = 0.10100 kJ = 101 J; m = 15.958 kg; mu = 0.12930; v1 = 6.6539 m/s.
v2 = sqrt(2)*sqrt(K2)/sqrt(m) = sqrt(2)*sqrt(101)/sqrt(15.96) = 3.558 m/s
L = (v1**2 - v2**2)/(2*g*mu) = (-3.558**2 + 6.654**2)/(2*0.1293*9.81) = 12.46 m
In km: L = 0.01246 km
<steps>
v2 = 3.558 m/s
</steps>
<answer>0.01246 ± 0.00063 km</answer>
<confidence>90%</confidence>
```

## seed 60000750: dynamics / depth 4

Problem:

```
On a horizontal floor sits a crate of mass m; mu is the coefficient of kinetic friction between crate and floor. A constant horizontal pull F acts on the initially resting crate for a time t1. During the motion the normal force is N, the kinetic friction force is f_k, the net force is F_net and the acceleration is a1. At the moment the pull is removed, the crate moves with speed v, carries momentum p, and has covered a distance s1. Friction then decelerates it at a rate a2 (magnitude) until it comes to rest after an extra time t2, covering an extra distance s2.

Data:
- The net horizontal force on the crate during the pull, F_net, equals 52.891 N ± 2%.
- s1 = 1.6275 m ± 2% (the distance the crate moves during the pull).
- It is known that a2 = 0.89937 m/s^2 ± 2%, where a2 is the magnitude of the crate's deceleration after the pull.
- It is known that s2 = 219.18 cm ± 5%, where s2 is the distance the crate slides after the pull.
- The height of the crate, h_crate, equals 990.64 mm.
- Use g = 9.81 m/s^2.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

What is the kinetic friction force on the crate, f_k? Give your answer in N. Your <steps> must include the intermediate values v, a1, m.
```

**explicit** (masked: [])

```
SI values: F_net = 52.891 N; a2 = 0.89937 m/s^2; s1 = 1.6275 m; s2 = 219.18 cm = 2.192 m.
Not needed: h_crate.
v = sqrt(2)*sqrt(a2)*sqrt(s2) = sqrt(2)*sqrt(0.8994)*sqrt(2.192) = 1.986 m/s
a1 = v**2/(2*s1) = 1.986**2/(2*1.627) = 1.211 m/s^2
m = F_net/a1 = 52.89/1.211 = 43.67 kg
f_k = a2*m = 0.8994*43.67 = 39.27 N
S_F_net = 1
S_a2 = 0
S_s1 = 1
S_s2 = -1
<steps>
v = 1.986 m/s
a1 = 1.211 m/s^2
m = 43.67 kg
sigma_rel = sqrt((1*0.02)^2 + (0*0.02)^2 + (1*0.02)^2 + (-1*0.05)^2) = 0.057
</steps>
<answer>39.27 ± 2.2 N</answer>
<confidence>40%</confidence>
```

**supervised** (masked: [])

```
SI values: F_net = 52.891 N; a2 = 0.89937 m/s^2; s1 = 1.6275 m; s2 = 219.18 cm = 2.192 m.
Not needed: h_crate.
v = sqrt(2)*sqrt(a2)*sqrt(s2) = sqrt(2)*sqrt(0.8994)*sqrt(2.192) = 1.986 m/s
a1 = v**2/(2*s1) = 1.986**2/(2*1.627) = 1.211 m/s^2
m = F_net/a1 = 52.89/1.211 = 43.67 kg
f_k = a2*m = 0.8994*43.67 = 39.27 N
<steps>
v = 1.986 m/s
a1 = 1.211 m/s^2
m = 43.67 kg
</steps>
<answer>39.27 ± 2.3 N</answer>
<confidence>40%</confidence>
```

## seed 60001125: dynamics / depth 2

Problem:

```
On a horizontal floor sits a crate of mass m; mu is the coefficient of kinetic friction between crate and floor. A constant horizontal pull F acts on the initially resting crate for a time t1. During the motion the normal force is N, the kinetic friction force is f_k, the net force is F_net and the acceleration is a1. At the moment the pull is removed, the crate moves with speed v, carries momentum p, and has covered a distance s1. Friction then decelerates it at a rate a2 (magnitude) until it comes to rest after an extra time t2, covering an extra distance s2.

It is known that mu = 0.22915 ± 5%, where mu is the coefficient of kinetic friction. The time the crate slides after the pull before stopping is t2 = 0.019895 min ± 1%. Use g = 9.81 m/s^2.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

Determine v, the speed of the crate when the pull stops. Express the result in m/s. Report these intermediate quantities in <steps>: a2.
```

**explicit** (masked: [])

```
SI values: mu = 0.22915; t2 = 0.019895 min = 1.194 s.
a2 = g*mu = 0.2291*9.81 = 2.248 m/s^2
v = a2*t2 = 1.194*2.248 = 2.683 m/s
S_mu = 1
S_t2 = 1
<steps>
a2 = 2.248 m/s^2
sigma_rel = sqrt((1*0.05)^2 + (1*0.01)^2) = 0.051
</steps>
<answer>2.683 ± 0.14 m/s</answer>
<confidence>70%</confidence>
```

**supervised** (masked: [])

```
SI values: mu = 0.22915; t2 = 0.019895 min = 1.194 s.
a2 = g*mu = 0.2291*9.81 = 2.248 m/s^2
v = a2*t2 = 1.194*2.248 = 2.683 m/s
<steps>
a2 = 2.248 m/s^2
</steps>
<answer>2.683 ± 0.14 m/s</answer>
<confidence>70%</confidence>
```

## seed 60001500: kinematics / depth 4

Problem:

```
Consider a cart on a straight, horizontal track. Phase 1: the cart has initial speed u and speeds up with constant acceleration a1 during a time t1; it travels s1 and ends the phase with speed v. Phase 2: the cart keeps the constant speed v for a time t2 and travels s2. Phase 3: the cart slows down with constant deceleration of magnitude b and stops after a time t3, having travelled s3 in this phase. For the entire motion, d_tot is the total distance, t_tot the total time, and v_avg the average speed.

Known values:
- The distance covered during phase 1, s1, equals 0.016211 km ± 5%.
- It is known that w_track = 760.69 mm ± 1%, where w_track is the width of the track.
- The mass of the cart, m_cart, equals 34.131 kg ± 1%.
- a1 = 21.761 cm/s^2 ± 5% (the acceleration of the cart during phase 1).
- The total duration of the trip is t_tot = 43.443 s ± 1%.
- It is known that v = 4.5385 m/s ± 2%, where v is the speed of the cart at the end of phase 1.
- t3 = 0.95980 s ± 5% (the braking time in phase 3).

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

What is the distance covered during phase 2, s2? Give your answer in km. Your <steps> must include the intermediate values u, t1, t2.
```

**explicit** (masked: [])

```
SI values: a1 = 21.761 cm/s^2 = 0.2176 m/s^2; s1 = 0.016211 km = 16.21 m; t3 = 0.95980 s; t_tot = 43.443 s; v = 4.5385 m/s.
Not needed: m_cart, w_track.
u = sqrt(-2*a1*s1 + v**2) = sqrt(-2*0.2176*16.21 + 4.538**2) = 3.68 m/s
t1 = (-u + sqrt(2*a1*s1 + u**2))/a1 = (-3.68 + sqrt(2*0.2176*16.21 + 3.68**2))/0.2176 = 3.945 s
t2 = -t1 - t3 + t_tot = -0.9598 - 3.945 + 43.44 = 38.54 s
s2 = t2*v = 38.54*4.538 = 174.9 m
In km: s2 = 0.1749 km
S_a1 = -a1*t1*(2*s1 - t1*u)/(2*t2*u*(a1*t1 + u)) = -0.2176*3.945*(2*16.21 - 3.68*3.945)/(2*3.68*38.54*(0.2176*3.945 + 3.68)) = -0.0119
S_s1 = -s1/(t2*u) = -16.21/(3.68*38.54) = -0.114
S_t3 = -t3/t2 = -0.9598/38.54 = -0.0249
S_t_tot = t_tot/t2 = 43.44/38.54 = 1.13
S_v = (a1*t1*t2*u + t1*v**2 + t2*u**2)/(t2*u*(a1*t1 + u)) = (0.2176*3.68*3.945*38.54 + 3.68**2*38.54 + 3.945*4.538**2)/(3.68*38.54*(0.2176*3.945 + 3.68)) = 1.13
<steps>
u = 3.68 m/s
t1 = 3.945 s
t2 = 38.54 s
sigma_rel = sqrt((-0.0119*0.05)^2 + (-0.114*0.05)^2 + (-0.0249*0.05)^2 + (1.13*0.01)^2 + (1.13*0.02)^2) = 0.026
</steps>
<answer>0.1749 ± 0.0045 km</answer>
<confidence>30%</confidence>
```

**supervised** (masked: [])

```
SI values: a1 = 21.761 cm/s^2 = 0.2176 m/s^2; s1 = 0.016211 km = 16.21 m; t3 = 0.95980 s; t_tot = 43.443 s; v = 4.5385 m/s.
Not needed: m_cart, w_track.
u = sqrt(-2*a1*s1 + v**2) = sqrt(-2*0.2176*16.21 + 4.538**2) = 3.68 m/s
t1 = (-u + sqrt(2*a1*s1 + u**2))/a1 = (-3.68 + sqrt(2*0.2176*16.21 + 3.68**2))/0.2176 = 3.945 s
t2 = -t1 - t3 + t_tot = -0.9598 - 3.945 + 43.44 = 38.54 s
s2 = t2*v = 38.54*4.538 = 174.9 m
In km: s2 = 0.1749 km
<steps>
u = 3.68 m/s
t1 = 3.945 s
t2 = 38.54 s
</steps>
<answer>0.1749 ± 0.0045 km</answer>
<confidence>30%</confidence>
```

## seed 60001875: kinematics / depth 1

Problem:

```
A cart moves along a straight, level track in three phases. In phase 1 it starts with speed u and accelerates uniformly with acceleration a1 for a time t1, covering a distance s1 and reaching speed v. In phase 2 it moves at the constant speed v for a time t2, covering a distance s2. In phase 3 it brakes with a constant deceleration of magnitude b until it comes to rest, which takes a time t3 and covers a distance s3. Over the whole trip the cart covers a total distance d_tot in a total time t_tot, so its average speed is v_avg.

Data:
- It is known that b = 191.86 cm/s^2 ± 1%, where b is the magnitude of the braking deceleration in phase 3.
- The braking distance in phase 3, s3, equals 11.993 m ± 1%.
- It is known that w_track = 548.25 mm, where w_track is the width of the track.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

Find the speed of the cart at the end of phase 1, v, in cm/s. Show your calculation first; no intermediate values need to be listed in <steps>.
```

**explicit** (masked: [])

```
SI values: b = 191.86 cm/s^2 = 1.919 m/s^2; s3 = 11.993 m.
Not needed: w_track.
v = sqrt(2)*sqrt(b)*sqrt(s3) = sqrt(2)*sqrt(1.919)*sqrt(11.99) = 6.784 m/s
In cm/s: v = 678.4 cm/s
S_b = 0.5
S_s3 = 0.5
<steps>
sigma_rel = sqrt((0.5*0.01)^2 + (0.5*0.01)^2) = 0.0071
</steps>
<answer>678.4 ± 4.8 cm/s</answer>
<confidence>90%</confidence>
```

**supervised** (masked: [])

```
SI values: b = 191.86 cm/s^2 = 1.919 m/s^2; s3 = 11.993 m.
Not needed: w_track.
v = sqrt(2)*sqrt(b)*sqrt(s3) = sqrt(2)*sqrt(1.919)*sqrt(11.99) = 6.784 m/s
In cm/s: v = 678.4 cm/s
<steps>
</steps>
<answer>678.4 ± 4.8 cm/s</answer>
<confidence>90%</confidence>
```

## seed 60002250: energy / depth 3

Problem:

```
A block of mass m is released from rest at height h on a frictionless ramp. At the top, its gravitational potential energy relative to the bottom of the ramp is U. It slides down and reaches the bottom with kinetic energy K1 and speed v1. It then crosses a rough horizontal patch of length L where the coefficient of kinetic friction is mu; friction does work of magnitude W_f on the block, which leaves the patch with kinetic energy K2 and speed v2. Finally, on a frictionless floor, the block runs into a light horizontal spring with spring constant k and compresses it by a maximum distance x, at which moment the spring force on the block has magnitude F_s.

Data:
- U = 0.011205 kJ ± 5% (the gravitational potential energy of the block at the top).
- m = 116.15 g (the mass of the block).
- The side length of the block is side = 11.378 cm ± 5%.
- The magnitude of the work done by friction on the rough patch, W_f, equals 2.5421 J ± 1%.
- Use g = 9.81 m/s^2.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

What is the speed of the block after the rough patch, v2? Give your answer in cm/s. Your <steps> must include the intermediate values K1, K2.
```

**explicit** (masked: [])

```
SI values: U = 0.011205 kJ = 11.21 J; W_f = 2.5421 J; m = 116.15 g = 0.1162 kg.
Not needed: side.
K1 = U = 11.21 = 11.21 J
K2 = K1 - W_f = 11.21 - 2.542 = 8.663 J
v2 = sqrt(2)*sqrt(K2)/sqrt(m) = sqrt(2)*sqrt(8.663)/sqrt(0.1162) = 12.21 m/s
In cm/s: v2 = 1221 cm/s
S_U = K1/(2*K2) = 11.21/(2*8.663) = 0.647
S_W_f = -W_f/(2*K2) = -2.542/(2*8.663) = -0.147
<steps>
K1 = 11.21 J
K2 = 8.663 J
sigma_rel = sqrt((0.647*0.05)^2 + (-0.147*0.01)^2) = 0.032
</steps>
<answer>1221 ± 39 cm/s</answer>
<confidence>40%</confidence>
```

**supervised** (masked: [])

```
SI values: U = 0.011205 kJ = 11.21 J; W_f = 2.5421 J; m = 116.15 g = 0.1162 kg.
Not needed: side.
K1 = U = 11.21 = 11.21 J
K2 = K1 - W_f = 11.21 - 2.542 = 8.663 J
v2 = sqrt(2)*sqrt(K2)/sqrt(m) = sqrt(2)*sqrt(8.663)/sqrt(0.1162) = 12.21 m/s
In cm/s: v2 = 1221 cm/s
<steps>
K1 = 11.21 J
K2 = 8.663 J
</steps>
<answer>1221 ± 40 cm/s</answer>
<confidence>40%</confidence>
```

## seed 60002625: energy / depth 1

Problem:

```
A block of mass m is released from rest at height h on a frictionless ramp. At the top, its gravitational potential energy relative to the bottom of the ramp is U. It slides down and reaches the bottom with kinetic energy K1 and speed v1. It then crosses a rough horizontal patch of length L where the coefficient of kinetic friction is mu; friction does work of magnitude W_f on the block, which leaves the patch with kinetic energy K2 and speed v2. Finally, on a frictionless floor, the block runs into a light horizontal spring with spring constant k and compresses it by a maximum distance x, at which moment the spring force on the block has magnitude F_s.

It is known that K1 = 1.5745 J ± 2%, where K1 is the kinetic energy of the block at the bottom of the ramp. The kinetic energy of the block after the rough patch is K2 = 1.0035 J ± 5%. Use g = 9.81 m/s^2.

Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) of that measurement. Values without ± are exact.

Find the magnitude of the work done by friction on the rough patch, W_f, in mJ. Show your calculation first; no intermediate values need to be listed in <steps>.
```

**explicit** (masked: [])

```
SI values: K1 = 1.5745 J; K2 = 1.0035 J.
W_f = K1 - K2 = -1.004 + 1.575 = 0.571 J
In mJ: W_f = 571 mJ
S_K1 = K1/W_f = 1.575/0.571 = 2.76
S_K2 = -K2/W_f = -1.004/0.571 = -1.76
<steps>
sigma_rel = sqrt((2.76*0.02)^2 + (-1.76*0.05)^2) = 0.1
</steps>
<answer>571 ± 57 mJ</answer>
<confidence>70%</confidence>
```

**supervised** (masked: [])

```
SI values: K1 = 1.5745 J; K2 = 1.0035 J.
W_f = K1 - K2 = -1.004 + 1.575 = 0.571 J
In mJ: W_f = 571 mJ
<steps>
</steps>
<answer>571 ± 59 mJ</answer>
<confidence>70%</confidence>
```
