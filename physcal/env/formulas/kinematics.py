"""Kinematics: a cart that accelerates, cruises, then brakes to rest on a straight track."""

from physcal.env.graph import Family, Formula, Quantity

INTROS = (
    "A cart moves along a straight, level track in three phases. In phase 1 it starts with speed u and "
    "accelerates uniformly with acceleration a1 for a time t1, covering a distance s1 and reaching speed v. "
    "In phase 2 it moves at the constant speed v for a time t2, covering a distance s2. In phase 3 it brakes "
    "with a constant deceleration of magnitude b until it comes to rest, which takes a time t3 and covers a "
    "distance s3. Over the whole trip the cart covers a total distance d_tot in a total time t_tot, so its "
    "average speed is v_avg.",
    "Consider a cart on a straight, horizontal track. Phase 1: the cart has initial speed u and speeds up "
    "with constant acceleration a1 during a time t1; it travels s1 and ends the phase with speed v. Phase 2: "
    "the cart keeps the constant speed v for a time t2 and travels s2. Phase 3: the cart slows down with "
    "constant deceleration of magnitude b and stops after a time t3, having travelled s3 in this phase. "
    "For the entire motion, d_tot is the total distance, t_tot the total time, and v_avg the average speed.",
)

QUANTITIES = (
    Quantity("u", "speed of the cart at the start of phase 1", "m/s", (0.5, 15.0)),
    Quantity("a1", "acceleration of the cart during phase 1", "m/s^2", (0.2, 3.0)),
    Quantity("t1", "duration of phase 1", "s", (2.0, 20.0)),
    Quantity("v", "speed of the cart at the end of phase 1", "m/s", bounds_si=(1.0, 40.0)),
    Quantity("s1", "distance covered during phase 1", "m"),
    Quantity("t2", "duration of phase 2", "s", (5.0, 120.0)),
    Quantity("s2", "distance covered during phase 2", "m"),
    Quantity("b", "magnitude of the braking deceleration in phase 3", "m/s^2", (0.5, 8.0)),
    Quantity("t3", "braking time in phase 3", "s"),
    Quantity("s3", "braking distance in phase 3", "m"),
    Quantity("d_tot", "total distance covered by the cart", "m"),
    Quantity("t_tot", "total duration of the trip", "s"),
    Quantity("v_avg", "average speed of the cart over the whole trip", "m/s"),
)

FORMULAS = (
    Formula("K1", "v = u + a1*t1", "v u a1 t1"),
    Formula("K2", "s1 = (u + v)*t1/2", "s1 u v t1"),
    Formula("K3", "v**2 = u**2 + 2*a1*s1", "v u a1 s1"),
    Formula("K4", "s1 = u*t1 + a1*t1**2/2", "s1 u a1 t1"),
    Formula("K5", "s1 = v*t1 - a1*t1**2/2", "s1 v a1"),
    Formula("K6", "s2 = v*t2", "s2 v t2"),
    Formula("K7", "v = b*t3", "v b t3"),
    Formula("K8", "v**2 = 2*b*s3", "v b s3"),
    Formula("K9", "s3 = v*t3/2", "s3 v t3"),
    Formula("K10", "d_tot = s1 + s2 + s3", "d_tot s1 s2 s3"),
    Formula("K11", "t_tot = t1 + t2 + t3", "t_tot t1 t2 t3"),
    Formula("K12", "v_avg = d_tot/t_tot", "v_avg d_tot t_tot"),
)

DISTRACTORS = (
    Quantity("m_cart", "mass of the cart", "kg", (5.0, 500.0)),
    Quantity("len_cart", "length of the cart", "m", (0.3, 3.0)),
    Quantity("w_track", "width of the track", "m", (0.5, 3.0)),
)


def build() -> Family:
    return Family(
        name="kinematics",
        intros=INTROS,
        quantities=QUANTITIES,
        formulas=FORMULAS,
        basis=("u", "a1", "t1", "t2", "b"),
        distractors=DISTRACTORS,
    )
