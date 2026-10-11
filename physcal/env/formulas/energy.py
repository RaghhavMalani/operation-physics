"""Energy: a block slides down a frictionless ramp, crosses a rough patch, and compresses a spring."""

from physcal.env.constants import G
from physcal.env.graph import Family, Formula, Quantity, log_uniform

INTROS = (
    "A block of mass m is released from rest at height h on a frictionless ramp. At the top, its "
    "gravitational potential energy relative to the bottom of the ramp is U. It slides down and reaches the "
    "bottom with kinetic energy K1 and speed v1. It then crosses a rough horizontal patch of length L where "
    "the coefficient of kinetic friction is mu; friction does work of magnitude W_f on the block, which leaves "
    "the patch with kinetic energy K2 and speed v2. Finally, on a frictionless floor, the block runs into a "
    "light horizontal spring with spring constant k and compresses it by a maximum distance x, at which "
    "moment the spring force on the block has magnitude F_s.",
    "Starting from rest at a height h, a block of mass m slides down a frictionless ramp; U denotes its "
    "potential energy at the top relative to the bottom. At the bottom it has kinetic energy K1 and speed "
    "v1. Next it slides across a rough level section of length L (coefficient of kinetic friction mu), "
    "losing energy W_f to friction, and exits with kinetic energy K2 and speed v2. On the frictionless floor "
    "beyond, it hits a light spring of spring constant k, which it compresses by at most x; at that maximum "
    "compression the spring exerts a force of magnitude F_s on the block.",
)

QUANTITIES = (
    Quantity("m", "mass of the block", "kg", (0.1, 20.0)),
    Quantity("h", "starting height of the block", "m", (0.5, 20.0)),
    Quantity("U", "gravitational potential energy of the block at the top", "J"),
    Quantity("K1", "kinetic energy of the block at the bottom of the ramp", "J"),
    Quantity("v1", "speed of the block at the bottom of the ramp", "m/s"),
    Quantity("L", "length of the rough patch", "m", bounds_si=(0.1, 50.0)),
    Quantity("mu", "coefficient of kinetic friction on the rough patch", "", (0.05, 0.8)),
    Quantity("W_f", "magnitude of the work done by friction on the rough patch", "J"),
    Quantity("K2", "kinetic energy of the block after the rough patch", "J"),
    Quantity("v2", "speed of the block after the rough patch", "m/s"),
    Quantity("k", "spring constant", "N/m", (100.0, 1.0e5)),
    Quantity("x", "maximum compression of the spring", "m", bounds_si=(0.005, 1.0)),
    Quantity("F_s", "magnitude of the spring force at maximum compression", "N"),
)

FORMULAS = (
    Formula("E1", "U = m*g*h", "U m h"),
    Formula("E2", "K1 = U", "K1 U"),
    Formula("E3", "K1 = m*v1**2/2", "K1 m v1"),
    Formula("E4", "v1**2 = 2*g*h", "v1 h"),
    Formula("E5", "W_f = mu*m*g*L", "W_f mu m L"),
    Formula("E6", "K2 = K1 - W_f", "K2 K1 W_f"),
    Formula("E7", "K2 = m*v2**2/2", "K2 m v2"),
    Formula("E8", "v2**2 = v1**2 - 2*mu*g*L", "v2 v1 mu L"),
    Formula("E9", "K2 = k*x**2/2", "K2 k x"),
    Formula("E10", "F_s = k*x", "F_s k x"),
)

DISTRACTORS = (
    Quantity("side", "side length of the block", "m", (0.02, 0.3)),
    Quantity("len_spring", "natural length of the spring", "m", (0.1, 1.0)),
    Quantity("P_air", "air pressure in the room", "Pa", (9.5e4, 1.05e5)),
)

# Friction must not stop the block on the patch: L = r * h / mu, so W_f / U = r, log-uniform in this range.
FRICTION_LOSS_FRACTION = (0.1, 0.8)


def _sample_basis(rng, quantities):
    m = log_uniform(rng, *quantities["m"].range_si)
    h = log_uniform(rng, *quantities["h"].range_si)
    mu = log_uniform(rng, *quantities["mu"].range_si)
    k = log_uniform(rng, *quantities["k"].range_si)
    L = log_uniform(rng, *FRICTION_LOSS_FRACTION) * h / mu
    return {"m": m, "h": h, "mu": mu, "k": k, "L": L}


def build() -> Family:
    return Family(
        name="energy",
        intros=INTROS,
        quantities=QUANTITIES,
        formulas=FORMULAS,
        basis=("m", "h", "mu", "k", "L"),
        sample_basis=_sample_basis,
        constants={"g": (G, "m/s^2")},
        distractors=DISTRACTORS,
    )
