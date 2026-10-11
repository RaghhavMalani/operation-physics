"""Dynamics: a crate pulled from rest across a rough floor, then released to slide to a stop."""

from physcal.env.graph import Family, Formula, Quantity, log_uniform
from physcal.env.constants import G

INTROS = (
    "A crate of mass m rests on a level floor, and the coefficient of kinetic friction between the crate and "
    "the floor is mu. Starting from rest, the crate is pulled by a constant horizontal force F for a time t1. "
    "While it slides, the floor exerts a normal force N and a kinetic friction force f_k on it; the net "
    "horizontal force on the crate is F_net and its acceleration is a1. When the pull stops, the crate has "
    "speed v and momentum p, and it has moved a distance s1. Afterwards it slides freely, slowing down under "
    "friction alone with a deceleration of magnitude a2, and stops after a further time t2 and distance s2.",
    "On a horizontal floor sits a crate of mass m; mu is the coefficient of kinetic friction between crate "
    "and floor. A constant horizontal pull F acts on the initially resting crate for a time t1. During the "
    "motion the normal force is N, the kinetic friction force is f_k, the net force is F_net and the "
    "acceleration is a1. At the moment the pull is removed, the crate moves with speed v, carries momentum "
    "p, and has covered a distance s1. Friction then decelerates it at a rate a2 (magnitude) until it comes "
    "to rest after an extra time t2, covering an extra distance s2.",
)

QUANTITIES = (
    Quantity("m", "mass of the crate", "kg", (0.5, 200.0)),
    Quantity("mu", "coefficient of kinetic friction", "", (0.05, 0.8)),
    Quantity("F", "magnitude of the applied pulling force", "N"),
    Quantity("t1", "duration of the pull", "s", (1.0, 10.0)),
    Quantity("N", "normal force on the crate", "N"),
    Quantity("f_k", "kinetic friction force on the crate", "N"),
    Quantity("F_net", "net horizontal force on the crate during the pull", "N"),
    Quantity("a1", "acceleration of the crate during the pull", "m/s^2"),
    Quantity("v", "speed of the crate when the pull stops", "m/s", bounds_si=(0.2, 25.0)),
    Quantity("p", "momentum of the crate when the pull stops", "kg m/s"),
    Quantity("s1", "distance the crate moves during the pull", "m", bounds_si=(0.05, 300.0)),
    Quantity("a2", "magnitude of the crate's deceleration after the pull", "m/s^2"),
    Quantity("t2", "time the crate slides after the pull before stopping", "s"),
    Quantity("s2", "distance the crate slides after the pull", "m", bounds_si=(0.05, 300.0)),
)

FORMULAS = (
    Formula("D1", "N = m*g", "N m"),
    Formula("D2", "f_k = mu*N", "f_k mu N"),
    Formula("D3", "F_net = F - f_k", "F_net F f_k"),
    Formula("D4", "F_net = m*a1", "F_net m a1"),
    Formula("D5", "v = a1*t1", "v a1 t1"),
    Formula("D6", "s1 = a1*t1**2/2", "s1 a1 t1"),
    Formula("D7", "v**2 = 2*a1*s1", "v a1 s1"),
    Formula("D8", "s1 = v*t1/2", "s1 v t1"),
    Formula("D9", "p = m*v", "p m v"),
    Formula("D10", "F_net*t1 = p", "F_net t1 p"),
    Formula("D11", "a2 = mu*g", "a2 mu"),
    Formula("D12", "f_k = m*a2", "f_k m a2"),
    Formula("D13", "v = a2*t2", "v a2 t2"),
    Formula("D14", "v**2 = 2*a2*s2", "v a2 s2"),
    Formula("D15", "s2 = v*t2/2", "s2 v t2"),
)

DISTRACTORS = (
    Quantity("h_crate", "height of the crate", "m", (0.2, 2.0)),
    Quantity("len_rope", "length of the rope used to pull the crate", "m", (0.5, 10.0)),
    Quantity("P_air", "air pressure in the room", "Pa", (9.5e4, 1.05e5)),
)

# The pull must beat kinetic friction: F = r * mu * m * g with r log-uniform in this range.
PULL_TO_FRICTION_RATIO = (1.2, 4.0)


def _sample_basis(rng, quantities):
    m = log_uniform(rng, *quantities["m"].range_si)
    mu = log_uniform(rng, *quantities["mu"].range_si)
    t1 = log_uniform(rng, *quantities["t1"].range_si)
    F = log_uniform(rng, *PULL_TO_FRICTION_RATIO) * mu * m * G
    return {"m": m, "mu": mu, "F": F, "t1": t1}


def build() -> Family:
    return Family(
        name="dynamics",
        intros=INTROS,
        quantities=QUANTITIES,
        formulas=FORMULAS,
        basis=("m", "mu", "F", "t1"),
        sample_basis=_sample_basis,
        constants={"g": (G, "m/s^2")},
        distractors=DISTRACTORS,
    )
