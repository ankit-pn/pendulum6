"""ctypes wrapper around the exact C physics in pendulum6.h (via libpendulum6.so).

Used for pre-training feasibility checks (LQR-in-the-loop) and post-training
eval/rendering of the learned policy against the true environment.
"""
import ctypes
import os

N_LINKS = 6
OBS_SIZE = 2 + 3 * N_LINKS
MAX_STEPS = 1000

_DIR = os.path.dirname(os.path.abspath(__file__))


class _Log(ctypes.Structure):
    _fields_ = [(f, ctypes.c_float) for f in (
        "perf", "score", "episode_return", "episode_length",
        "x_threshold_termination", "angle_termination",
        "max_steps_termination", "n")]


class _Pendulum6(ctypes.Structure):
    _fields_ = [
        ("observations", ctypes.POINTER(ctypes.c_float)),
        ("actions", ctypes.POINTER(ctypes.c_float)),
        ("rewards", ctypes.POINTER(ctypes.c_float)),
        ("terminals", ctypes.POINTER(ctypes.c_float)),
        ("num_agents", ctypes.c_int),
        ("rng", ctypes.c_uint),
        ("log", _Log),
        ("x", ctypes.c_float),
        ("x_dot", ctypes.c_float),
        ("theta", ctypes.c_float * N_LINKS),
        ("theta_dot", ctypes.c_float * N_LINKS),
        ("tick", ctypes.c_int),
        ("episode_return", ctypes.c_float),
        ("cart_mass", ctypes.c_float),
        ("link_mass", ctypes.c_float),
        ("link_length", ctypes.c_float),
        ("gravity", ctypes.c_float),
        ("force_mag", ctypes.c_float),
        ("dt", ctypes.c_float),
        ("substeps", ctypes.c_int),
        ("damping", ctypes.c_float),
        ("spring_k", ctypes.c_float),
        ("theta_fail", ctypes.c_float),
        ("x_threshold", ctypes.c_float),
        ("theta_init", ctypes.c_float),
    ]


DEFAULTS = dict(cart_mass=1.0, link_mass=0.1, link_length=0.3, gravity=9.8,
                force_mag=30.0, dt=0.02, substeps=4, damping=0.1, spring_k=2.0,
                theta_fail=0.5, x_threshold=3.0, theta_init=0.0873)


class Pendulum6Env:
    def __init__(self, seed=0, **kwargs):
        self._lib = ctypes.CDLL(os.path.join(_DIR, "libpendulum6.so"))
        for fn in ("c_reset", "c_step"):
            getattr(self._lib, fn).argtypes = [ctypes.POINTER(_Pendulum6)]
            getattr(self._lib, fn).restype = None
        self._obs = (ctypes.c_float * OBS_SIZE)()
        self._act = (ctypes.c_float * 1)()
        self._rew = (ctypes.c_float * 1)()
        self._term = (ctypes.c_float * 1)()
        self.env = _Pendulum6()
        self.env.observations = self._obs
        self.env.actions = self._act
        self.env.rewards = self._rew
        self.env.terminals = self._term
        self.env.num_agents = 1
        self.env.rng = seed
        params = {**DEFAULTS, **kwargs}
        for k, v in params.items():
            setattr(self.env, k, int(v) if k == "substeps" else float(v))

    def reset(self):
        self._lib.c_reset(ctypes.byref(self.env))
        return list(self._obs)

    def step(self, action):
        """action in [-1, 1]. Returns (obs, reward, terminal, tick)."""
        self._act[0] = float(action)
        self._lib.c_step(ctypes.byref(self.env))
        return list(self._obs), self._rew[0], self._term[0] > 0.5, self.env.tick

    @property
    def state(self):
        """Full state [x, theta_1..N, x_dot, omega_1..N] (for LQR / rendering)."""
        e = self.env
        return [e.x] + list(e.theta) + [e.x_dot] + list(e.theta_dot)
