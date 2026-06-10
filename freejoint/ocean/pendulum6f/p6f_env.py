"""ctypes wrapper around the exact C physics in pendulum6f.h (libpendulum6f.so)."""
import ctypes
import os

N_LINKS = 6
OBS_SIZE = 2 + 3 * N_LINKS
MIN_STEPS = 1600
MAX_STEPS = 4000

_DIR = os.path.dirname(os.path.abspath(__file__))


class _Log(ctypes.Structure):
    _fields_ = [(f, ctypes.c_float) for f in (
        "perf", "score", "episode_return", "episode_length", "hold_time",
        "curriculum_p", "x_threshold_termination", "timeout_termination", "n")]


class _Pendulum6F(ctypes.Structure):
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
        ("max_steps_ep", ctypes.c_int),
        ("episode_return", ctypes.c_float),
        ("hold_steps", ctypes.c_int),
        ("max_hold_steps", ctypes.c_int),
        ("curriculum", ctypes.c_float),
        ("cart_mass", ctypes.c_float),
        ("link_mass", ctypes.c_float),
        ("link_length", ctypes.c_float),
        ("gravity", ctypes.c_float),
        ("force_mag", ctypes.c_float),
        ("dt", ctypes.c_float),
        ("substeps", ctypes.c_int),
        ("damping", ctypes.c_float),
        ("spring_k", ctypes.c_float),
        ("x_threshold", ctypes.c_float),
        ("hold_target", ctypes.c_int),
        ("eval_mode", ctypes.c_int),
    ]


DEFAULTS = dict(cart_mass=1.0, link_mass=0.2, link_length=1.0, gravity=9.8,
                force_mag=120.0, dt=0.005, substeps=3, damping=0.01, spring_k=0.0,
                x_threshold=4.0, hold_target=75, eval_mode=0)


class Pendulum6FEnv:
    def __init__(self, seed=0, **kwargs):
        self._lib = ctypes.CDLL(os.path.join(_DIR, "libpendulum6f.so"))
        for fn in ("c_reset", "c_step", "init"):
            getattr(self._lib, fn).argtypes = [ctypes.POINTER(_Pendulum6F)]
            getattr(self._lib, fn).restype = None
        self._obs = (ctypes.c_float * OBS_SIZE)()
        self._act = (ctypes.c_float * 1)()
        self._rew = (ctypes.c_float * 1)()
        self._term = (ctypes.c_float * 1)()
        self.env = _Pendulum6F()
        self.env.observations = self._obs
        self.env.actions = self._act
        self.env.rewards = self._rew
        self.env.terminals = self._term
        self.env.num_agents = 1
        self.env.rng = seed
        params = {**DEFAULTS, **kwargs}
        for k, v in params.items():
            setattr(self.env, k,
                    int(v) if k in ("substeps", "hold_target", "eval_mode") else float(v))
        self._lib.init(ctypes.byref(self.env))

    def reset(self):
        self._lib.c_reset(ctypes.byref(self.env))
        return list(self._obs)

    def step(self, action):
        """action in [-1, 1]. Returns (obs, reward, terminal, tick)."""
        self._act[0] = float(action)
        self._lib.c_step(ctypes.byref(self.env))
        return list(self._obs), self._rew[0], self._term[0] > 0.5, self.env.tick

    def set_state(self, x, x_dot, theta, theta_dot):
        self.env.x = x
        self.env.x_dot = x_dot
        for j in range(N_LINKS):
            self.env.theta[j] = theta[j]
            self.env.theta_dot[j] = theta_dot[j]

    @property
    def state(self):
        e = self.env
        return [e.x] + list(e.theta) + [e.x_dot] + list(e.theta_dot)

    @property
    def height01(self):
        import math
        return 0.5 * (sum(math.cos(t) for t in self.env.theta) / N_LINKS + 1.0)
