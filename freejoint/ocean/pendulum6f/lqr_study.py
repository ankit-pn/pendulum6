"""Feasibility study for the FREE-JOINT rod chain: what hold time is achievable?

Linearizes the rod-chain model about upright, computes a discrete LQR, then
measures hold-time distributions on the EXACT nonlinear C physics (ctypes)
under actuation noise. This calibrates the "solved" bar before any training.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from p6f_env import Pendulum6FEnv, DEFAULTS, N_LINKS

N = N_LINKS
P = DEFAULTS


def build_linear(damping):
    m0, m, l, g = P["cart_mass"], P["link_mass"], P["link_length"], P["gravity"]
    bb = np.zeros(N + 1)
    for i in range(1, N + 1):
        bb[i] = m * l * (N - i + 0.5)
    M = np.zeros((N + 1, N + 1))
    M[0, 0] = m0 + N * m
    for i in range(1, N + 1):
        M[0, i] = M[i, 0] = bb[i]
        for k in range(1, N + 1):
            if i == k:
                M[i, i] = m * l * l * (N - i + 1.0 / 3.0)
            else:
                M[i, k] = m * l * l * (N - max(i, k) + 0.5)
    G = np.zeros((N + 1, N + 1))
    for i in range(1, N + 1):
        G[i, i] = g * bb[i]
    D = np.zeros((N + 1, N + 1))
    for j in range(1, N + 1):
        D[j, j] += damping
        if j > 1: D[j, j - 1] -= damping
        if j < N:
            D[j, j] += damping
            D[j, j + 1] -= damping
    Bf = np.zeros((N + 1, 1)); Bf[0, 0] = 1.0
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((N + 1, N + 1)), np.eye(N + 1)],
                  [Minv @ G, -Minv @ D]])
    B = np.vstack([np.zeros((N + 1, 1)), Minv @ Bf])
    return A, B


def lqr_gain(A, B, dt):
    n = A.shape[0]
    Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A * dt; Mx[:n, n:] = B * dt
    Ed = expm(Mx)
    Ad, Bd = Ed[:n, :n], Ed[:n, n:]
    Pm = solve_discrete_are(Ad, Bd, np.eye(n), np.array([[1.0]]))
    K = np.linalg.solve(np.array([[1.0]]) + Bd.T @ Pm @ Bd, Bd.T @ Pm @ Ad)
    return K.flatten()


def hold_times(K, fmax, noise_frac, theta0, episodes=50, max_steps=6000, seed=0):
    """LQR on the real nonlinear C env, from theta0-scale perturbations."""
    rng = np.random.default_rng(seed)
    times = []
    for ep in range(episodes):
        env = Pendulum6FEnv(seed=10000 + ep)
        env.reset()
        env.set_state(
            rng.uniform(-0.02, 0.02), 0.0,
            rng.uniform(-theta0, theta0, N), rng.uniform(-0.02, 0.02, N))
        held = 0
        for t in range(max_steps):
            z = np.array(env.state)
            u = -K @ z + rng.normal(0, noise_frac * fmax)
            env.step(np.clip(u / fmax, -1, 1))
            if env.height01 > 0.95:
                held = t + 1
            else:
                break
        times.append(held * P["dt"])
    return np.array(times)


def main():
    A, B = build_linear(P["damping"])
    lam = max(np.linalg.eigvals(A).real)
    print(f"rod chain, fastest unstable mode: {lam:.1f}/s "
          f"(point-mass model was 22.9/s at same geometry)")

    K = lqr_gain(A, B, P["dt"])
    print(f"LQR gain magnitude: {np.abs(K).max():.0f}")

    fmax = P["force_mag"]
    for noise in (0.0, 0.01, 0.02, 0.05):
        t = hold_times(K, fmax, noise, theta0=0.02)
        print(f"noise {noise*100:4.1f}% of fmax: hold time "
              f"median {np.median(t):6.1f}s  p10 {np.percentile(t, 10):6.1f}s  "
              f"max {t.max():6.1f}s  (60s cap)")


if __name__ == "__main__":
    main()
