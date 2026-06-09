"""LQR-in-the-loop feasibility check against the exact nonlinear C physics.

Builds the linearized model matching pendulum6.h (elastic joints), computes a
discrete LQR gain, and drives the real C env with it for 100 episodes.
If this passes, the task is certifiably solvable by a smooth feedback policy.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from p6_env import Pendulum6Env, DEFAULTS, N_LINKS, MAX_STEPS

N = N_LINKS
P = DEFAULTS


def lqr_gain():
    m0, m, l, g = P["cart_mass"], P["link_mass"], P["link_length"], P["gravity"]
    k, c = P["spring_k"], P["damping"]
    dt = P["dt"]
    mu = np.zeros(N + 1)
    for j in range(1, N + 1):
        mu[j] = m * (N - j + 1)
    M = np.zeros((N + 1, N + 1))
    M[0, 0] = m0 + mu[1]
    for j in range(1, N + 1):
        M[0, j] = M[j, 0] = mu[j] * l
        for kk in range(1, N + 1):
            M[j, kk] = mu[max(j, kk)] * l * l
    Ks = np.zeros((N + 1, N + 1))
    D = np.zeros((N + 1, N + 1))
    for j in range(1, N + 1):
        Ks[j, j] += k; D[j, j] += c          # joint with parent
        if j > 1:
            Ks[j, j - 1] -= k; D[j, j - 1] -= c
        if j < N:
            Ks[j, j] += k; D[j, j] += c      # joint with child
            Ks[j, j + 1] -= k; D[j, j + 1] -= c
    G = np.zeros((N + 1, N + 1))
    for j in range(1, N + 1):
        G[j, j] = mu[j] * g * l
    G -= Ks
    Bf = np.zeros((N + 1, 1)); Bf[0, 0] = 1.0
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((N + 1, N + 1)), np.eye(N + 1)],
                  [Minv @ G, -Minv @ D]])
    B = np.vstack([np.zeros((N + 1, 1)), Minv @ Bf])
    n = 2 * (N + 1)
    Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A * dt; Mx[:n, n:] = B * dt
    Ed = expm(Mx)
    Ad, Bd = Ed[:n, :n], Ed[:n, n:]
    Pm = solve_discrete_are(Ad, Bd, np.eye(n), np.array([[1.0]]))
    K = np.linalg.solve(np.array([[1.0]]) + Bd.T @ Pm @ Bd, Bd.T @ Pm @ Ad)
    return K.flatten()


def main():
    K = lqr_gain()
    fmax = P["force_mag"]
    episodes, solved, lengths = 100, 0, []
    for ep in range(episodes):
        env = Pendulum6Env(seed=1000 + ep)
        env.reset()
        steps = 0
        for _ in range(MAX_STEPS):
            z = np.array(env.state)
            u = float(-K @ z)
            _, _, terminal, tick = env.step(np.clip(u / fmax, -1, 1))
            steps += 1
            if terminal or tick == 0:
                break
        lengths.append(steps)
        if steps >= MAX_STEPS:
            solved += 1
    lengths = np.array(lengths)
    print(f"LQR on real C physics: {solved}/{episodes} full episodes, "
          f"mean length {lengths.mean():.1f}/{MAX_STEPS}, min {lengths.min()}")
    return 0 if solved == episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
