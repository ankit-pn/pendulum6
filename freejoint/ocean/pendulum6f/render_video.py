"""Render dumped pendulum6f trajectories (eval_policy.py --dump-traj) to MP4.

State rows: [x, theta_1..6, x_dot, omega_1..6]. Usage:
    python3 render_video.py traj.npz out.mp4 [stride]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

LINK_LEN = 1.0
N = 6
DT = 0.005

def main():
    npz = np.load(sys.argv[1])
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pendulum6f.mp4"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # 50fps output from 100Hz sim
    episodes = [npz[k] for k in sorted(npz.files)]

    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=110)
    fig.patch.set_facecolor("#0a1818")
    ax.set_facecolor("#0a1818")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.plot([-7.0, 7.0], [0, 0], color="#00bbbb", lw=1.5)
    cart = plt.Rectangle((0, -0.07), 0.44, 0.14, color="#00bbbb")
    ax.add_patch(cart)
    chain, = ax.plot([], [], color="#bb0000", lw=4, solid_capstyle="round",
                     marker="o", markersize=5, markerfacecolor="#f1f1f1",
                     markeredgecolor="none")
    title = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="#f1f1f1",
                    fontsize=12, va="top", family="monospace")

    frames = []
    for ei, ep in enumerate(episodes):
        for t in range(0, len(ep), stride):
            frames.append((ei, t, ep[t]))

    def update(frame):
        ei, t, s = frame
        x = s[0]
        xs, ys = [x], [0.0]
        h = 0.0
        for j in range(N):
            xs.append(xs[-1] + LINK_LEN * np.sin(s[1 + j]))
            ys.append(ys[-1] + LINK_LEN * np.cos(s[1 + j]))
            h += np.cos(s[1 + j])
        h01 = 0.5 * (h / N + 1.0)
        cart.set_xy((x - 0.22, -0.07))
        chain.set_data(xs, ys)
        title.set_text(f"6-link FREE-JOINT swing-up | episode {ei+1} "
                       f"| t = {t * DT:6.2f} s | height {h01:4.2f}")
        return cart, chain, title

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(out_path, writer=animation.FFMpegWriter(fps=50, bitrate=2400))
    print(f"wrote {out_path} ({len(frames)} frames)")

if __name__ == "__main__":
    main()
