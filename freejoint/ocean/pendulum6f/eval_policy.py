"""Evaluate a PufferLib CUDA-backend checkpoint (flat fp32 .bin) for the
FREE-JOINT swing-up + balance task, on the real C physics via ctypes.

Episodes start fully hanging (eval_mode=1). Reports swing-up success rate,
hold-time distribution, and the curriculum success criterion.

Flat layout: encoder {H,obs} -> decoder {act+1,H} -> logstd {act} -> [16-byte
alignment padding] -> MinGRU layers {3H,H}. See pendulum6 eval_policy.py for
the alignment-gap discovery.

Usage: python eval_policy.py CKPT.bin [--episodes 100] [--hidden 512]
       [--layers 2] [--steps 3000] [--hold-target 300] [--dump-traj t.npz]
"""
import argparse
import numpy as np
import torch

from p6f_env import Pendulum6FEnv, OBS_SIZE

ACT = 1


def load_flat(path, hidden, layers, align=8):
    """align: 16-byte tensor alignment in param-buffer elements.
    8 for default bf16 builds, 4 for --float (fp32) builds."""
    flat = np.fromfile(path, dtype=np.float32)
    ALIGN = align

    def aligned(x):
        return (x + ALIGN - 1) // ALIGN * ALIGN

    out, off = {}, 0
    n_enc = hidden * OBS_SIZE
    out["encoder"] = flat[off:off + n_enc].reshape(hidden, OBS_SIZE); off += n_enc
    off = aligned(off)
    n_dec = (ACT + 1) * hidden
    out["decoder"] = flat[off:off + n_dec].reshape(ACT + 1, hidden); off += n_dec
    off = aligned(off)
    out["logstd"] = flat[off:off + ACT]; off += ACT
    off = aligned(off)
    out["gru"] = []
    gsz = 3 * hidden * hidden
    for i in range(layers):
        g = flat[off:off + gsz]
        if g.size < gsz:
            g = np.concatenate([g, np.zeros(gsz - g.size, np.float32)])
        out["gru"].append(g.reshape(3 * hidden, hidden))
        off = aligned(off + gsz)
    return out


class FlatPolicy:
    def __init__(self, w, hidden, layers):
        self.enc = torch.from_numpy(w["encoder"].copy())
        self.gru = [torch.from_numpy(g.copy()) for g in w["gru"]]
        self.dec = torch.from_numpy(w["decoder"].copy())
        self.logstd = torch.from_numpy(w["logstd"].copy())
        self.hidden, self.layers = hidden, layers

    @staticmethod
    def _g(x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def initial_state(self):
        return torch.zeros(self.layers, self.hidden)

    def act(self, obs, state, stochastic=False):
        h = torch.from_numpy(np.asarray(obs, np.float32)) @ self.enc.T
        new_state = torch.empty_like(state)
        for i in range(self.layers):
            hid, gate, proj = (h @ self.gru[i].T).chunk(3, dim=-1)
            out = torch.lerp(state[i], self._g(hid), gate.sigmoid())
            g = proj.sigmoid()
            h = g * out + (1.0 - g) * h
            new_state[i] = out
        logits = h @ self.dec.T
        mean = logits[:ACT]
        if stochastic:
            a = torch.normal(mean, self.logstd.exp())
        else:
            a = mean
        return float(a[0]), new_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000, help="eval horizon (30 s @200Hz)")
    ap.add_argument("--hold-target", type=int, default=100)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--dump-traj", default=None)
    ap.add_argument("--align", type=int, default=8,
                    help="8 for bf16 builds, 4 for --float builds")
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()

    w = load_flat(args.checkpoint, args.hidden, args.layers, args.align)
    policy = FlatPolicy(w, args.hidden, args.layers)

    swung_up, best_holds, total_balanced, trajs = [], [], [], []
    with torch.no_grad():
        for ep in range(args.episodes):
            env = Pendulum6FEnv(seed=args.seed0 + ep, eval_mode=1,
                                hold_target=args.hold_target)
            obs = env.reset()
            state = policy.initial_state()
            hold, best, balanced, up = 0, 0, 0, False
            traj = []
            for t in range(args.steps):
                if args.dump_traj and ep < 3:
                    traj.append(env.state)
                a, state = policy.act(obs, state, args.stochastic)
                obs, r, terminal, tick = env.step(np.clip(a, -1, 1))
                if env.height01 > 0.95:
                    up = True
                    balanced += 1
                    hold += 1
                    best = max(best, hold)
                else:
                    hold = 0
                if terminal or tick == 0:
                    break
            swung_up.append(up)
            best_holds.append(best * 0.005)
            total_balanced.append(balanced * 0.005)
            if args.dump_traj and ep < 3:
                trajs.append(np.array(traj, np.float32))

    bh = np.array(best_holds)
    tb = np.array(total_balanced)
    succ = (bh >= args.hold_target * 0.005).sum()
    print(f"episodes (hanging start, {args.steps/200:.0f}s horizon): {args.episodes}")
    print(f"swing-up reached top:   {int(np.sum(swung_up))}/{args.episodes}")
    print(f"best hold   median {np.median(bh):6.2f}s  p10 {np.percentile(bh,10):6.2f}s  "
          f"p90 {np.percentile(bh,90):6.2f}s  max {bh.max():6.2f}s")
    print(f"balanced/ep median {np.median(tb):6.2f}s")
    print(f"hold >= {args.hold_target*0.005:.1f}s: {succ}/{args.episodes}")

    if args.dump_traj:
        np.savez(args.dump_traj, **{f"ep{i}": t for i, t in enumerate(trajs)})
        print(f"trajectories saved to {args.dump_traj}")


if __name__ == "__main__":
    main()
