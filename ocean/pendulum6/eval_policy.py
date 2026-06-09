"""Evaluate a PufferLib CUDA-backend checkpoint (flat fp32 .bin) on the real
C physics via ctypes. Reconstructs the policy in torch:

  flat layout (registration order, models.cu:841-843 — encoder, DECODER, network):
      encoder W {H, obs} -> decoder W {act+1, H} (act mean rows + value row)
      -> logstd {act} -> [alignment padding] -> MinGRU layers W {3H, H} each

  Gotcha (kernels.cu alloc_create): each tensor is aligned to 16 BYTES in the
  bf16 param buffer, but the fp32 master/checkpoint maps onto it LINEARLY by
  element index. The 1-float logstd therefore leaves a 7-element padding gap
  before the GRU weights, and the final 7 GRU weights fall past the end of the
  master buffer (they stay at their cudaMemset value, zero, forever).

All linear ops are bias-free matmuls (verified against src/models.cu).
MinGRU eval semantics mirror pufferlib/models.py MinGRU.forward_eval.

Usage: python eval_policy.py CHECKPOINT.bin [--episodes 100] [--hidden 256]
       [--layers 2] [--stochastic] [--dump-traj traj.npz]
"""
import argparse
import numpy as np
import torch

from p6_env import Pendulum6Env, OBS_SIZE, MAX_STEPS

ACT = 1


def load_flat(path, hidden, layers):
    flat = np.fromfile(path, dtype=np.float32)
    sizes = {
        "encoder": hidden * OBS_SIZE,
        "gru": layers * 3 * hidden * hidden,
        "decoder": (ACT + 1) * hidden,
        "logstd": ACT,
    }
    total = sum(sizes.values())
    if flat.size != total:
        raise SystemExit(
            f"size mismatch: file has {flat.size} floats, expected {total} "
            f"(hidden={hidden} layers={layers} obs={OBS_SIZE}). "
            f"Try other --hidden/--layers.")
    ALIGN = 8  # 16 bytes / 2-byte bf16 elements

    def aligned(x):
        return (x + ALIGN - 1) // ALIGN * ALIGN

    out, off = {}, 0
    out["encoder"] = flat[off:off + sizes["encoder"]].reshape(hidden, OBS_SIZE); off += sizes["encoder"]
    off = aligned(off)
    out["decoder"] = flat[off:off + sizes["decoder"]].reshape(ACT + 1, hidden); off += sizes["decoder"]
    off = aligned(off)
    out["logstd"] = flat[off:off + ACT]; off += ACT
    off = aligned(off)
    out["gru"] = []
    gsz = 3 * hidden * hidden
    for i in range(layers):
        g = flat[off:off + gsz]
        if g.size < gsz:  # tail past master buffer end: zeros, never trained
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
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--dump-traj", default=None,
                    help="save state trajectories of the first 3 episodes")
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()

    w = load_flat(args.checkpoint, args.hidden, args.layers)
    policy = FlatPolicy(w, args.hidden, args.layers)

    lengths, returns, trajs = [], [], []
    with torch.no_grad():
        for ep in range(args.episodes):
            env = Pendulum6Env(seed=args.seed0 + ep)
            obs = env.reset()
            state = policy.initial_state()
            steps, ret = 0, 0.0
            traj = []
            for _ in range(MAX_STEPS):
                if args.dump_traj and ep < 3:
                    traj.append(env.state)
                a, state = policy.act(obs, state, args.stochastic)
                obs, r, terminal, tick = env.step(np.clip(a, -1, 1))
                ret += r
                steps += 1
                if terminal or tick == 0:
                    break
            lengths.append(steps)
            returns.append(ret)
            if args.dump_traj and ep < 3:
                trajs.append(np.array(traj, np.float32))

    lengths = np.array(lengths)
    print(f"episodes:        {args.episodes}")
    print(f"mean ep length:  {lengths.mean():.1f} / {MAX_STEPS}")
    print(f"min ep length:   {lengths.min()}")
    print(f"full episodes:   {(lengths >= MAX_STEPS).sum()}/{args.episodes}")
    print(f"mean return:     {np.mean(returns):.1f}")
    print(f"SOLVED: {'YES' if lengths.mean() >= 950 else 'NO'} "
          f"(threshold: mean >= 950)")

    if args.dump_traj:
        np.savez(args.dump_traj, **{f"ep{i}": t for i, t in enumerate(trajs)})
        print(f"trajectories saved to {args.dump_traj}")


if __name__ == "__main__":
    main()
