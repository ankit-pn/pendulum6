# pendulum6 — 6-Link Inverted Pendulum on a Cart, Solved with RL

A cart balancing a chain of **six** pendulum links using only horizontal force,
trained with [PufferLib 4.0](https://github.com/PufferAI/PufferLib)'s CUDA PPO
backend. 1 billion environment steps in ~6 minutes on a single RTX 4090
(3.4M steps/sec), roughly $0.50 of RunPod GPU time.

![demo](artifacts/pendulum6_balance.mp4)

## Result

| Metric | Value |
|---|---|
| Mean episode length (100 deterministic eval episodes) | **1000.0 / 1000** |
| Min episode length | 1000 |
| Episodes reaching full horizon | 100/100 |
| Success threshold | mean ≥ 950 |

Evaluated against the exact C physics via ctypes (`eval_policy.py`), verified
on the training pod and reproduced independently on a second machine.

## Task

- Cart (1 kg) on a ±3 m rail, six 0.3 m links (0.1 kg each) chained on top.
- Action: continuous cart force, ±30 N at 50 Hz.
- Links start within ±5° of upright; episode ends if any link tips past
  0.5 rad or the cart leaves the rail; horizon 1000 steps (20 s).
- Reward: 1 per step alive, minus small angle/cart-offset penalties.

### Why the joints have weak springs (read this before calling it cheating)

Before training, an LQR feasibility study (`lqr_check.py`, plus saturated
closed-loop simulations against the real nonlinear physics) showed that a
**free-jointed** 6-link chain on a cart is unstabilizable by *any* controller
under practical budgets: required feedback gains are ~10⁵, and even 0.1°
initial offsets with 1000 N force at 500 Hz fail 100/100 — for both serial
and parallel pole configurations. (The physical world record for free joints
is 3 links.) Each extra link multiplies the required gain ~10–30×; joint
*damping* actually makes it worse because friction fights force transmission
up the chain.

Adding weak torsional springs at the joints (k = 2 N·m/rad, with 0.1 relative
damping) keeps the system **actively unstable** — the fastest unstable mode is
λ = 2.9/s and the chain collapses in ~1.3 s without control — but makes it
controllable. LQR certified 100/100 stabilization on the real C physics before
any RL training was attempted; the RL policy then learned the same task
model-free from reward alone.

## Repo layout

```
ocean/pendulum6/pendulum6.h    # the environment: 7-DOF Lagrangian dynamics in C
ocean/pendulum6/binding.c      # PufferLib 4.0 binding (continuous action)
ocean/pendulum6/test_physics.c # gcc-only unit tests (energy conservation etc.)
ocean/pendulum6/lqr_check.py   # feasibility certification (linearize -> DARE -> sim)
ocean/pendulum6/p6_env.py      # ctypes wrapper around the exact C physics
ocean/pendulum6/eval_policy.py # flat-checkpoint loader + 100-episode evaluator
ocean/pendulum6/render_video.py# matplotlib MP4 renderer for dumped trajectories
ocean/pendulum6/p6_shim.c      # builds libpendulum6.so for the ctypes wrapper
ocean/pendulum6/pod_setup.sh   # RunPod pod bootstrap
config/pendulum6.ini           # env params + PPO hyperparameters
artifacts/                     # trained checkpoints, eval video, train log
```

## Reproduce the eval (CPU only, no PufferLib needed)

```bash
cd ocean/pendulum6
gcc -O2 -shared -fPIC -o libpendulum6.so p6_shim.c -lm
python3 eval_policy.py ../../artifacts/checkpoint_393M.bin --episodes 100
```

## Retrain from scratch (needs CUDA toolkit, clang, torch >= 2.9)

```bash
git clone https://github.com/PufferAI/PufferLib && cd PufferLib
cp -r path/to/this/repo/ocean/pendulum6 ocean/
cp path/to/this/repo/config/pendulum6.ini config/
pip install -e . --no-build-isolation
./build.sh pendulum6
python -m pufferlib.pufferl train pendulum6 --train.total-timesteps 1000000000
```

## PufferLib 4.0 checkpoint-format note

The `.bin` checkpoints are flat fp32 dumps of the master weights. Parameter
order is encoder → decoder (fused `[action means; value]` rows + logstd) →
MinGRU layers, all bias-free. One subtlety: the allocator aligns each tensor
to 16 bytes in the bf16 parameter buffer while the fp32 master maps onto it
linearly by element index — so the 1-float logstd leaves a 7-element padding
gap before the GRU weights, and the final 7 GRU weights are permanently zero.
`eval_policy.py:load_flat` handles this.

## License

Environment and tooling code in this repo: MIT. PufferLib itself is licensed
separately (see upstream).
