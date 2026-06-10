// 6-link FREE-JOINT inverted pendulum on a cart: swing-up + balance + recover.
//
// Links are uniform rods (distributed mass: COM at l/2, inertia m*l^2/12 about
// COM), not point masses — this matches MuJoCo-style articulated dynamics and
// has meaningfully slower unstable modes than the point-mass chain.
//
// Built-in self-paced reverse curriculum: each env tracks its own progress
// p in [0,1]. Episodes start near upright at p=0 and fully hanging at p=1;
// p advances on success (held balanced >= hold_target steps) and backs off on
// failure, keeping the task at the edge of competence. Episode lengths are
// randomized per episode to prevent termination-laziness exploits.
//
// Reward: normalized tip height each step + a hold-streak bonus near upright.
// No angle termination (the chain must swing through everything); episodes
// end on cart leaving the rail, numerical blowup, or (randomized) timeout.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifndef PENDULUM6F_NO_RENDER
#include "raylib.h"
#endif

#define P6F_LINKS 6
#define P6F_NDOF (P6F_LINKS + 1)
#define P6F_OBS_SIZE (2 + 3 * P6F_LINKS)
#define P6F_MIN_STEPS 1600
#define P6F_MAX_STEPS 4000
#define P6F_WIDTH 860
#define P6F_HEIGHT 640
#define P6F_SCALE 45.0f

typedef struct Log {
    float perf;                     // fraction of successful episodes
    float score;
    float episode_return;
    float episode_length;
    float hold_time;                // best balanced streak (steps)
    float curriculum_p;
    float x_threshold_termination;
    float timeout_termination;
    float n;
} Log;

typedef struct Pendulum6F {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    float x;
    float x_dot;
    float theta[P6F_LINKS];
    float theta_dot[P6F_LINKS];
    int tick;
    int max_steps_ep;               // randomized per episode
    float episode_return;
    int hold_steps;
    int max_hold_steps;
    float curriculum;               // p in [0,1], persists across episodes

    float cart_mass;
    float link_mass;
    float link_length;
    float gravity;
    float force_mag;
    float dt;
    int substeps;
    float damping;                  // relative viscous damping at joints
    float spring_k;                 // optional joint stiffness (0 = free)
    float x_threshold;
    int hold_target;                // balanced steps that count as success
    int eval_mode;                  // 1: fixed hanging starts, no curriculum
} Pendulum6F;

static inline float p6f_randf(Pendulum6F* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

static inline float p6f_height01(Pendulum6F* env) {
    // Normalized tip height: 1 fully upright, 0 fully hanging.
    float y = 0.0f;
    for (int j = 0; j < P6F_LINKS; j++) y += cosf(env->theta[j]);
    return 0.5f * (y / (float)P6F_LINKS + 1.0f);
}

void compute_observations(Pendulum6F* env) {
    env->observations[0] = env->x / env->x_threshold;
    env->observations[1] = env->x_dot / 10.0f;
    for (int j = 0; j < P6F_LINKS; j++) {
        env->observations[2 + 3*j] = sinf(env->theta[j]);
        env->observations[3 + 3*j] = cosf(env->theta[j]);
        env->observations[4 + 3*j] = env->theta_dot[j] / 20.0f;
    }
}

void add_log(Pendulum6F* env, bool success, bool x_done, bool timeout) {
    env->log.perf += success ? 1.0f : 0.0f;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.hold_time += (float)env->max_hold_steps;
    env->log.curriculum_p += env->curriculum;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.timeout_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(Pendulum6F* env) {
    env->num_agents = 1;
    env->curriculum = env->eval_mode ? 1.0f : 0.05f;
}

void c_reset(Pendulum6F* env) {
    // Reverse curriculum: tilt the (straightened) chain by u*pi from upright,
    // u sampled up to current progress with a bias toward the frontier.
    float u;
    if (env->eval_mode) {
        u = 1.0f;
    } else {
        float t = p6f_randf(env, 0.0f, 1.0f);
        u = env->curriculum * sqrtf(t);
    }
    float side = (rand_r(&env->rng) & 1) ? 1.0f : -1.0f;
    float tilt = side * u * (float)M_PI;
    env->x = p6f_randf(env, -0.05f, 0.05f);
    env->x_dot = p6f_randf(env, -0.05f, 0.05f);
    for (int j = 0; j < P6F_LINKS; j++) {
        env->theta[j] = tilt + p6f_randf(env, -1.0f, 1.0f) * (0.02f + 0.15f * u);
        env->theta_dot[j] = p6f_randf(env, -0.05f, 0.05f);
    }
    env->tick = 0;
    env->max_steps_ep = P6F_MIN_STEPS
        + (int)(rand_r(&env->rng) % (P6F_MAX_STEPS - P6F_MIN_STEPS));
    env->episode_return = 0.0f;
    env->hold_steps = 0;
    env->max_hold_steps = 0;
    compute_observations(env);
}

// Gaussian elimination with partial pivoting. A and b are destroyed.
static void solve_ndof(float A[P6F_NDOF][P6F_NDOF], float b[P6F_NDOF], float out[P6F_NDOF]) {
    for (int i = 0; i < P6F_NDOF; i++) {
        int pivot = i;
        float best = fabsf(A[i][i]);
        for (int r = i + 1; r < P6F_NDOF; r++) {
            float v = fabsf(A[r][i]);
            if (v > best) { best = v; pivot = r; }
        }
        if (pivot != i) {
            for (int c = i; c < P6F_NDOF; c++) {
                float tmp = A[i][c]; A[i][c] = A[pivot][c]; A[pivot][c] = tmp;
            }
            float tmp = b[i]; b[i] = b[pivot]; b[pivot] = tmp;
        }
        float inv = 1.0f / A[i][i];
        for (int c = i; c < P6F_NDOF; c++) A[i][c] *= inv;
        b[i] *= inv;
        for (int r = 0; r < P6F_NDOF; r++) {
            if (r == i) continue;
            float f = A[r][i];
            if (f == 0.0f) continue;
            for (int c = i; c < P6F_NDOF; c++) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    for (int i = 0; i < P6F_NDOF; i++) out[i] = b[i];
}

// One substep of semi-implicit Euler on the full nonlinear rod-chain dynamics.
// Coefficients (uniform rods, 1-indexed links; m = link mass, l = length):
//   b_i = m*l*(N - i + 1/2)            cart coupling / gravity lever
//   G_ik = m*l^2*(N - max(i,k) + 1/2)  off-diagonal inertia coupling
//   e_i = m*l^2*(N - i + 1/3)          diagonal inertia (incl. rod term)
// Verified to reduce to the classic single-rod cartpole at N=1.
static void p6f_substep(Pendulum6F* env, float force, float h) {
    float l = env->link_length;
    float m = env->link_mass;
    float g = env->gravity;
    int N = P6F_LINKS;

    float s[P6F_LINKS], c[P6F_LINKS], bb[P6F_LINKS + 1];
    for (int j = 0; j < N; j++) {
        s[j] = sinf(env->theta[j]);
        c[j] = cosf(env->theta[j]);
    }
    for (int i = 1; i <= N; i++) bb[i] = m * l * ((float)(N - i) + 0.5f);

    float A[P6F_NDOF][P6F_NDOF];
    float b[P6F_NDOF];

    A[0][0] = env->cart_mass + (float)N * m;
    b[0] = force;
    for (int i = 1; i <= N; i++) {
        int ii = i - 1;
        A[0][i] = bb[i] * c[ii];
        A[i][0] = A[0][i];
        b[0] += bb[i] * s[ii] * env->theta_dot[ii] * env->theta_dot[ii];
        b[i] = g * bb[i] * s[ii];
        // elastic joint with parent (cart frame for link 1) and child,
        // plus relative viscous damping at both joints
        float parent_th = (ii == 0) ? 0.0f : env->theta[ii - 1];
        float parent_w = (ii == 0) ? 0.0f : env->theta_dot[ii - 1];
        b[i] -= env->spring_k * (env->theta[ii] - parent_th);
        b[i] -= env->damping * (env->theta_dot[ii] - parent_w);
        if (ii < N - 1) {
            b[i] -= env->spring_k * (env->theta[ii] - env->theta[ii + 1]);
            b[i] -= env->damping * (env->theta_dot[ii] - env->theta_dot[ii + 1]);
        }
        for (int k = 1; k <= N; k++) {
            int ki = k - 1;
            if (k == i) {
                A[i][i] = m * l * l * ((float)(N - i) + 1.0f / 3.0f);
                continue;
            }
            int mx = (i > k) ? i : k;
            float Gik = m * l * l * ((float)(N - mx) + 0.5f);
            A[i][k] = Gik * cosf(env->theta[ii] - env->theta[ki]);
            b[i] -= Gik * sinf(env->theta[ii] - env->theta[ki])
                    * env->theta_dot[ki] * env->theta_dot[ki];
        }
    }

    float qdd[P6F_NDOF];
    solve_ndof(A, b, qdd);

    env->x_dot += h * qdd[0];
    env->x_dot = fminf(fmaxf(env->x_dot, -25.0f), 25.0f);
    env->x += h * env->x_dot;
    for (int j = 0; j < P6F_LINKS; j++) {
        env->theta_dot[j] += h * qdd[j + 1];
        env->theta_dot[j] = fminf(fmaxf(env->theta_dot[j], -50.0f), 50.0f);
        env->theta[j] += h * env->theta_dot[j];
    }
}

static bool p6f_balanced(Pendulum6F* env) {
    // Height-only: dynamic balancing legitimately uses cart/joint motion.
    // (Spinning exploits self-limit: any fast link motion dips tip height.)
    return p6f_height01(env) >= 0.95f;
}

void c_step(Pendulum6F* env) {
    float a = env->actions[0];
    if (!isfinite(a)) a = 0.0f;
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->actions[0] = a;
    float force = a * env->force_mag;

    float h = env->dt / (float)env->substeps;
    for (int i = 0; i < env->substeps; i++) {
        p6f_substep(env, force, h);
    }
    env->tick += 1;

    bool invalid = !isfinite(env->x) || !isfinite(env->x_dot);
    for (int j = 0; j < P6F_LINKS; j++) {
        if (!isfinite(env->theta[j]) || !isfinite(env->theta_dot[j])) invalid = true;
    }
    bool x_done = env->x < -env->x_threshold || env->x > env->x_threshold;
    bool timeout = env->tick >= env->max_steps_ep;
    bool terminated = invalid || x_done;
    bool done = terminated || timeout;

    if (p6f_balanced(env)) env->hold_steps += 1;
    else env->hold_steps = 0;
    if (env->hold_steps > env->max_hold_steps) env->max_hold_steps = env->hold_steps;

    float height = p6f_height01(env);
    float hold_bonus = fminf(2.0f * (float)env->hold_steps / (float)env->hold_target, 1.0f);
    float reward = terminated ? 0.0f : 0.5f * height + hold_bonus;
    env->rewards[0] = reward;
    env->episode_return += reward;
    env->terminals[0] = terminated ? 1.0f : 0.0f;

    if (done) {
        bool success = env->max_hold_steps >= env->hold_target;
        if (!env->eval_mode) {
            if (success) env->curriculum += 0.02f;
            else env->curriculum -= 0.002f;
            env->curriculum = fminf(fmaxf(env->curriculum, 0.05f), 1.0f);
        }
        add_log(env, success, invalid || x_done, timeout);
        c_reset(env);
        return;
    }
    compute_observations(env);
}

#ifndef PENDULUM6F_NO_RENDER

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(Pendulum6F* env) {
    if (!IsWindowReady()) {
        InitWindow(P6F_WIDTH, P6F_HEIGHT, "PufferLib 6-Link Free Pendulum");
        SetTargetFPS(100);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();
    if (!isfinite(env->x)) return;

    float rail_y = P6F_HEIGHT * 0.55f;
    float cart_x = P6F_WIDTH / 2.0f + env->x * P6F_SCALE;
    cart_x = fminf(fmaxf(cart_x, 32.0f), P6F_WIDTH - 32.0f);
    float cart_y = rail_y - 14.0f;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, (int)rail_y, P6F_WIDTH, (int)rail_y, PUFF_CYAN);
    DrawRectangle((int)(cart_x - 26), (int)(cart_y - 11), 52, 22, PUFF_CYAN);

    Vector2 p = {cart_x, cart_y};
    float l = env->link_length * P6F_SCALE;
    for (int j = 0; j < P6F_LINKS; j++) {
        Vector2 q = {p.x + sinf(env->theta[j]) * l, p.y - cosf(env->theta[j]) * l};
        DrawLineEx(p, q, 5.0f, PUFF_RED);
        DrawCircleV(p, 5.0f, PUFF_WHITE);
        p = q;
    }
    DrawCircleV(p, 7.0f, PUFF_WHITE);
    DrawText(TextFormat("steps %d  return %.1f  hold %d/%d  p %.2f",
        env->tick, env->episode_return, env->hold_steps, env->max_hold_steps,
        env->curriculum), 20, 20, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(Pendulum6F* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif
