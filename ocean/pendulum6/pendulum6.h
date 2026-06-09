// 6-link inverted pendulum on a cart: balance task with continuous force.
// Links start near upright; episode ends when any link tips past theta_fail
// or the cart leaves the rail. Dynamics are the n-link generalization of
// ocean/double_pendulum (point mass at the end of each massless rod).
//
// Joints are elastic: a weak torsional spring (spring_k) at each joint pulls
// adjacent links into alignment, plus relative viscous damping (damping).
// This is load-bearing: LQR analysis shows the free-jointed 6-link chain is
// unstabilizable by cart force under any practical force/precision budget
// (required gains ~1e5), while with k=2 N*m/rad the chain is still actively
// unstable (lambda=2.9/s, falls without control) but controllable.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifndef PENDULUM6_NO_RENDER
#include "raylib.h"
#endif

#define P6_LINKS 6
#define P6_NDOF (P6_LINKS + 1)
#define P6_OBS_SIZE (2 + 3 * P6_LINKS)
#define P6_MAX_STEPS 1000
#define P6_WIDTH 800
#define P6_HEIGHT 560
#define P6_SCALE 140.0f

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float x_threshold_termination;
    float angle_termination;
    float max_steps_termination;
    float n;
} Log;

typedef struct Pendulum6 {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    float x;
    float x_dot;
    float theta[P6_LINKS];
    float theta_dot[P6_LINKS];
    int tick;
    float episode_return;

    float cart_mass;
    float link_mass;
    float link_length;
    float gravity;
    float force_mag;
    float dt;
    int substeps;
    float damping;
    float spring_k;
    float theta_fail;
    float x_threshold;
    float theta_init;
} Pendulum6;

static inline float p6_randf(Pendulum6* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

void compute_observations(Pendulum6* env) {
    env->observations[0] = env->x / env->x_threshold;
    env->observations[1] = env->x_dot / 5.0f;
    for (int j = 0; j < P6_LINKS; j++) {
        env->observations[2 + 3*j] = sinf(env->theta[j]);
        env->observations[3 + 3*j] = cosf(env->theta[j]);
        env->observations[4 + 3*j] = env->theta_dot[j] / 10.0f;
    }
}

void add_log(Pendulum6* env, bool x_done, bool angle_done, bool timeout) {
    float normalized = env->episode_return / (float)P6_MAX_STEPS;
    env->log.perf += fminf(fmaxf(normalized, 0.0f), 1.0f);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.angle_termination += angle_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(Pendulum6* env) {
    env->num_agents = 1;
}

void c_reset(Pendulum6* env) {
    env->x = p6_randf(env, -0.05f, 0.05f);
    env->x_dot = p6_randf(env, -0.05f, 0.05f);
    for (int j = 0; j < P6_LINKS; j++) {
        env->theta[j] = p6_randf(env, -env->theta_init, env->theta_init);
        env->theta_dot[j] = p6_randf(env, -0.05f, 0.05f);
    }
    env->tick = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

// Gaussian elimination with partial pivoting. A and b are destroyed.
static void solve_ndof(float A[P6_NDOF][P6_NDOF], float b[P6_NDOF], float out[P6_NDOF]) {
    for (int i = 0; i < P6_NDOF; i++) {
        int pivot = i;
        float best = fabsf(A[i][i]);
        for (int r = i + 1; r < P6_NDOF; r++) {
            float v = fabsf(A[r][i]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (pivot != i) {
            for (int c = i; c < P6_NDOF; c++) {
                float tmp = A[i][c];
                A[i][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
            float tmp = b[i];
            b[i] = b[pivot];
            b[pivot] = tmp;
        }

        float inv = 1.0f / A[i][i];
        for (int c = i; c < P6_NDOF; c++) A[i][c] *= inv;
        b[i] *= inv;
        for (int r = 0; r < P6_NDOF; r++) {
            if (r == i) continue;
            float f = A[r][i];
            if (f == 0.0f) continue;
            for (int c = i; c < P6_NDOF; c++) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    for (int i = 0; i < P6_NDOF; i++) out[i] = b[i];
}

// One substep of semi-implicit Euler on the full nonlinear dynamics.
// Generalized coords q = [x, theta_1..theta_N], theta measured from upright.
// mu[j] = sum of link masses j..N (1-indexed over links).
static void p6_substep(Pendulum6* env, float force, float h) {
    float l = env->link_length;
    float g = env->gravity;
    float mu[P6_LINKS + 1];
    mu[P6_LINKS] = env->link_mass;
    for (int j = P6_LINKS - 1; j >= 1; j--) {
        mu[j] = mu[j + 1] + env->link_mass;
    }

    float s[P6_LINKS], c[P6_LINKS];
    for (int j = 0; j < P6_LINKS; j++) {
        s[j] = sinf(env->theta[j]);
        c[j] = cosf(env->theta[j]);
    }

    float A[P6_NDOF][P6_NDOF];
    float b[P6_NDOF];

    A[0][0] = env->cart_mass + mu[1];
    b[0] = force;
    for (int j = 1; j <= P6_LINKS; j++) {
        int ji = j - 1;
        A[0][j] = mu[j] * l * c[ji];
        A[j][0] = A[0][j];
        b[0] += mu[j] * l * s[ji] * env->theta_dot[ji] * env->theta_dot[ji];
        b[j] = mu[j] * g * l * s[ji];
        // Elastic joint with parent (cart frame for link 1: rest angle is
        // vertical) and child, plus relative viscous damping at both joints.
        float parent_th = (ji == 0) ? 0.0f : env->theta[ji - 1];
        float parent_w = (ji == 0) ? 0.0f : env->theta_dot[ji - 1];
        b[j] -= env->spring_k * (env->theta[ji] - parent_th);
        b[j] -= env->damping * (env->theta_dot[ji] - parent_w);
        if (ji < P6_LINKS - 1) {
            b[j] -= env->spring_k * (env->theta[ji] - env->theta[ji + 1]);
            b[j] -= env->damping * (env->theta_dot[ji] - env->theta_dot[ji + 1]);
        }
        for (int k = 1; k <= P6_LINKS; k++) {
            int ki = k - 1;
            int m = (j > k) ? j : k;
            float cjk = cosf(env->theta[ji] - env->theta[ki]);
            float sjk = sinf(env->theta[ji] - env->theta[ki]);
            A[j][k] = mu[m] * l * l * cjk;
            b[j] -= mu[m] * l * l * sjk * env->theta_dot[ki] * env->theta_dot[ki];
        }
    }

    float qdd[P6_NDOF];
    solve_ndof(A, b, qdd);

    env->x_dot += h * qdd[0];
    env->x_dot = fminf(fmaxf(env->x_dot, -20.0f), 20.0f);
    env->x += h * env->x_dot;
    for (int j = 0; j < P6_LINKS; j++) {
        env->theta_dot[j] += h * qdd[j + 1];
        env->theta_dot[j] = fminf(fmaxf(env->theta_dot[j], -30.0f), 30.0f);
        env->theta[j] += h * env->theta_dot[j];
    }
}

void c_step(Pendulum6* env) {
    float a = env->actions[0];
    if (!isfinite(a)) a = 0.0f;
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->actions[0] = a;
    float force = a * env->force_mag;

    float h = env->dt / (float)env->substeps;
    for (int i = 0; i < env->substeps; i++) {
        p6_substep(env, force, h);
    }
    env->tick += 1;

    bool invalid = !isfinite(env->x) || !isfinite(env->x_dot);
    bool angle_done = false;
    float angle_cost = 0.0f;
    for (int j = 0; j < P6_LINKS; j++) {
        if (!isfinite(env->theta[j]) || !isfinite(env->theta_dot[j])) invalid = true;
        if (fabsf(env->theta[j]) > env->theta_fail) angle_done = true;
        angle_cost += fabsf(env->theta[j]);
    }
    angle_cost /= (float)P6_LINKS * env->theta_fail;
    bool x_done = env->x < -env->x_threshold || env->x > env->x_threshold;
    bool timeout = env->tick >= P6_MAX_STEPS;
    bool terminated = invalid || x_done || angle_done;
    bool done = terminated || timeout;

    float reward = terminated ? 0.0f :
        1.0f - 0.2f * angle_cost - 0.05f * fabsf(env->x) / env->x_threshold;
    env->rewards[0] = reward;
    env->episode_return += reward;
    env->terminals[0] = terminated ? 1.0f : 0.0f;

    if (done) {
        add_log(env, invalid || x_done, angle_done, timeout);
        c_reset(env);
        return;
    }
    compute_observations(env);
}

#ifndef PENDULUM6_NO_RENDER

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(Pendulum6* env) {
    if (!IsWindowReady()) {
        InitWindow(P6_WIDTH, P6_HEIGHT, "PufferLib 6-Link Pendulum");
        SetTargetFPS(50);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();
    if (!isfinite(env->x)) return;

    float rail_y = P6_HEIGHT * 0.85f;
    float cart_x = P6_WIDTH / 2.0f + env->x * P6_SCALE;
    cart_x = fminf(fmaxf(cart_x, 32.0f), P6_WIDTH - 32.0f);
    float cart_y = rail_y - 14.0f;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, (int)rail_y, P6_WIDTH, (int)rail_y, PUFF_CYAN);
    DrawRectangle((int)(cart_x - 26), (int)(cart_y - 11), 52, 22, PUFF_CYAN);

    Vector2 p = {cart_x, cart_y};
    float l = env->link_length * P6_SCALE;
    for (int j = 0; j < P6_LINKS; j++) {
        Vector2 q = {p.x + sinf(env->theta[j]) * l, p.y - cosf(env->theta[j]) * l};
        DrawLineEx(p, q, 5.0f, PUFF_RED);
        DrawCircleV(p, 5.0f, PUFF_WHITE);
        p = q;
    }
    DrawCircleV(p, 7.0f, PUFF_WHITE);
    DrawText(TextFormat("steps %d  return %.1f", env->tick, env->episode_return),
        20, 20, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(Pendulum6* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif
