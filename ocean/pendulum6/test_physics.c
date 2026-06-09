// Standalone physics sanity test. Build (no raylib needed):
//   gcc -O2 -DPENDULUM6_NO_RENDER -o test_physics test_physics.c -lm
#define PENDULUM6_NO_RENDER
#include "pendulum6.h"
#include <stdio.h>

static void setup(Pendulum6* env, unsigned int seed) {
    env->cart_mass = 1.0f;
    env->link_mass = 0.1f;
    env->link_length = 0.3f;
    env->gravity = 9.8f;
    env->force_mag = 30.0f;
    env->dt = 0.02f;
    env->substeps = 4;
    env->damping = 0.1f;
    env->spring_k = 2.0f;
    env->theta_fail = 0.5f;
    env->x_threshold = 3.0f;
    env->theta_init = 0.0873f;
    env->rng = seed;
    env->observations = (float*)calloc(P6_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    init(env);
}

// Total mechanical energy: point mass at the end of each link.
static double energy(Pendulum6* env) {
    double ke = 0.0, pe = 0.0;
    double vx_chain = env->x_dot, vy_chain = 0.0, y = 0.0;
    ke += 0.5 * env->cart_mass * env->x_dot * env->x_dot;
    for (int j = 0; j < P6_LINKS; j++) {
        vx_chain += env->link_length * env->theta_dot[j] * cos(env->theta[j]);
        vy_chain += -env->link_length * env->theta_dot[j] * sin(env->theta[j]);
        y += env->link_length * cos(env->theta[j]);
        ke += 0.5 * env->link_mass * (vx_chain * vx_chain + vy_chain * vy_chain);
        pe += env->link_mass * env->gravity * y;
    }
    return ke + pe;
}

int main(void) {
    int failures = 0;

    // Test 1: energy conservation, gentle swing near hanging (theta ~ pi),
    // no damping, no force, velocity clamps never bind.
    {
        Pendulum6 env;
        setup(&env, 12345);
        env.damping = 0.0f;
        env.spring_k = 0.0f;
        c_reset(&env);
        env.x = 0.0f; env.x_dot = 0.0f;
        for (int j = 0; j < P6_LINKS; j++) {
            env.theta[j] = (float)M_PI + 0.1f * (j % 2 ? -1.0f : 1.0f);
            env.theta_dot[j] = 0.0f;
        }
        double e0 = energy(&env);
        double max_drift = 0.0;
        float h = env.dt / 32.0f;
        for (int t = 0; t < 50; t++) {  // 1 second of free swing
            for (int i = 0; i < 32; i++) p6_substep(&env, 0.0f, h);
            double drift = fabs(energy(&env) - e0) / fmax(fabs(e0), 1e-9);
            if (drift > max_drift) max_drift = drift;
        }
        printf("test 1  energy drift over 1s gentle swing: %.4f%%  %s\n",
            100.0 * max_drift, max_drift < 0.02 ? "PASS" : "FAIL");
        if (max_drift >= 0.02) failures++;
    }

    // Test 2: zero action -> chain falls; episode ends well before timeout.
    // c_step auto-resets on done, so detect episode end via tick wrapping to 0.
    {
        Pendulum6 env;
        setup(&env, 999);
        c_reset(&env);
        int steps = 0;
        while (steps < P6_MAX_STEPS) {
            env.actions[0] = 0.0f;
            c_step(&env);
            steps++;
            if (env.tick == 0) break;  // env auto-reset: episode over
        }
        printf("test 2  zero-action fall time: %d steps (terminal=%.0f)  %s\n",
            steps, env.terminals[0],
            (steps < 300 && steps > 3 && env.terminals[0] > 0.5f) ? "PASS" : "FAIL");
        if (steps >= 300 || steps <= 3 || env.terminals[0] < 0.5f) failures++;
    }

    // Test 3: random actions, 200 episodes, no NaNs, episodes stay short
    {
        Pendulum6 env;
        setup(&env, 777);
        c_reset(&env);
        unsigned int arng = 42;
        long episodes = 0;
        double len_sum = 0;
        int prev_tick = 0;
        bool nan_seen = false;
        while (episodes < 200) {
            env.actions[0] = 2.0f * ((float)rand_r(&arng) / (float)RAND_MAX) - 1.0f;
            c_step(&env);
            for (int k = 0; k < P6_OBS_SIZE; k++) {
                if (!isfinite(env.observations[k])) nan_seen = true;
            }
            if (env.tick == 0) {
                episodes++; len_sum += prev_tick + 1;
            }
            prev_tick = env.tick;
        }
        printf("test 3  random policy: mean ep len %.1f over 200 eps, nan=%d  %s\n",
            len_sum / 200.0, nan_seen, (!nan_seen && len_sum / 200.0 < 300) ? "PASS" : "FAIL");
        if (nan_seen || len_sum / 200.0 >= 300) failures++;
    }

    // Test 4: constant push accelerates the cart (direct substeps, no resets)
    {
        Pendulum6 env;
        setup(&env, 5);
        c_reset(&env);
        env.x = 0.0f; env.x_dot = 0.0f;
        for (int j = 0; j < P6_LINKS; j++) { env.theta[j] = 0.0f; env.theta_dot[j] = 0.0f; }
        for (int i = 0; i < 40; i++) p6_substep(&env, 20.0f, 0.005f);  // 0.2 s
        printf("test 4  cart responds to +force: x=%.3f x_dot=%.3f  %s\n",
            env.x, env.x_dot, (env.x_dot > 1.0f && env.x > 0.0f) ? "PASS" : "FAIL");
        if (env.x_dot <= 1.0f || env.x <= 0.0f) failures++;
    }

    printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures;
}
