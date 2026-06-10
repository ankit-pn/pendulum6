// Standalone physics sanity tests. Build (no raylib needed):
//   gcc -O2 -DPENDULUM6F_NO_RENDER -o test_physics test_physics.c -lm
#define PENDULUM6F_NO_RENDER
#include "pendulum6f.h"
#include <stdio.h>

static void setup(Pendulum6F* env, unsigned int seed) {
    env->cart_mass = 1.0f;
    env->link_mass = 0.2f;
    env->link_length = 1.0f;
    env->gravity = 9.8f;
    env->force_mag = 120.0f;
    env->dt = 0.01f;
    env->substeps = 5;
    env->damping = 0.01f;
    env->spring_k = 0.0f;
    env->x_threshold = 4.0f;
    env->hold_target = 300;
    env->eval_mode = 0;
    env->rng = seed;
    env->observations = (float*)calloc(P6F_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    init(env);
}

// Energy and horizontal momentum for uniform rods (COM at l/2, Ic = m l^2/12).
static void invariants(Pendulum6F* env, double* E, double* px) {
    double m = env->link_mass, l = env->link_length, g = env->gravity;
    double Ic = m * l * l / 12.0;
    double ke = 0.5 * env->cart_mass * env->x_dot * env->x_dot;
    double pe = 0.0;
    double mom = env->cart_mass * env->x_dot;
    double base_vx = env->x_dot, base_vy = 0.0, base_y = 0.0;
    for (int j = 0; j < P6F_LINKS; j++) {
        double cj = cos(env->theta[j]), sj = sin(env->theta[j]);
        double w = env->theta_dot[j];
        double vx = base_vx + 0.5 * l * w * cj;
        double vy = base_vy - 0.5 * l * w * sj;
        double yc = base_y + 0.5 * l * cj;
        ke += 0.5 * m * (vx * vx + vy * vy) + 0.5 * Ic * w * w;
        pe += m * g * yc;
        mom += m * vx;
        base_vx += l * w * cj;
        base_vy += -l * w * sj;
        base_y += l * cj;
    }
    *E = ke + pe;
    *px = mom;
}

int main(void) {
    int failures = 0;

    // Test 1: energy + horizontal momentum conservation, free swing from a
    // bent configuration near hanging, no damping, no force.
    {
        Pendulum6F env;
        setup(&env, 12345);
        env.damping = 0.0f;
        c_reset(&env);
        env.x = 0.0f; env.x_dot = 0.0f;
        for (int j = 0; j < P6F_LINKS; j++) {
            env.theta[j] = (float)M_PI + 0.3f * (j % 2 ? -1.0f : 1.0f);
            env.theta_dot[j] = 0.0f;
        }
        double E0, p0; invariants(&env, &E0, &p0);
        double maxE = 0.0, maxP = 0.0;
        float h = env.dt / 32.0f;
        for (int t = 0; t < 200; t++) {  // 2 seconds of free swing
            for (int i = 0; i < 32; i++) p6f_substep(&env, 0.0f, h);
            double E, p; invariants(&env, &E, &p);
            double dE = fabs(E - E0) / fmax(fabs(E0), 1e-9);
            double dp = fabs(p - p0);
            if (dE > maxE) maxE = dE;
            if (dp > maxP) maxP = dp;
        }
        printf("test 1  free swing 2s: energy drift %.4f%%, momentum drift %.2e  %s\n",
            100.0 * maxE, maxP, (maxE < 0.02 && maxP < 1e-2) ? "PASS" : "FAIL");
        if (maxE >= 0.02 || maxP >= 1e-2) failures++;
    }

    // Test 2: zero action from near-upright -> chain falls (height drops)
    {
        Pendulum6F env;
        setup(&env, 999);
        c_reset(&env);
        env.x = 0.0f; env.x_dot = 0.0f;
        for (int j = 0; j < P6F_LINKS; j++) {
            env.theta[j] = 0.03f * (j % 2 ? -1.0f : 1.0f);
            env.theta_dot[j] = 0.0f;
        }
        int steps_above = 0;
        for (int t = 0; t < 300; t++) {
            env.actions[0] = 0.0f;
            c_step(&env);
            if (p6f_height01(&env) > 0.95f) steps_above++;
            else break;
        }
        printf("test 2  zero-action fall: upright for %d steps  %s\n",
            steps_above, (steps_above > 3 && steps_above < 200) ? "PASS" : "FAIL");
        if (steps_above <= 3 || steps_above >= 200) failures++;
    }

    // Test 3: random actions, 100 episodes, no NaNs in obs
    {
        Pendulum6F env;
        setup(&env, 777);
        c_reset(&env);
        unsigned int arng = 42;
        long episodes = 0;
        bool nan_seen = false;
        long steps = 0;
        while (episodes < 100 && steps < 2000000) {
            env.actions[0] = 2.0f * ((float)rand_r(&arng) / (float)RAND_MAX) - 1.0f;
            c_step(&env);
            steps++;
            for (int k = 0; k < P6F_OBS_SIZE; k++) {
                if (!isfinite(env.observations[k])) nan_seen = true;
            }
            if (env.tick == 0) episodes++;
        }
        printf("test 3  random policy: %ld episodes in %ld steps, nan=%d  %s\n",
            episodes, steps, nan_seen, (!nan_seen && episodes == 100) ? "PASS" : "FAIL");
        if (nan_seen || episodes != 100) failures++;
    }

    // Test 4: constant push accelerates the cart (direct substeps from upright)
    {
        Pendulum6F env;
        setup(&env, 5);
        c_reset(&env);
        env.x = 0.0f; env.x_dot = 0.0f;
        for (int j = 0; j < P6F_LINKS; j++) { env.theta[j] = 0.0f; env.theta_dot[j] = 0.0f; }
        for (int i = 0; i < 100; i++) p6f_substep(&env, 120.0f, 0.002f);  // 0.2 s
        printf("test 4  cart responds to +force: x=%.3f x_dot=%.3f  %s\n",
            env.x, env.x_dot, (env.x_dot > 1.0f && env.x > 0.0f) ? "PASS" : "FAIL");
        if (env.x_dot <= 1.0f || env.x <= 0.0f) failures++;
    }

    // Test 5: curriculum mechanics — eval_mode starts hanging; training mode
    // starts near upright at p=0.05
    {
        Pendulum6F env;
        setup(&env, 31);
        env.eval_mode = 1;
        init(&env);
        c_reset(&env);
        float h_eval = p6f_height01(&env);
        Pendulum6F env2;
        setup(&env2, 32);
        init(&env2);
        c_reset(&env2);
        float h_train = p6f_height01(&env2);
        printf("test 5  curriculum: eval start height %.3f (hanging), "
               "train start height %.3f (upright)  %s\n",
            h_eval, h_train, (h_eval < 0.1f && h_train > 0.9f) ? "PASS" : "FAIL");
        if (h_eval >= 0.1f || h_train <= 0.9f) failures++;
    }

    printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures;
}
