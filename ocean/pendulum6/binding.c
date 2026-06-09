#include "pendulum6.h"

#define OBS_SIZE P6_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {1}  // 1 continuous action: horizontal cart force
#define OBS_TENSOR_T FloatTensor

#define Env Pendulum6
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->cart_mass = dict_get(kwargs, "cart_mass")->value;
    env->link_mass = dict_get(kwargs, "link_mass")->value;
    env->link_length = dict_get(kwargs, "link_length")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->force_mag = dict_get(kwargs, "force_mag")->value;
    env->dt = dict_get(kwargs, "dt")->value;
    env->substeps = (int)dict_get(kwargs, "substeps")->value;
    env->damping = dict_get(kwargs, "damping")->value;
    env->spring_k = dict_get(kwargs, "spring_k")->value;
    env->theta_fail = dict_get(kwargs, "theta_fail")->value;
    env->x_threshold = dict_get(kwargs, "x_threshold")->value;
    env->theta_init = dict_get(kwargs, "theta_init")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "angle_termination", log->angle_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "n", log->n);
}
