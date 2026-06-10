#include "pendulum6f.h"

#define OBS_SIZE P6F_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {1}  // 1 continuous action: horizontal cart force
#define OBS_TENSOR_T FloatTensor

#define Env Pendulum6F
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
    env->x_threshold = dict_get(kwargs, "x_threshold")->value;
    env->hold_target = (int)dict_get(kwargs, "hold_target")->value;
    env->eval_mode = (int)dict_get(kwargs, "eval_mode")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hold_time", log->hold_time);
    dict_set(out, "curriculum_p", log->curriculum_p);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "timeout_termination", log->timeout_termination);
    dict_set(out, "n", log->n);
}
