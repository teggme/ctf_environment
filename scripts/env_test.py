import os

import numpy as np
from config import Config
from env_utils import EnvWrapper

# from env_utils import get_obs, img2gif, render

# "attack" or "return"
subtask = "attack"
cfg = Config(subtask=subtask)
os.makedirs(f"{cfg.base_path}/agents", exist_ok=True)

env = EnvWrapper(cfg)
init_obs = env.reset()

## INIT DYNAMICS MODEL
# f = lambda x: flow(x, env, None)  # env2)
# C = lambda x: inside_C(x, env, None)  # env2)
# g = lambda x: jump(
#     x,
#     env,
#     None,  # env2,
#     None,
#     None,
#     cfg.noise_magnitude_position,
#     cfg.noise_magnitude_orientation,
#     cfg.HyRL_MP_sets,
#     cfg.normal_base_position,
#     cfg.hyrlmp_base_position,
#     cfg.collision_radius,
#     cfg.capture_radius,
# )
# D = lambda x: inside_D(x, env, None)  # env2)


## START
num_episodes = 20

obss = []
acts = []
rews = []
infos = []
terms = []
gt_acts = []


for episode in range(num_episodes + 1):
    elapsed_time = 0
    done = False

    ep_obss = []
    ep_acts = []
    ep_rews = []
    ep_infos = []
    ep_term = []
    ep_gt_acts = []

    obs = env.reset()

    while not done:

        act = env.sample_action()

        next_obs, reward, done, info = env.step(act, curr_obs=obs)

        ep_obss.append(obs)
        ep_acts.append(act)
        ep_rews.append(reward)
        ep_infos.append(
            {
                "timer": elapsed_time,
            }
        )
        ep_term.append(done)
        ep_gt_acts.append(cfg.goal)

        # render(
        #     obs,
        #     env,
        #     None,  # env2,
        #     t,
        #     cfg.base_path,
        #     capture_radius=cfg.capture_radius,
        #     normal_base_position=cfg.normal_base_position,
        #     hyrlmp_base_position=cfg.hyrlmp_base_position,
        #     noise_magnitude_position=cfg.noise_magnitude_position,
        #     noise_magnitude_orientation=cfg.noise_magnitude_orientation,
        #     NUMBER_OF_STATIC_OBSTACLES=cfg.number_of_static_obstacles,
        #     task_level=cfg.task_level,
        # )

        obs = next_obs

    obss.append(ep_obss)
    acts.append(ep_acts)
    rews.append(ep_rews)
    infos.append(ep_infos)
    terms.append(ep_term)
    gt_acts.append(ep_gt_acts)

    # img2gif(cfg.base_path, episode, cfg.task_level)

np.savez_compressed(
    f"{cfg.base_path}/agents/data_ctf_random_primitive_new{subtask}.npz",
    actions=acts,
    observations=obss,
    terminals=terms,
    gt_actions=gt_acts,
)  # save the file

print("DONE")
