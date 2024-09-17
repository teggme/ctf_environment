import glob
import os
import random
from contextlib import contextmanager

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment

NUMBER_OF_STATIC_OBSTACLES = 1
NORMAL_BASE_POSITION = (-5, 0)
HYRLMP_BASE_POSITION = (5, 0)


class EnvWrapper:
    def __init__(self, cfg):
        self.cfg = cfg

        self.SUBTASK_ORDER = {
            "attack": 0,
            "return": 1,
        }
        self.SUBTASK_STEPS = {
            "attack": 100,
            "return": 100,
        }
        self.SUBTASK_PREV_SUBTASK = {
            "return": "attack",
        }
        self.SUBTASK_NEXT_SUBTASK = {
            "attack": "return",
        }
        self.START_SUBTASK = "attack"
        self.LAST_SUBTASK = "return"

        self._start_subtask = self.START_SUBTASK
        self.subtask = self.cfg.subtask

        env_kwargs = dict(
            initialize_state_randomly=True,
            TIME_STEPS=1000,
            train_hyrl=False,
            NUMBER_OF_STATIC_OBSTACLES=self.cfg.number_of_static_obstacles,
            clip_through_obstacles=False,
            SAMPLING_TIME_SECONDS=0.05,
            train=False,
            RANDOM_NUMBER_OF_STATIC_OBSTACLES=False,
            STATIC_OBSTACLE_RADII=self.cfg.static_obstacle_radii,
            VEHICLE_MODEL="Dubin",
            IGNORE_OBSTACLES=True,
            OBSTACLE_MODE="static",
            USE_IMAGE=True,
            FUTURE_STEPS_DYNAMIC_OBSTACLE=0,
            setpoint_radius=0.5,
        )

        _ = self.get_init_obs()
        init_normal_obs = self.jan_observation[:4]
        init_hyrlmp_obs = self.jan_observation[7:11]

        self.bev_env = BirdsEyeViewStaticObstacleLocationsEnvironment(
            INITIAL_STATE=(
                init_normal_obs if self.cfg.subtask == "attack" else init_hyrlmp_obs
            ),
            STATIC_OBSTACLE_LOCATIONS=self.cfg.static_obstacle_locations_normal,
            POSITIONAL_SET_POINT=(
                self.cfg.hyrlmp_base_position
                if self.cfg.subtask == "attack"
                else self.cfg.normal_base_position
            ),
            **env_kwargs,
        )

        self.action_space = self.bev_env.action_space
        self.MAX_ACTION_RANGE = self.action_space.shape[0]
        # TODO observation space

        self._elapsed_steps = 0

    def get_init_obs(self):

        obs = {}

        if self.cfg.subtask == "return":
            normal_base_position = self.cfg.hyrlmp_base_position
            hyrlmp_base_position = self.cfg.normal_base_position
        else:
            normal_base_position = self.cfg.normal_base_position
            hyrlmp_base_position = self.cfg.hyrlmp_base_position

        theta_normal = np.arctan2(
            hyrlmp_base_position[1] - normal_base_position[1],
            hyrlmp_base_position[0] - normal_base_position[0],
        )
        env_init_normal = np.array(
            [
                normal_base_position[0],
                normal_base_position[1],
                np.cos(theta_normal),
                np.sin(theta_normal),
            ]
        )
        theta_hyrlmp = np.arctan2(
            normal_base_position[1] - hyrlmp_base_position[1],
            normal_base_position[0] - hyrlmp_base_position[0],
        )
        env_init_hyrlmp = np.array(
            [
                hyrlmp_base_position[0],
                hyrlmp_base_position[1],
                np.cos(theta_hyrlmp),
                np.sin(theta_hyrlmp),
            ]
        )

        ### Observation for Jan's environment
        self.jan_observation = np.array(
            [
                env_init_normal[0],
                env_init_normal[1],
                env_init_normal[2],
                env_init_normal[3],
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                0,
                #
                env_init_hyrlmp[0],
                env_init_hyrlmp[1],
                env_init_hyrlmp[2],
                env_init_hyrlmp[3],
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.choice([0, 1]),
                random.choice([*range(self.cfg.number_of_static_obstacles)]),
                0,
                #
                0,
                1,
                0,
                0,
            ]
        )

        if self.cfg.subtask == "return":
            # set holding flag to True
            self.jan_observation[6] = 1
            self.jan_observation[15] = 1

        observation = np.array(
            [
                env_init_normal[0],
                env_init_normal[1],
                env_init_normal[2],
                env_init_normal[3],
            ]
        )

        obs["observation"] = observation
        obs["achieved_goal"] = observation[:2] - self.cfg.goal
        obs["desired_goal"] = self.cfg.goal

        return obs

    def step(self, act, curr_obs=None):

        ### FLOW function ###
        # obs = prev_obs.copy()
        obs = curr_obs["observation"]

        # normal obs
        obs[0] += act[0]
        obs[1] += act[1]
        orientation_vehicle_angle = np.arctan2(obs[1], obs[0])
        angle_deg = np.degrees(orientation_vehicle_angle)
        obs[2] = obs[0] * np.cos(angle_deg) - obs[1] * np.sin(angle_deg)
        obs[3] = obs[0] * np.sin(angle_deg) - obs[1] * np.cos(angle_deg)

        # obs[16] += 1  # timer
        self._elapsed_steps += 1

        curr_obs["observation"] = obs
        curr_obs["achieved_goal"] = obs[:2] - curr_obs["desired_goal"]

        info = {}
        info["gt_goal"] = self.cfg.goal
        reward = self.compute_reward(curr_obs["observation"][:2], self.cfg.goal)
        info["is_success"] = reward + 1
        info["step"] = 1 - reward
        done = self._elapsed_steps == self.SUBTASK_STEPS[self.subtask]
        reward = done * reward
        info["subtask"] = self.subtask
        info["subtask_done"] = False
        info["subtask_is_success"] = reward

        if done:
            info["subtask_done"] = True
            # Transit to next subtask (if current subtask is not terminal) and reset elapsed steps
            if self.subtask in self.SUBTASK_NEXT_SUBTASK.keys():
                done = False
                self._elapsed_steps = 0
                self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]
                info["is_success"] = False
                reward = 0
            else:
                info["is_success"] = reward

        return curr_obs, reward, done, info

    def reset(self, subtask=None):
        self.subtask = self._start_subtask if subtask is None else subtask

        init_obs = self.get_init_obs()
        self._elapsed_steps = 0

        return init_obs

    def compute_reward(self, loc, goal):
        if len(loc.shape) != 1:
            metric = np.all(loc == goal, axis=1)
            return np.where(metric, 1, -1)
        else:
            if (loc == goal).all():
                return 1
            return -1

    def goal_adapator(self, goal, subtask, device=None):
        """Make predicted goal compatible with wrapper"""
        if isinstance(goal, np.ndarray):
            return np.append(goal, self.SUBTASK_CONTACT_CONDITION[subtask])
        elif isinstance(goal, torch.Tensor):
            assert device is not None
            ct_cond = torch.tensor(
                self.SUBTASK_CONTACT_CONDITION[subtask], dtype=torch.float32
            )
            ct_cond = ct_cond.repeat(goal.shape[0], 1).to(device)
            adp_goal = torch.cat([goal, ct_cond], 1)
            return adp_goal

    def get_reward_functions(self):
        reward_funcs = {}
        for subtask in self.subtask_order.keys():
            with self.switch_subtask(subtask):
                reward_funcs[subtask] = self.compute_reward
        return reward_funcs

    def sample_action(self):
        return self.bev_env.action_space.sample()

    @contextmanager
    def switch_subtask(self, subtask=None):
        """Temporally switch subtask, default: next subtask"""
        if subtask is not None:
            curr_subtask = self.subtask
            self.subtask = subtask
            yield
            self.subtask = curr_subtask
        else:
            self.subtask = self.SUBTASK_PREV_SUBTASK[self.subtask]
            yield
            self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]

    @property
    def start_subtask(self):
        return self._start_subtask

    @property
    def max_episode_steps(self):
        if self.cfg.skill_chaining:
            return np.sum([x for x in self.SUBTASK_STEPS.values()])
        return self.SUBTASK_STEPS[self.subtask]

    @property
    def max_action_range(self):
        return self.MAX_ACTION_RANGE

    @property
    def subtask_order(self):
        return self.SUBTASK_ORDER

    @property
    def subtask_steps(self):
        return self.SUBTASK_STEPS

    @property
    def subtasks(self):
        subtasks = []
        for subtask, order in self.subtask_order.items():
            if order >= self.subtask_order[self.start_subtask]:
                subtasks.append(subtask)
        return subtasks

    @property
    def prev_subtasks(self):
        return self.SUBTASK_PREV_SUBTASK

    @property
    def next_subtasks(self):
        return self.SUBTASK_NEXT_SUBTASK

    @property
    def last_subtask(self):
        return self.LAST_SUBTASK

    @property
    def len_cond(self):
        return 0


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin[0], origin[1]
    px, py = point[0], point[1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def compute_input(norm_obs, hyrl_obs):
    theta_normal = np.arctan2(hyrl_obs[1] - norm_obs[1], hyrl_obs[0] - norm_obs[0])
    obs_normal = np.array(
        [norm_obs[0], norm_obs[1], np.cos(theta_normal), np.sin(theta_normal)]
    )
    theta_hyrlmp = np.arctan2(norm_obs[1] - hyrl_obs[1], norm_obs[0] - hyrl_obs[0])
    obs_hyrl = np.array(
        [hyrl_obs[0], hyrl_obs[1], np.cos(theta_hyrlmp), np.sin(theta_hyrlmp)]
    )

    return obs_normal, obs_hyrl


def img2gif(base_path, ep, task_level):
    file_path = f"{base_path}/gifstorage/capturetheflag"
    files = glob.glob(f"{file_path}/*.png")

    # Function to extract the number from the filename
    def extract_number(filename):
        # Split by '_n' and then take the part before '.png'
        return int(filename.split("_n")[-1].split(".png")[0])

    # Sort the list using the extract_number function as the key
    sorted_files = sorted(files, key=extract_number)

    frames = []
    for idx in range(len(files)):
        image = imageio.v2.imread(sorted_files[idx])
        frames.append(image)
    imageio.mimsave(
        f"{base_path}/gifs/capturetheflag/capturetheflag_task{task_level}_ep{ep}.gif",
        frames,
    )

    for f in files:
        os.remove(f)


def render(
    obs,
    env,
    hyrlmp_env,
    step,
    base_path,
    capture_radius=0.5,
    normal_base_position=(-5, 0),
    hyrlmp_base_position=(5, 0),
    noise_magnitude_position=0.2,
    noise_magnitude_orientation=0,
    NUMBER_OF_STATIC_OBSTACLES=1,
    task_level=1,
):
    """obs should be Jan's observation"""
    xs_normal = obs[0]
    ys_normal = obs[1]
    angle_normal = np.arctan2(obs[3], obs[2])

    has_enemy_flag_normal = obs[6]

    xs_hyrlmp = obs[7]
    ys_hyrlmp = obs[8]
    angle_hyrlmp = np.arctan2(obs[10], obs[9])
    has_enemy_flag_hyrlmp = obs[15]

    normal_points_scored = obs[18]
    hyrlmp_points_scored = obs[19]

    markersize = 100

    if task_level == 1:
        has_enemy_flag_normal = False
        has_enemy_flag_hyrlmp = False
    else:
        has_enemy_flag_normal = True
        has_enemy_flag_hyrlmp = True

    thetas_plot = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    plt.plot(
        normal_base_position[0] + np.cos(thetas_plot) * capture_radius,
        normal_base_position[1] + np.sin(thetas_plot) * capture_radius,
        "--",
        color="red",
        linewidth=0.5,
    )
    plt.plot(
        hyrlmp_base_position[0] + np.cos(thetas_plot) * capture_radius,
        hyrlmp_base_position[1] + np.sin(thetas_plot) * capture_radius,
        "--",
        color="blue",
        linewidth=0.5,
    )

    normal_view = matplotlib.patches.Rectangle(
        (
            xs_normal - env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE / 2,
            ys_normal - env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE / 2,
        ),
        env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE,
        env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE,
        edgecolor="red",
        facecolor="none",
        rotation_point="center",
        angle=angle_normal * 180 / np.pi,
        linestyle="--",
    )
    plt.gca().add_patch(normal_view)

    rotated_marker_normal = matplotlib.markers.MarkerStyle(marker=9)
    rotated_marker_normal._transform = rotated_marker_normal.get_transform().rotate_deg(
        angle_normal * 180 / np.pi
    )

    plt.scatter(
        xs_normal,
        ys_normal,
        marker=rotated_marker_normal,
        s=(markersize),
        facecolors="red",
    )
    plt.scatter(
        xs_normal,
        ys_normal,
        marker="o",
        s=(markersize) / 1,
        facecolors="red",
        edgecolors="red",
    )

    if hyrlmp_env is not None:
        hyrlmp_view = matplotlib.patches.Rectangle(
            (
                xs_hyrlmp - hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE / 2,
                ys_hyrlmp - hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE / 2,
            ),
            hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE,
            hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE,
            edgecolor="blue",
            facecolor="none",
            rotation_point="center",
            angle=angle_hyrlmp * 180 / np.pi,
            linestyle="--",
        )
        plt.gca().add_patch(hyrlmp_view)

        rotated_marker_hyrlmp = matplotlib.markers.MarkerStyle(marker=9)
        rotated_marker_hyrlmp._transform = (
            rotated_marker_hyrlmp.get_transform().rotate_deg(angle_hyrlmp * 180 / np.pi)
        )

        plt.scatter(
            xs_hyrlmp,
            ys_hyrlmp,
            marker=rotated_marker_hyrlmp,
            s=(markersize),
            facecolors="blue",
        )
        plt.scatter(
            xs_hyrlmp,
            ys_hyrlmp,
            marker="o",
            s=(markersize) / 1,
            facecolors="blue",
            edgecolors="blue",
        )

    for idob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS[1:]):
        idob += 1
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle(
            (x_obst, y_obst), radius=env.STATIC_OBSTACLE_RADII[idob], color="gray"
        )
        plt.gca().add_patch(obstaclePatch)

    if has_enemy_flag_normal == False and has_enemy_flag_hyrlmp == False:
        plt.plot(
            [xs_normal, hyrlmp_base_position[0]],
            [ys_normal, hyrlmp_base_position[1]],
            "--",
            color="red",
            linewidth=0.5,
        )
        plt.plot(
            [xs_hyrlmp, normal_base_position[0]],
            [ys_hyrlmp, normal_base_position[1]],
            "--",
            color="blue",
            linewidth=0.5,
        )
    elif has_enemy_flag_normal == True and has_enemy_flag_hyrlmp == False:
        plt.plot(
            [xs_normal, normal_base_position[0]],
            [ys_normal, normal_base_position[1]],
            "--",
            color="red",
            linewidth=0.5,
        )
        plt.plot(
            [xs_hyrlmp, xs_normal],
            [ys_hyrlmp, ys_normal],
            "--",
            color="blue",
            linewidth=0.5,
        )
    elif has_enemy_flag_normal == False and has_enemy_flag_hyrlmp == True:
        plt.plot(
            [xs_hyrlmp, hyrlmp_base_position[0]],
            [ys_hyrlmp, hyrlmp_base_position[1]],
            "--",
            color="blue",
            linewidth=0.5,
        )
        plt.plot(
            [xs_hyrlmp, xs_normal],
            [ys_hyrlmp, ys_normal],
            "--",
            color="red",
            linewidth=0.5,
        )
    else:
        plt.plot(
            [xs_hyrlmp, xs_normal],
            [ys_hyrlmp, ys_normal],
            "--",
            color="black",
            linewidth=0.5,
        )

    # flag
    if has_enemy_flag_normal == True:
        plt.plot(
            xs_normal + 0.1,
            ys_normal + 0.1,
            marker="$\Gamma$",
            markersize=10,
            color="blue",
            zorder=10,
        )
    else:
        plt.plot(
            hyrlmp_base_position[0],
            hyrlmp_base_position[1],
            marker="$\Gamma$",
            markersize=10,
            color="blue",
            zorder=10,
        )

    if has_enemy_flag_hyrlmp == True:
        plt.plot(
            xs_hyrlmp + 0.1,
            ys_hyrlmp + 0.1,
            marker="$\Gamma$",
            markersize=10,
            color="red",
            zorder=10,
        )
    else:
        plt.plot(
            normal_base_position[0],
            normal_base_position[1],
            marker="$\Gamma$",
            markersize=10,
            color="red",
            zorder=10,
        )

    # score board
    plt.text(
        -3,
        4,
        f"Score {str(round(normal_points_scored))}",
        fontsize=15,
        fontweight="bold",
        color="red",
        horizontalalignment="center",
    )
    plt.text(
        3,
        4,
        f"Score {str(round(hyrlmp_points_scored))}",
        fontsize=15,
        fontweight="bold",
        color="blue",
        horizontalalignment="center",
    )

    plt.grid(visible=True)
    plt.xlabel("$p_x$", fontsize=22)
    plt.ylabel("$p_y$", fontsize=22)
    plt.xlim(-6, 6)
    plt.ylim([-6, 6])
    plt.tight_layout()

    plt.savefig(
        f"{base_path}/gifstorage/capturetheflag/capturetheflag_obstacles"
        + str(NUMBER_OF_STATIC_OBSTACLES)
        + "_noisemagposition"
        + str(noise_magnitude_position).replace(".", "")
        + "_noisemagorienation"
        + str(noise_magnitude_orientation).replace(".", "")
        + "_frame"
        + f"_n{step}"
        + ".png",
        transparent=False,
        facecolor="white",
    )
    # plt.pause(0.05)
    plt.clf()
