import pickle
from dataclasses import dataclass
from os.path import abspath, dirname

import numpy as np


@dataclass
class Config:
    """class of configuration"""

    subtask: str

    def __init__(self, subtask: str):
        self.subtask = subtask

        self.normal_base_position = (-5, 0)
        self.hyrlmp_base_position = (5, 0)
        self.goal = (
            np.array(self.hyrlmp_base_position)
            if self.subtask == "attack"
            else np.array(self.normal_base_position)
        )

        self.noise_magnitude_position = 0.2 + 0 * 0.05
        self.noise_magnitude_orientation = 0 * 0.1

        self.capture_radius = 0.5
        self.collision_radius = 0.5
        self.base_path = "/".join(abspath(dirname(__file__)).split("/")[:-1])

        self._set_obstacle_property()
        self._set_dynamics()

    def _set_obstacle_property(self):
        self.static_obstacle_locations_shared = [
            (0, 0)
        ]  # , (-2.75, 2.75), (2.75, 2.75), (2.75, -2.75), (-2.75, -2.75)]
        self.static_obstacle_locations_normal = [
            self.hyrlmp_base_position
        ] + self.static_obstacle_locations_shared
        self.STATIC_OBSTACLE_LOCATIONS_hyrlmp = [
            self.normal_base_position
        ] + self.static_obstacle_locations_shared

        self.static_obstacle_radii = [0.1] + [0.5] * len(
            self.static_obstacle_locations_shared
        )
        self.number_of_static_obstacles = 1

    def _set_dynamics(self):
        save_name = (
            f"{self.base_path}/hyrl_sets/ob1_Mij_x00_y00_static_example11_v1.pkl"
        )

        with open(save_name, "rb") as inp:
            self.HyRL_MP_sets = pickle.load(inp)

    def set_mode(self, is_sc):
        if is_sc is True:
            self.skill_chaining = True
        else:
            self.skill_chaining = False
