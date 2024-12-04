# players.py
import random
from .config import BUILDING_TYPES, BUILDING_COSTS
from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")

class BasePlayer:
    def __init__(self, name):
        self.name = name
        self.self_score = 0
        self.integrated_score = 0
        self.final_score = 0
        self.resources = {
            "money": 20,
            "reputation": 20,
        }

    def select_action(self, observation):
        # To be implemented by subclasses
        pass

    def update_state(self, reward, info):
        self.self_score += reward

        if "resources" in info:
            for resource, change in info["resources"].items():
                if resource in self.resources:
                    self.resources[resource] += change
                else:
                    self.resources[resource] = change


class BalancedPlayer(BasePlayer):
    def select_action(self, observation):
        resources = observation["resources"]
        available_actions = []
        for action_id in range(len(BUILDING_TYPES) * observation["grid"].shape[0] ** 2):
            building_type_idx = action_id // (observation["grid"].shape[0] ** 2)
            building_type = BUILDING_TYPES[building_type_idx]
            if (
                resources["money"] >= BUILDING_COSTS[building_type]["money"]
                and resources["reputation"]
                >= BUILDING_COSTS[building_type]["reputation"]
            ):
                available_actions.append(action_id)
        if available_actions:
            return random.choice(available_actions)
        return -1
