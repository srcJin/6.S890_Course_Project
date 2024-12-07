# src/envs/simcity/players.py

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
        """
        Updates the player's self_score and resources. This method can be
        further customized in subclasses to reflect player-specific behaviors.
        """
        self.self_score += reward

        if "resources" in info:
            for resource, change in info["resources"].items():
                if resource in self.resources:
                    self.resources[resource] += change
                else:
                    self.resources[resource] = change

        # Log the updated resources
        logger.debug(f"players: Player {self.name} updated resources: {self.resources}")


class InterestDrivenPlayer(BasePlayer):
    def update_state(self, reward, info):
        super().update_state(reward, info)

        # Update integrated score using a player-specific formula
        self.self_score = (
            0.5 * self.resources["money"] + 0.5 * self.resources["reputation"]
        )
        logger.debug(f"players: Interest-Driven Player {self.name} self_score: {self.self_score}")


class AltruisticPlayer(BasePlayer):
    def update_state(self, reward, info):
        super().update_state(reward, info)

        # Update integrated score using a player-specific formula
        self.self_score = (
            0.5 * self.resources["money"] + 0.5 * self.resources["reputation"]
        )
        logger.debug(f"players: Altruistic Player {self.name} self_score: {self.self_score}")


class BalancedPlayer(BasePlayer):
    def update_state(self, reward, info):
        super().update_state(reward, info)

        # Update integrated score using a player-specific formula
        self.self_score = (
            0.5 * self.resources["money"] + 0.5 * self.resources["reputation"]
        )
        logger.debug(f"players: Balanced Player {self.name} self_score: {self.self_score}")
