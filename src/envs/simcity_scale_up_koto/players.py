# src/envs/simcity/players.py

from utils.logging import get_logger

logger = get_logger(log_file_path="simulation.log")


class BasePlayer:
    def __init__(self, name, alpha=0.33, beta=0.33, gamma=0.34):
        self.name = name
        self.alpha = alpha  # Weight for personal resources (money + reputation)
        self.beta = beta  # Weight for environment score
        self.gamma = gamma  # Weight for environmental impact
        self.self_score = 0
        self.integrated_score = 0
        self.final_score = 0
        self.environmental_impact_score = 0
        self.resources = {
            "money": 100,
            "reputation": 70,
        }

    def select_action(self, observation):
        # To be implemented by subclasses
        pass

    def calculate_utility_score(self, gvd_score, asf_score):
        """
        Calculate the player's utility score using the three-parameter function:
        utility = α * personal_score + β * GVD_score + γ * ASF_score
        where:
        - α: weight for personal resources (money + reputation)
        - β: weight for G, V, D parameters (Greenery, Vitality, Density)
        - γ: weight for A, S, F parameters (Adaptability, Sustainability, Flood_Resistance)
        """
        personal_score = self.resources["money"] + self.resources["reputation"]
        utility = (
            self.alpha * personal_score + self.beta * gvd_score + self.gamma * asf_score
        )
        return utility

    def update_state(self, reward, info, gvd_score=0, asf_score=0):
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

        # Calculate integrated score using the new utility function
        self.integrated_score = self.calculate_utility_score(gvd_score, asf_score)

        # Log the updated resources and scores
        logger.debug(f"players: Player {self.name} updated resources: {self.resources}")
        logger.debug(
            f"players: Player {self.name} integrated score: {self.integrated_score} (α={self.alpha}, β={self.beta}, γ={self.gamma})"
        )


class InterestDrivenPlayer(BasePlayer):
    def __init__(self, name):
        # Interest-Driven: α=0.6, β=0.2, γ=0.2
        super().__init__(name, alpha=0.6, beta=0.2, gamma=0.2)
        # Economic focus - higher money, balanced reputation
        self.resources = {
            "money": 100,
            "reputation": 80,
        }

    def update_state(self, reward, info, gvd_score=0, asf_score=0):
        super().update_state(reward, info, gvd_score, asf_score)

        # Additional interest-driven specific logic if needed
        logger.debug(
            f"players: Interest-Driven Player {self.name} focuses on personal gains (α={self.alpha})"
        )


class AltruisticPlayer(BasePlayer):
    def __init__(self, name):
        # Altruistic: α=0.2, β=0.6, γ=0.2
        super().__init__(name, alpha=0.2, beta=0.6, gamma=0.2)
        # Community focus - balanced with slight reputation advantage
        self.resources = {
            "money": 80,
            "reputation": 100,
        }

    def update_state(self, reward, info, gvd_score=0, asf_score=0):
        super().update_state(reward, info, gvd_score, asf_score)

        # Additional altruistic specific logic if needed
        logger.debug(
            f"players: Altruistic Player {self.name} prioritizes environment (β={self.beta})"
        )


class BalancedPlayer(BasePlayer):
    def __init__(self, name):
        # Balanced: α=0.34, β=0.33, γ=0.33
        super().__init__(name, alpha=0.34, beta=0.33, gamma=0.33)
        # Balanced approach - equal resources
        self.resources = {
            "money": 90,
            "reputation": 90,
        }

    def update_state(self, reward, info, gvd_score=0, asf_score=0):
        super().update_state(reward, info, gvd_score, asf_score)

        # Additional balanced specific logic if needed
        logger.debug(
            f"players: Balanced Player {self.name} balances all factors equally"
        )


class EnvironmentalFocusedPlayer(BasePlayer):
    def __init__(self, name):
        # Environmental Focused: α=0.2, β=0.2, γ=0.6
        super().__init__(name, alpha=0.2, beta=0.2, gamma=0.6)
        # Environmental focus - highest reputation, moderate money
        self.resources = {
            "money": 80,
            "reputation": 100,
        }

    def update_state(self, reward, info, gvd_score=0, asf_score=0):
        super().update_state(reward, info, gvd_score, asf_score)

        # Additional environmental focused specific logic if needed
        logger.debug(
            f"players: Environmental Focused Player {self.name} prioritizes environmental impact (γ={self.gamma})"
        )
