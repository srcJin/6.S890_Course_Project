# src/envs/simcity/environment.py

from pettingzoo.utils import AECEnv, agent_selector
from gymnasium import spaces
import numpy as np
from .players import BasePlayer, BalancedPlayer, InterestDrivenPlayer, AltruisticPlayer
from .config import BUILDING_TYPES, BUILDING_COSTS, BUILDING_UTILITIES, BUILDING_EFFECTS
from .log import (
    display_current_turn,
    display_board,
    display_player_stats,
    log_environment_score,
)
from utils.logging import get_logger

logger = get_logger(log_file_path="simulation.log")

NO_OP = 0
BUILDING_TYPES = ["Park", "House", "Shop"]
NUM_BUILDING_TYPES = len(BUILDING_TYPES)

class SimCityEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "SimCityEnv"}

    def __init__(self, grid_x=4, grid_y=4, common_reward=False, reward_alpha=0.5, reward_beta=0.5):
        super().__init__()
        self.BUILDING_COSTS = BUILDING_COSTS
        self.BUILDING_TYPES = BUILDING_TYPES
        self.BUILDING_UTILITIES = BUILDING_UTILITIES
        self.BUILDING_EFFECTS = BUILDING_EFFECTS
        self.common_reward = common_reward
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_cells = grid_x * grid_y
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.agents = ["P1", "P2", "P3"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.total_actions = 1 + (NUM_BUILDING_TYPES * self.num_cells)  # no_op + (NUM_BUILDING_TYPES * self.num_cells)

        self.action_spaces = {agent: spaces.Discrete(self.total_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=100,
                        shape=(self.grid_x, self.grid_y, 3),
                        dtype=np.int32,
                    ),
                    "resources": spaces.Dict(
                        {
                            "money": spaces.Discrete(100),
                            "reputation": spaces.Discrete(100),
                        }
                    ),
                    "builders": spaces.Box(
                        low=-1,
                        high=len(self.agents) - 1,
                        shape=(self.grid_x, self.grid_y),
                        dtype=np.int32,
                    ),
                    "building_types": spaces.Box(
                        low=0,
                        high=NUM_BUILDING_TYPES - 1,
                        shape=(self.grid_x, self.grid_y),
                        dtype=np.int32,
                    ),
                }
            )
            for agent in self.agents
        }

        # Mode 1: all players are balanced player
        # self.players = {agent: BalancedPlayer(agent) for agent in self.agents}

        # Mode 2: iteratively assign player types, we have 3 types of players
        self.players = {}
        for i, agent in enumerate(self.agents):
            if i % 3 == 0:
                self.players[agent] = AltruisticPlayer(agent)
            elif i % 3 == 1:
                self.players[agent] = BalancedPlayer(agent)
            else:
                self.players[agent] = InterestDrivenPlayer(agent)

        # Initialize previous integrated scores for reward calculation
        self.previous_integrated_score = {agent: 0 for agent in self.agents}

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.grid = np.empty((self.grid_x, self.grid_y, 3), dtype=np.int32)
        self.grid[:, :, 0] = 15  # G
        self.grid[:, :, 1] = 20  # V
        self.grid[:, :, 2] = 30  # D

        self.buildings = np.full((self.grid_x, self.grid_y), None)
        self.builders = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)
        self.building_types = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)  # -1 for no building

        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        for player in self.players.values():
            player.resources = {"money": 50, "reputation": 50}
            player.self_score = 0
            player.integrated_score = 0

        self.env_score = self.calculate_environment_score()["env_score"]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()
        self.num_moves = 0
        self.has_reset = True
        self.agent_index = 0

        # Reset previous integrated scores
        self.previous_integrated_score = {agent: 0 for agent in self.agents}

        logger.debug("environment: Environment reset completed.")
        return self.observe(self.agent_selection), {}

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before calling step.")

        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(action)
            return

        self._cumulative_rewards[agent] = 0
        self.infos[agent] = {}
        reward = 0
        info_resources = {}

        logger.debug(f"environment: Agent {agent} received action {action}")

        building_type, x, y = self.decode_action(action)
        logger.debug(f"environment: Decoded action for {agent}: Building={building_type}, Cell=({x},{y})")

        if action == NO_OP:
            logger.debug(f"environment: Agent {agent} performed No-op.")
        else:
            if self.buildings[x][y] is not None:
                build_on_occupied_penalty = 999999999999999999
                reward -= build_on_occupied_penalty
                logger.debug(f"environment: Agent {agent} tried to build on an occupied cell ({x},{y}). Penalty: {build_on_occupied_penalty}.")
            else:
                building_cost = BUILDING_COSTS[building_type]
                player_resources = self.players[agent].resources
                if player_resources["money"] < building_cost["money"] or player_resources["reputation"] < building_cost["reputation"]:
                    # Penalty for not having enough resources
                    not_enough_resource_penalty = 999999999999999999
                    reward -= not_enough_resource_penalty
                    logger.debug(f"environment: Agent {agent} does not have enough resources to build {building_type} at ({x},{y}). Penalty: {not_enough_resource_penalty}.")
                else:
                    # Deduct resources
                    player_resources["money"] -= building_cost["money"]
                    player_resources["reputation"] -= building_cost["reputation"]
                    logger.debug(f"environment: Agent {agent} resources after building: {player_resources}")

                    # Update buildings and builders
                    self.buildings[x][y] = {"type": building_type, "turn_built": self.num_moves}
                    self.builders[x][y] = self.agents.index(agent)  # 0 for P1, 1 for P2, 2 for P3
                    self.building_types[x][y] = BUILDING_TYPES.index(building_type)  # 0 for Park, 1 for House, 2 for Shop


                    # Update self grid score
                    building_effect = BUILDING_EFFECTS[building_type]
                    self.grid[x][y][0] += building_effect["G"]
                    self.grid[x][y][1] += building_effect["V"]
                    self.grid[x][y][2] += building_effect["D"]

                    # Update neighbors score
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                            self.grid[nx][ny][0] += building_effect["neighbors"]["G"]
                            self.grid[nx][ny][1] += building_effect["neighbors"]["V"]
                            self.grid[nx][ny][2] += building_effect["neighbors"]["D"]

                    building_utility = BUILDING_UTILITIES[building_type]
                    immediate_reward = building_utility["money"] + building_utility["reputation"]
                    reward += immediate_reward
                    logger.debug(f"environment: Agent {agent} built {building_type} at ({x},{y}) gaining immediate reward: {immediate_reward}")

                    info_resources = {
                        "money": -building_cost["money"] + building_utility["money"],
                        "reputation": -building_cost["reputation"] + building_utility["reputation"],
                    }

        # Update utilities based on buildings
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                if self.buildings[gx][gy] is not None:
                    b_type = self.buildings[gx][gy]["type"]
                    b_utility = BUILDING_UTILITIES[b_type]
                    self.players[agent].resources["money"] += b_utility["money"]
                    self.players[agent].resources["reputation"] += b_utility["reputation"]
                    self.players[agent].self_score += (b_utility["money"] + b_utility["reputation"])

        # Calculate environment score
        self.env_score = self.calculate_environment_score()["env_score"]
        logger.debug(f"environment: env_score after step: {self.env_score}")


        # Mode 1: all players intergrated score is the same
        # alpha, beta = 0.5, 0.5
        # self.players[agent].integrated_score = alpha * self.players[agent].self_score + beta * self.env_score

        # Mode 2: assign different alpha and beta for different player types
        if isinstance(self.players[agent], InterestDrivenPlayer):
            alpha, beta = 0.8, 0.2
            logger.debug(f"environment: Player {agent} is InterestDrivenPlayer, using alpha={alpha}, beta={beta}")
        elif isinstance(self.players[agent], AltruisticPlayer):
            alpha, beta = 0.2, 0.8
            logger.debug(f"environment: Player {agent} is AltruisticPlayer, using alpha={alpha}, beta={beta}")
        elif isinstance(self.players[agent], BalancedPlayer):
            alpha, beta = 0.5, 0.5
            logger.debug(f"environment: Player {agent} is BalancedPlayer, using alpha={alpha}, beta={beta}")
        else:
            alpha, beta = 0.5, 0.5  # Default for unknown player types
            logger.debug(f"environment: Player {agent} is unknown type, using alpha={alpha}, beta={beta}")

        self.players[agent].integrated_score = (alpha * self.players[agent].self_score + beta * self.env_score)
        logger.debug(
            f"environment: Player {agent} - Self score: {self.players[agent].self_score}, "
            f"Integrated score: {self.players[agent].integrated_score} (alpha={alpha}, beta={beta})"
        )

        log_environment_score(self.num_moves, self.env_score)
        self.infos[agent]["resources"] = info_resources

        # Compute and assign reward based on integrated_score and delta
        reward = self.compute_reward(agent, self.reward_alpha, self.reward_beta)

        # Increment move count and check for termination
        self.num_moves += 1
        if self.is_game_over():
            for ag in self.agents:
                self.terminations[ag] = True

        # Select next agent
        self.agent_selection = self._agent_selector.next()
        logger.debug(f"environment: Agent selection after step: {self.agent_selection}")
        self.has_reset = True

        # Assign the computed reward
        self.rewards[agent] = reward

    def decode_action(self, action):
        if not isinstance(action, int) or action < 0 or action >= 1 + NUM_BUILDING_TYPES * self.num_cells:
            logger.warning(f"environment: Received invalid action: {action}, defaulting to No-op.")
            return "Park", 0, 0

        if action == NO_OP:
            return "Park", 0, 0

        cell_id = (action - 1) % self.num_cells
        building_type_index = (action - 1) // self.num_cells
        building_type = BUILDING_TYPES[building_type_index]

        x = cell_id // self.grid_y
        y = cell_id % self.grid_y

        return building_type, x, y

    def calculate_environment_score(self):
        G_avg = np.mean(self.grid[:, :, 0])
        V_avg = np.mean(self.grid[:, :, 1])
        D_avg = np.mean(self.grid[:, :, 2])
        env_score = (G_avg + V_avg + D_avg) / 3
        logger.debug(f"environment: calculate_environment_score G_avg={G_avg}, V_avg={V_avg}, D_avg={D_avg}, env_score={env_score}")

        return {"G_avg": G_avg, "V_avg": V_avg, "D_avg": D_avg, "env_score": env_score}

    def compute_reward(self, agent, reward_alpha, reward_beta):
        """
        Compute the reward for an agent based on the change in integrated_score and the current integrated_score.
        Formula: reward = alpha * delta + beta * integrated_score
        """
        current_score = self.players[agent].integrated_score
        previous_score = self.previous_integrated_score[agent]
        delta = current_score - previous_score
        self.previous_integrated_score[agent] = current_score

        reward = reward_alpha * delta + reward_beta * current_score
        logger.debug(
            f"environment: Compute reward for {agent} - Delta: {delta}, "
            f"Integrated Score: {current_score}, Reward: {reward} (alpha={reward_alpha}, beta={reward_beta}"
        )
        return reward


    def is_game_over(self):
        board_filled = np.all(self.buildings != None)
        logger.debug(f"environment: Game over check - board_filled={board_filled}")
        return board_filled

    def observe(self, agent):
        observation = {
            "grid": self.grid.copy(),
            "resources": self.players[agent].resources.copy(),
            "builders": self.builders.copy(),
            "building_types": self.building_types.copy(),
        }
        logger.debug(f"environment: observe Observation for {agent}: {observation}")
        return observation

    def render(self, mode="human"):
        display_grid = ""
        for x in range(self.grid_x):
            row = ""
            for y in range(self.grid_y):
                b = self.buildings[x][y]
                if b is None:
                    row += "[ ]"
                elif b["type"] == "Park":
                    row += "[P]"
                elif b["type"] == "House":
                    row += "[H]"
                elif b["type"] == "Shop":
                    row += "[S]"
            display_grid += row + "\n"
        logger.debug(f"environment: Render output:\n{display_grid}")
        print(display_grid)

    def close(self):
        logger.debug("environment: Closing environment.")
        pass

    def _was_done_step(self, action):
        self._cumulative_rewards[self.agent_selection] = 0
        self.rewards[self.agent_selection] = 0
        self.infos[self.agent_selection] = {}
        self.agent_selection = self._agent_selector.next()

    def get_observation(self, agent_id):
        logger.debug("environment: get_observation")
        return self.observe(agent_id)
