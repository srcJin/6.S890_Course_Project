# environment.py
from pettingzoo.utils import AECEnv, agent_selector
from gymnasium import spaces
import numpy as np
from .players import BasePlayer, BalancedPlayer
from .config import BUILDING_TYPES, BUILDING_COSTS, BUILDING_UTILITIES, BUILDING_EFFECTS
from .log import (
    display_current_turn,
    display_board,
    display_player_stats,
    log_environment_score,
)

from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")


class SimCityEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, common_reward=False):
        super().__init__()
        self.common_reward = common_reward
        self.grid_size = 4
        self.agents = ["P1", "P2", "P3"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.action_spaces = {
            agent: spaces.Discrete(self.grid_size**2 * len(BUILDING_TYPES))
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=100,
                        shape=(self.grid_size, self.grid_size, 3),
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
                        shape=(self.grid_size, self.grid_size),
                        dtype=np.int32,
                    ),
                }
            )
            for agent in self.agents
        }
        # Initialize players
        self.players = {
            agent: BalancedPlayer(agent) for agent in self.agents
        }
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment with optional seed and options"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the game state
        # Set all grid cells to the specified G, V, D values
        self.grid = np.empty((self.grid_size, self.grid_size, 3), dtype=np.int32)
        self.grid[:, :, 0] = 15  # Set G values
        self.grid[:, :, 1] = 20  # Set V values
        self.grid[:, :, 2] = 30  # Set D values

        self.buildings = np.full(
            (self.grid_size, self.grid_size), None
        )  # To track building types
        self.builders = np.full(
            (self.grid_size, self.grid_size), -1, dtype=np.int32
        )  # -1 indicates no builder
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        # Reset player resources
        for player in self.players.values():
            player.resources = {
                "money": 20,
                "reputation": 20,
            }
            player.self_score = 0
            player.integrated_score = 0
        
        self.env_score = self.calculate_environment_score()["env_score"]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()
        self.num_moves = 0
        self.has_reset = True
        self.agent_index = 0
        
        # Return initial observation and info
        return self.observe(self.agent_selection), {}

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("environment.py Environment must be reset before calling step.")

        agent = self.agent_selection

        if self.terminations[agent]:
            # No action required if agent is terminated
            self._was_done_step(action)
            return

        # Reset the cumulative reward for this agent
        self._cumulative_rewards[agent] = 0

        # Initialize info
        self.infos[agent] = {}

        # Initialize reward for this step
        reward = 0
        info_resources = {}

        # Decode action
        building_type, x, y = self.decode_action(action)

        # Check if the cell is empty
        if self.buildings[x][y] is not None:
            # Cell already occupied: assign a penalty
            reward -= 5  # Example penalty
        else:
            building_cost = BUILDING_COSTS[building_type]
            player_resources = self.players[agent].resources

            # Check if the player has enough resources for initial build
            if (
                player_resources["money"] < building_cost["money"]
                or player_resources["reputation"] < building_cost["reputation"]
            ):
                # Not enough resources: assign a penalty
                reward -= 5
            else:
                # Deduct initial costs
                player_resources["money"] -= building_cost["money"]
                player_resources["reputation"] -= building_cost["reputation"]
                logger.debug(f"environment.py Player {agent} resources after action: {self.players[agent].resources}")
                logger.debug(f"environment.py Player {agent} scores: {self.players[agent].self_score}, {self.players[agent].integrated_score}")

                # Place the building
                self.buildings[x][y] = {
                    "type": building_type,
                    "turn_built": self.num_moves,  # Track the turn it was built
                }
                self.builders[x][y] = self.agents.index(agent)

                # Update the grid with building effects
                building_effect = BUILDING_EFFECTS[building_type]
                self.grid[x][y][0] += building_effect["G"]
                self.grid[x][y][1] += building_effect["V"]
                self.grid[x][y][2] += building_effect["D"]

                # Apply effects to neighboring cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.grid[nx][ny][0] += building_effect["neighbors"]["G"]
                        self.grid[nx][ny][1] += building_effect["neighbors"]["V"]
                        self.grid[nx][ny][2] += building_effect["neighbors"]["D"]

                # Add immediate utility reward for the initial build
                building_utility = BUILDING_UTILITIES[building_type]
                reward += building_utility["money"] + building_utility["reputation"]

                # Record the resource changes
                info_resources = {
                    "money": -building_cost["money"] + building_utility["money"],
                    "reputation": -building_cost["reputation"]
                    + building_utility["reputation"],
                }

        # Apply ongoing building utilities (passive income or effects)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.buildings[x][y] is not None:
                    building = self.buildings[x][y]
                    building_type = building["type"]

                    # Apply utilities for each building on the grid
                    building_utility = BUILDING_UTILITIES[building_type]
                    self.players[agent].resources["money"] += building_utility["money"]
                    self.players[agent].resources["reputation"] += building_utility[
                        "reputation"
                    ]

                    # Accumulate the passive utility in self_score
                    reward += building_utility["money"] + building_utility["reputation"]

        # Update player's self_score
        self.players[agent].self_score += reward

        # Update cumulative reward for the agent
        self._cumulative_rewards[agent] += reward

        # Update the environment score
        self.env_score = self.calculate_environment_score()["env_score"]

        # Update the agent's integrated score
        alpha, beta = 0.5, 0.5  # Weights for self_score and environment score
        self.players[agent].integrated_score = (
            alpha * self.players[agent].self_score + beta * self.env_score
        )

        # Log the environment score
        log_environment_score(self.num_moves, self.env_score)

        # Prepare info to send to the player
        self.infos[agent]["resources"] = info_resources

        # Check for game end conditions
        self.num_moves += 1
        if self.is_game_over():
            for ag in self.agents:
                self.terminations[ag] = True

        # Advance to the next agent
        self.agent_selection = self._agent_selector.next()

        # Allow the next step
        self.has_reset = True

    def decode_action(self, action):
        # Decode action to building type and position
        if (
            not isinstance(action, int)
            or action < 0
            or action >= self.grid_size**2 * len(BUILDING_TYPES)
        ):
            # Invalid action, return default action
            return "Park", 0, 0
        total_positions = self.grid_size**2
        building_type_idx = action // total_positions
        position_idx = action % total_positions
        building_type = BUILDING_TYPES[building_type_idx]
        x = position_idx // self.grid_size
        y = position_idx % self.grid_size
        return building_type, x, y

    def calculate_environment_score(self):
        G_avg = np.mean(self.grid[:, :, 0])
        V_avg = np.mean(self.grid[:, :, 1])
        D_avg = np.mean(self.grid[:, :, 2])
        env_score = (1 / 3) * G_avg + (1 / 3) * V_avg + (1 / 3) * D_avg
        return {"G_avg": G_avg, "V_avg": V_avg, "D_avg": D_avg, "env_score": env_score}

    def is_game_over(self):
        board_filled = np.all(self.buildings != None)
        env_score = self.calculate_environment_score()["env_score"]
        logger.debug(f"is_game_over = {board_filled}, env_score={env_score}")
        # return board_filled or env_score < 10
        return board_filled

    def observe(self, agent):
        # Return the observation for the agent
        observation = {
            "grid": self.grid.copy(),
            "resources": self.players[
                agent
            ].resources.copy(),  # Use player's own resources
            "builders": self.builders.copy(),
        }
        return observation

    def render(self, mode="human"):
        # Render the game state as text
        display_grid = ""
        for x in range(self.grid_size):
            row = ""
            for y in range(self.grid_size):
                building = self.buildings[x][y]
                if building is None:
                    row += "[ ]"
                elif building == "Park":
                    row += "[P]"
                elif building == "House":
                    row += "[H]"
                elif building == "Shop":
                    row += "[S]"
            display_grid += row + "\n"
        print(display_grid)

    def close(self):
        pass

    def _was_done_step(self, action):
        # Required by PettingZoo
        self._cumulative_rewards[self.agent_selection] = 0
        self.rewards[self.agent_selection] = 0
        self.infos[self.agent_selection] = {}
        # Advance to the next agent
        self.agent_selection = self._agent_selector.next()

    def get_observation(self, agent_id):
        return self.observe(agent_id)

    def run_game(self):
        # Log initial state (Turn 0)
        display_current_turn(self.num_moves, self.agents[self.agent_index])
        display_board(self.buildings, self.grid, self.builders, self.agents)
        display_player_stats(self.players)

        while not self.is_game_over():
            agent_id = self.agents[self.agent_index]
            observation = self.get_observation(agent_id)
            action = self.players[agent_id].select_action(observation)
            self.step(action)
            # Log current turn and board state
            display_current_turn(self.num_moves, agent_id)
            display_board(self.buildings, self.grid, self.builders, self.agents)
            display_player_stats(self.players)
            self.agent_index = (self.agent_index + 1) % len(self.agents)
            log_environment_score(self.calculate_environment_score())
