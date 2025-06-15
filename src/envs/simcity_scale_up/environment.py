# src/envs/simcity_scale_up/environment.py
# Urban Resilience-Focused SimCity Environment

from pettingzoo.utils import AECEnv, agent_selector
from gymnasium import spaces
import numpy as np
from .players import BasePlayer, BalancedPlayer, InterestDrivenPlayer, AltruisticPlayer
from .config import (
    BUILDING_TYPES,
    BUILDING_COSTS,
    BUILDING_UTILITIES,
    BUILDING_EFFECTS,
    TERRAIN_TYPES,
    PREBUILT_INFRASTRUCTURE,
    DEFAULT_GRID_LAYOUT,
    DEFAULT_TERRAIN_ASSIGNMENT,
    DEFAULT_INFRASTRUCTURE_ASSIGNMENT,
)
from .log import (
    display_current_turn,
    display_board,
    display_player_stats,
    log_environment_score,
)
from utils.logging import get_logger

logger = get_logger(log_file_path="simulation_scale_up.log")

NO_OP = 0
BUILDING_TYPES = [
    "GreenPark",
    "ResilientHouse",
    "CommunityHub",
    "SolarGrid",
    "FloodBarrier",
]
NUM_BUILDING_TYPES = len(BUILDING_TYPES)


class SimCityScaleUpEnv(AECEnv):
    """
    Urban Resilience-Focused SimCity Environment

    Grid Parameters:
    - S: Sustainability (environmental impact, renewable energy)
    - W: Well-being (community health, social cohesion)
    - R: Resilience (disaster preparedness, adaptability)
    - C: Climate (carbon footprint, climate adaptation)

    Building Types:
    - GreenPark: Urban green infrastructure for sustainability and well-being
    - ResilientHouse: Climate-adapted housing for communities
    - CommunityHub: Social resilience centers for community cohesion
    - SolarGrid: Renewable energy infrastructure
    - FloodBarrier: Climate protection infrastructure
    """

    metadata = {"render_modes": ["human"], "name": "SimCityScaleUpEnv"}

    def __init__(
        self, grid_x=8, grid_y=8, common_reward=False, reward_alpha=0.5, reward_beta=0.5
    ):
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

        # 4 players for scaled-up environment
        self.agents = ["P1", "P2", "P3", "P4"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.total_actions = 1 + (
            NUM_BUILDING_TYPES * self.num_cells
        )  # no_op + (NUM_BUILDING_TYPES * self.num_cells)

        self.action_spaces = {
            agent: spaces.Discrete(self.total_actions) for agent in self.agents
        }

        # Observation space with 4 grid parameters (S, W, R, C)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=100,
                        shape=(self.grid_x, self.grid_y, 4),  # 4 parameters: S, W, R, C
                        dtype=np.int32,
                    ),
                    "resources": spaces.Dict(
                        {
                            "money": spaces.Discrete(200),  # Increased for larger scale
                            "reputation": spaces.Discrete(200),
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

        # Player type assignment - diverse resilience perspectives
        self.players = {}
        for i, agent in enumerate(self.agents):
            if i % 4 == 0:
                self.players[agent] = AltruisticPlayer(agent)  # Community-focused
            elif i % 4 == 1:
                self.players[agent] = BalancedPlayer(agent)  # Balanced approach
            elif i % 4 == 2:
                self.players[agent] = InterestDrivenPlayer(agent)  # Economic efficiency
            else:
                self.players[agent] = BalancedPlayer(agent)  # Environmental focus

        # Initialize previous integrated scores for reward calculation
        self.previous_integrated_score = {agent: 0 for agent in self.agents}

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Initialize grid with baseline resilience values
        self.grid = np.empty((self.grid_x, self.grid_y, 4), dtype=np.int32)
        self.grid[:, :, 0] = 20  # S - Sustainability baseline
        self.grid[:, :, 1] = 25  # W - Well-being baseline
        self.grid[:, :, 2] = 15  # R - Resilience baseline
        self.grid[:, :, 3] = 10  # C - Climate adaptation baseline

        # Initialize grid layout (0=buildable, 1=terrain, 2=infrastructure)
        self.grid_layout = np.array(DEFAULT_GRID_LAYOUT)

        # Initialize terrain and infrastructure mappings
        self.terrain_map = DEFAULT_TERRAIN_ASSIGNMENT.copy()
        self.infrastructure_map = DEFAULT_INFRASTRUCTURE_ASSIGNMENT.copy()

        self.buildings = np.full((self.grid_x, self.grid_y), None)
        self.builders = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)
        self.building_types = np.full(
            (self.grid_x, self.grid_y), -1, dtype=np.int32
        )  # -1 for no building

        # Apply pre-built infrastructure effects to grid
        self._apply_prebuilt_infrastructure()

        self.individual_rewards_list = {agent: 0 for agent in self.agents}
        self.common_reward_value = 0
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Higher starting resources for scaled environment
        for player in self.players.values():
            player.resources = {"money": 80, "reputation": 80}
            player.self_score = 0
            player.integrated_score = 0

        self.env_score = self.calculate_environment_score()["env_score"]
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()
        self.num_moves = 0
        self.has_reset = True
        self.agent_index = 0

        # Reset previous integrated scores
        self.previous_integrated_score = {agent: 0 for agent in self.agents}

        logger.debug("environment: Urban Resilience Environment reset completed.")
        return self.observe(self.agent_selection), {}

    def _apply_prebuilt_infrastructure(self):
        """Apply effects of pre-built infrastructure to the grid."""
        for (x, y), infrastructure_type in self.infrastructure_map.items():
            if infrastructure_type in PREBUILT_INFRASTRUCTURE:
                infra = PREBUILT_INFRASTRUCTURE[infrastructure_type]

                # Apply direct effects to the infrastructure cell
                self.grid[x][y][0] += infra["effects"]["S"]  # Sustainability
                self.grid[x][y][1] += infra["effects"]["W"]  # Well-being
                self.grid[x][y][2] += infra["effects"]["R"]  # Resilience
                self.grid[x][y][3] += infra["effects"]["C"]  # Climate

                # Apply neighbor effects
                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                        self.grid[nx][ny][0] += infra["neighbor_effects"]["S"]
                        self.grid[nx][ny][1] += infra["neighbor_effects"]["W"]
                        self.grid[nx][ny][2] += infra["neighbor_effects"]["R"]
                        self.grid[nx][ny][3] += infra["neighbor_effects"]["C"]

                logger.debug(
                    f"environment: Applied {infrastructure_type} effects at ({x},{y})"
                )

    def _is_buildable(self, x, y):
        """Check if a cell is buildable (not terrain or infrastructure)."""
        return self.grid_layout[x][y] == 0 and self.buildings[x][y] is None

        self.buildings = np.full((self.grid_x, self.grid_y), None)
        self.builders = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)
        self.building_types = np.full(
            (self.grid_x, self.grid_y), -1, dtype=np.int32
        )  # -1 for no building

        # Apply pre-built infrastructure effects to grid
        self._apply_prebuilt_infrastructure()

        self.individual_rewards_list = {agent: 0 for agent in self.agents}
        self.common_reward_value = 0
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Higher starting resources for scaled environment
        for player in self.players.values():
            player.resources = {"money": 80, "reputation": 80}
            player.self_score = 0
            player.integrated_score = 0

        self.env_score = self.calculate_environment_score()["env_score"]
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
        logger.debug("Calling environment step")
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before calling step.")

        agent = self.agent_selection
        logger.debug(f"environment: Agent {agent} is taking step.")

        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(action)
            return

        self.infos[agent] = {}
        reward = 0
        info_resources = {}

        logger.debug(f"environment: Agent {agent} received action {action}")

        building_type, x, y = self.decode_action(action)
        logger.debug(
            f"environment: Decoded action for {agent}: Building={building_type}, Cell=({x},{y})"
        )

        if action == NO_OP:
            logger.debug(f"environment: Agent {agent} performed No-op.")
        else:
            # Check if location is buildable (not terrain or infrastructure)
            if not self._is_buildable(x, y):
                terrain_penalty = -999999999999999999
                reward += terrain_penalty
                logger.debug(
                    f"environment: Agent {agent} tried to build on non-buildable terrain at ({x},{y}). Penalty: {terrain_penalty}."
                )
            elif self.buildings[x][y] is not None:
                build_on_occupied_penalty = -999999999999999999
                reward += build_on_occupied_penalty
                logger.debug(
                    f"environment: Agent {agent} tried to build on an occupied cell ({x},{y}). Penalty: {build_on_occupied_penalty}."
                )
            else:
                building_cost = BUILDING_COSTS[building_type]
                player_resources = self.players[agent].resources
                if (
                    player_resources["money"] < building_cost["money"]
                    or player_resources["reputation"] < building_cost["reputation"]
                ):
                    # Penalty for not having enough resources
                    not_enough_resource_penalty = -999999999999999999
                    reward += not_enough_resource_penalty
                    logger.debug(
                        f"environment: Agent {agent} does not have enough resources to build {building_type} at ({x},{y}). Penalty: {not_enough_resource_penalty}."
                    )
                else:
                    # Deduct resources
                    player_resources["money"] -= building_cost["money"]
                    player_resources["reputation"] -= building_cost["reputation"]
                    logger.debug(
                        f"environment: Agent {agent} resources after building: {player_resources}"
                    )

                    # Update buildings and builders
                    self.buildings[x][y] = {
                        "type": building_type,
                        "turn_built": self.num_moves,
                    }
                    self.builders[x][y] = self.agents.index(
                        agent
                    )  # 0 for P1, 1 for P2, 2 for P3, 3 for P4
                    self.building_types[x][y] = BUILDING_TYPES.index(building_type)

                    # Update self grid score with new parameters (S, W, R, C)
                    building_effect = BUILDING_EFFECTS[building_type]
                    self.grid[x][y][0] += building_effect["S"]  # Sustainability
                    self.grid[x][y][1] += building_effect["W"]  # Well-being
                    self.grid[x][y][2] += building_effect["R"]  # Resilience
                    self.grid[x][y][3] += building_effect["C"]  # Climate

                    # Update neighbors score
                    for dx, dy in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (1, 1),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                    ]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                            self.grid[nx][ny][0] += building_effect["neighbors"]["S"]
                            self.grid[nx][ny][1] += building_effect["neighbors"]["W"]
                            self.grid[nx][ny][2] += building_effect["neighbors"]["R"]
                            self.grid[nx][ny][3] += building_effect["neighbors"]["C"]

                    building_utility = BUILDING_UTILITIES[building_type]
                    immediate_reward = (
                        building_utility["money"] + building_utility["reputation"]
                    )
                    reward += immediate_reward
                    logger.debug(
                        f"environment: Agent {agent} built {building_type} at ({x},{y}) gaining immediate reward: {immediate_reward}"
                    )

                    info_resources = {
                        "money": -building_cost["money"] + building_utility["money"],
                        "reputation": -building_cost["reputation"]
                        + building_utility["reputation"],
                    }

        # Update utilities based on buildings
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                if self.buildings[gx][gy] is not None:
                    b_type = self.buildings[gx][gy]["type"]
                    b_utility = BUILDING_UTILITIES[b_type]
                    self.players[agent].resources["money"] += b_utility["money"]
                    self.players[agent].resources["reputation"] += b_utility[
                        "reputation"
                    ]
                    self.players[agent].self_score += (
                        b_utility["money"] + b_utility["reputation"]
                    )

        # Calculate environment score
        self.env_score = self.calculate_environment_score()["env_score"]
        logger.debug(f"environment: env_score after step: {self.env_score}")

        # Mode 1: all players intergrated score is the same
        # alpha, beta = 0.5, 0.5
        # self.players[agent].integrated_score = alpha * self.players[agent].self_score + beta * self.env_score

        # Mode 2: assign different alpha and beta for different player types
        if isinstance(self.players[agent], InterestDrivenPlayer):
            alpha, beta = 0.8, 0.2
            logger.debug(
                f"environment: Player {agent} is InterestDrivenPlayer, using alpha={alpha}, beta={beta}"
            )
        elif isinstance(self.players[agent], AltruisticPlayer):
            alpha, beta = 0.2, 0.8
            logger.debug(
                f"environment: Player {agent} is AltruisticPlayer, using alpha={alpha}, beta={beta}"
            )
        elif isinstance(self.players[agent], BalancedPlayer):
            alpha, beta = 0.5, 0.5
            logger.debug(
                f"environment: Player {agent} is BalancedPlayer, using alpha={alpha}, beta={beta}"
            )
        else:
            alpha, beta = 0.5, 0.5  # Default for unknown player types
            logger.debug(
                f"environment: Player {agent} is unknown type, using alpha={alpha}, beta={beta}"
            )

        # Update player's integrated score
        self.players[agent].integrated_score = (
            alpha * self.players[agent].self_score + beta * self.env_score
        )
        logger.debug(
            f"environment: Player {agent} - Self score: {self.players[agent].self_score}, "
            f"Integrated score: {self.players[agent].integrated_score} (alpha={alpha}, beta={beta})"
        )

        log_environment_score(self.num_moves, self.env_score)
        self.infos[agent]["resources"] = info_resources

        # Compute and assign reward based on integrated_score and delta
        self.individual_rewards_list[agent] = self.compute_individual_reward(
            agent, self.reward_alpha, self.reward_beta
        )
        self.common_reward_value = self.compute_common_reward_value(
            self.reward_alpha, self.reward_beta
        )

        # Increment move count and check for termination
        self.num_moves += 1
        if self.is_game_over():
            for ag in self.agents:
                self.terminations[ag] = True

        # Select next agent
        self.agent_selection = self._agent_selector.next()
        logger.debug(f"environment: Agent selection after step: {self.agent_selection}")
        self.has_reset = True

    def decode_action(self, action):
        if (
            not isinstance(action, int)
            or action < 0
            or action >= 1 + NUM_BUILDING_TYPES * self.num_cells
        ):
            logger.warning(
                f"environment: Received invalid action: {action}, defaulting to No-op."
            )
            return "GreenPark", 0, 0

        if action == NO_OP:
            return "GreenPark", 0, 0

        cell_id = (action - 1) % self.num_cells
        building_type_index = (action - 1) // self.num_cells
        building_type = BUILDING_TYPES[building_type_index]

        x = cell_id // self.grid_y
        y = cell_id % self.grid_y

        return building_type, x, y

    def calculate_environment_score(self):
        S_avg = np.mean(self.grid[:, :, 0])  # Sustainability
        W_avg = np.mean(self.grid[:, :, 1])  # Well-being
        R_avg = np.mean(self.grid[:, :, 2])  # Resilience
        C_avg = np.mean(self.grid[:, :, 3])  # Climate
        env_score = (S_avg + W_avg + R_avg + C_avg) / 4
        logger.debug(
            f"environment: calculate_environment_score S_avg={S_avg}, W_avg={W_avg}, R_avg={R_avg}, C_avg={C_avg}, env_score={env_score}"
        )

        return {
            "S_avg": S_avg,
            "W_avg": W_avg,
            "R_avg": R_avg,
            "C_avg": C_avg,
            "env_score": env_score,
        }

    def compute_individual_reward(self, agent, reward_alpha, reward_beta):
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
            f"environment: Compute individual reward for {agent} - Delta: {delta}, "
            f"Integrated Score: {current_score}, Reward: {reward} (alpha={reward_alpha}, beta={reward_beta})"
        )
        return reward

    def compute_common_reward_value(self, reward_alpha, reward_beta):
        # sum of all players' integrated scores
        common_reward_value = sum(
            [player.integrated_score for player in self.players.values()]
        )
        logger.debug(
            f"environment: Compute common reward - Common Reward: {common_reward_value}"
        )
        return common_reward_value

    def is_game_over(self):
        # Game is over when all buildable spaces are filled
        buildable_spaces_filled = True
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                if self._is_buildable(x, y):
                    buildable_spaces_filled = False
                    break
            if not buildable_spaces_filled:
                break

        logger.debug(
            f"environment: Game over check - buildable_spaces_filled={buildable_spaces_filled}"
        )
        return buildable_spaces_filled

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
                # Check for pre-built infrastructure first
                if (x, y) in self.infrastructure_map:
                    infra_type = self.infrastructure_map[(x, y)]
                    symbol = PREBUILT_INFRASTRUCTURE[infra_type]["symbol"]
                    row += f"[{symbol}]"
                # Check for terrain
                elif (x, y) in self.terrain_map:
                    terrain_type = self.terrain_map[(x, y)]
                    symbol = TERRAIN_TYPES[terrain_type]["symbol"][
                        :1
                    ]  # Take first character
                    row += f"[{symbol}]"
                # Check for buildings
                elif self.buildings[x][y] is not None:
                    b = self.buildings[x][y]
                    if b["type"] == "GreenPark":
                        row += "[G]"
                    elif b["type"] == "ResilientHouse":
                        row += "[R]"
                    elif b["type"] == "CommunityHub":
                        row += "[C]"
                    elif b["type"] == "SolarGrid":
                        row += "[S]"
                    elif b["type"] == "FloodBarrier":
                        row += "[F]"
                    else:
                        row += "[?]"
                else:
                    row += "[ ]"
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
