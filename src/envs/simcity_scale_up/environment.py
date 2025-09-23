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
    PREBUILT_PROJECTS,
    DEFAULT_GRID_LAYOUT,
    DEFAULT_TERRAIN_ASSIGNMENT,
    DEFAULT_PREBUILT_ASSIGNMENT,
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
# Use building types from config
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
    Basic Development:
    - House: Standard residential housing (economic focus)
    - Shop: Commercial retail space (immediate profits)
    
    Resilience Projects:
    - GreenPark: Urban green infrastructure for sustainability and well-being
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

        # Initialize terrain and pre-built project mappings
        self.terrain_map = DEFAULT_TERRAIN_ASSIGNMENT.copy()
        self.prebuilt_map = DEFAULT_PREBUILT_ASSIGNMENT.copy()

        self.buildings = np.full((self.grid_x, self.grid_y), None)
        self.builders = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)
        self.building_types = np.full(
            (self.grid_x, self.grid_y), -1, dtype=np.int32
        )  # -1 for no building

        # Apply pre-built city projects effects to grid
        self._apply_prebuilt_projects()

        self.individual_rewards_list = {agent: 0 for agent in self.agents}
        self.common_reward_value = 0
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Tight starting resources to create competition pressure
        # 50 money + 50 reputation forces strategic choices:
        # - Can build ~6 Houses OR 3 Factories OR 4 GreenParks initially
        # - Must accumulate resources through utilities to access high-end buildings
        for player in self.players.values():
            player.resources = {"money": 50, "reputation": 50}
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

    def _apply_prebuilt_projects(self):
        """Apply effects of pre-built city projects to the grid."""
        for (x, y), project_type in self.prebuilt_map.items():
            if project_type in PREBUILT_PROJECTS:
                project = PREBUILT_PROJECTS[project_type]

                # Apply direct effects to the project cell
                self.grid[x][y][0] += project["effects"]["S"]  # Sustainability
                self.grid[x][y][1] += project["effects"]["W"]  # Well-being
                self.grid[x][y][2] += project["effects"]["R"]  # Resilience
                self.grid[x][y][3] += project["effects"]["C"]  # Climate

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
                        self.grid[nx][ny][0] += project["neighbor_effects"]["S"]
                        self.grid[nx][ny][1] += project["neighbor_effects"]["W"]
                        self.grid[nx][ny][2] += project["neighbor_effects"]["R"]
                        self.grid[nx][ny][3] += project["neighbor_effects"]["C"]

                logger.debug(
                    f"environment: Applied {project_type} effects at ({x},{y})"
                )

    def _is_buildable(self, x, y):
        """Check if a cell is buildable (not terrain or pre-built projects)."""
        return self.grid_layout[x][y] == 0 and self.buildings[x][y] is None

    def _can_afford(self, agent, building_type):
        """Check if agent can afford to build this building type."""
        building_cost = BUILDING_COSTS[building_type]
        player_resources = self.players[agent].resources
        return (player_resources["money"] >= building_cost["money"] and
                player_resources["reputation"] >= building_cost["reputation"])

    def get_available_actions(self, agent):
        """Return a binary mask of available actions for the agent."""
        avail_actions = np.zeros(self.total_actions, dtype=np.float32)
        avail_actions[0] = 1.0  # No-op is always available

        for action in range(1, self.total_actions):
            building_type, x, y = self.decode_action(action)
            if self._is_buildable(x, y) and self._can_afford(agent, building_type):
                avail_actions[action] = 1.0

        return avail_actions

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
            # Validate action (should not happen if using action masking properly)
            if not self._is_buildable(x, y):
                invalid_penalty = -10.0
                reward += invalid_penalty
                logger.warning(
                    f"environment: Agent {agent} tried invalid action: non-buildable at ({x},{y}). Penalty: {invalid_penalty}."
                )
            elif not self._can_afford(agent, building_type):
                invalid_penalty = -10.0
                reward += invalid_penalty
                logger.warning(
                    f"environment: Agent {agent} tried invalid action: cannot afford {building_type}. Penalty: {invalid_penalty}."
                )
            else:
                # Deduct resources
                building_cost = BUILDING_COSTS[building_type]
                player_resources = self.players[agent].resources
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

                logger.debug(
                    f"environment: Agent {agent} built {building_type} at ({x},{y})"
                )

                info_resources = {
                    "money": -building_cost["money"],
                    "reputation": -building_cost["reputation"],
                }

        # Update utilities based on buildings owned by current agent
        agent_index = self.agents.index(agent)
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                if self.buildings[gx][gy] is not None and self.builders[gx][gy] == agent_index:
                    b_type = self.buildings[gx][gy]["type"]
                    b_utility = BUILDING_UTILITIES[b_type]
                    self.players[agent].resources["money"] += b_utility["money"]
                    self.players[agent].resources["reputation"] += b_utility["reputation"]
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
        Compute the reward for an agent based on the change in integrated_score.
        Formula: reward = delta_integrated_score (normalized by alpha/beta is already in integrated_score)
        """
        current_score = self.players[agent].integrated_score
        previous_score = self.previous_integrated_score[agent]
        delta = current_score - previous_score
        self.previous_integrated_score[agent] = current_score

        reward = delta
        logger.debug(
            f"environment: Compute individual reward for {agent} - Delta: {delta}, "
            f"Previous: {previous_score}, Current: {current_score}, Reward: {reward}"
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
            "terrain_map": self.terrain_map.copy(),
            "prebuilt_map": self.prebuilt_map.copy(),
        }
        logger.debug(f"environment: observe Observation for {agent}: {observation}")
        return observation

    def render(self, mode="human"):
        display_grid = ""
        for x in range(self.grid_x):
            row = ""
            for y in range(self.grid_y):
                # Check for pre-built projects first
                if (x, y) in self.prebuilt_map:
                    project_type = self.prebuilt_map[(x, y)]
                    symbol = PREBUILT_PROJECTS[project_type]["symbol"]
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
                    if b["type"] == "House":
                        row += "[H]"
                    elif b["type"] == "Shop":
                        row += "[S]"
                    elif b["type"] == "GreenPark":
                        row += "[G]"
                    elif b["type"] == "CommunityHub":
                        row += "[C]"
                    elif b["type"] == "SolarGrid":
                        row += "[O]"  # Solar grid uses O to avoid conflict with Shop's S
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
