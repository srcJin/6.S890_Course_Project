# src/envs/simcity_scale_up/environment.py
# Urban Resilience-Focused SimCity Environment

from pettingzoo.utils import AECEnv, agent_selector
from gymnasium import spaces
import numpy as np
from .players import (
    BasePlayer,
    BalancedPlayer,
    InterestDrivenPlayer,
    AltruisticPlayer,
    EnvironmentalFocusedPlayer,
)
from .config import (
    TERRAIN_AND_PROJECTS,
    BUILDING_UTILITIES,
    INITIAL_GRID,
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
# Extract building types from config
BUILDING_TYPES = [
    name for name, data in TERRAIN_AND_PROJECTS.items() if data["type"] == "project"
]
NUM_BUILDING_TYPES = len(BUILDING_TYPES)


class SimCityScaleUpEnv(AECEnv):
    """
    Urban Resilience-Focused SimCity Environment (Koto Experiment)

    Grid Parameters (6-parameter system):
    - G: Greenery (urban green spaces, biodiversity, environmental quality)
    - V: Vitality (economic activity, social vibrancy, community life)
    - D: Density (population density, urban development intensity)
    - A: Adaptability (climate adaptation capacity, infrastructure flexibility)
    - S: Sustainability (environmental footprint, resource efficiency)
    - F: Flood_Resistance (disaster preparedness, protective infrastructure)

    Building Types:
    Basic Development:
    - House: Standard residential housing
    - Shop: Commercial retail space
    - Office: Commercial office building
    - Factory: Industrial manufacturing facility

    Resilience Projects:
    - Park: Green park for community recreation
    - Shelter: Community resilience center
    - Watergate: Renewable energy infrastructure
    - FloodBarrier: Climate protection infrastructure
    """

    metadata = {"render_modes": ["human"], "name": "SimCityScaleUpEnv"}

    def __init__(
        self, grid_x=8, grid_y=8, common_reward=False, reward_alpha=0.5, reward_beta=0.5
    ):
        super().__init__()
        self.TERRAIN_AND_PROJECTS = TERRAIN_AND_PROJECTS
        self.BUILDING_TYPES = BUILDING_TYPES
        self.BUILDING_UTILITIES = BUILDING_UTILITIES
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

        # Observation space with 6 grid parameters (G, V, D, A, S, F)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=100,
                        shape=(
                            self.grid_x,
                            self.grid_y,
                            6,
                        ),  # 6 parameters: G, V, D, A, S, F
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
                self.players[agent] = EnvironmentalFocusedPlayer(
                    agent
                )  # Environmental focus

        # Initialize previous integrated scores for reward calculation
        self.previous_integrated_score = {agent: 0 for agent in self.agents}

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Initialize grid with baseline values for 6 parameters
        # These will be modified by terrain and project effects from INITIAL_GRID
        self.grid = np.empty((self.grid_x, self.grid_y, 6), dtype=np.int32)
        self.grid[:, :, 0] = 100  # G - Greenery baseline
        self.grid[:, :, 1] = 100  # V - Vitality baseline
        self.grid[:, :, 2] = 100  # D - Density baseline
        self.grid[:, :, 3] = 100  # A - Adaptability baseline
        self.grid[:, :, 4] = 100  # S - Sustainability baseline
        self.grid[:, :, 5] = 100  # F - Flood_Resistance baseline

        # Initialize grid layout from config (predefined terrain and projects)
        self.grid_layout = np.array(INITIAL_GRID)

        self.buildings = np.full((self.grid_x, self.grid_y), None)
        self.builders = np.full((self.grid_x, self.grid_y), -1, dtype=np.int32)
        self.building_types = np.full(
            (self.grid_x, self.grid_y), -1, dtype=np.int32
        )  # -1 for no building

        # Calculate actual grid parameters based on INITIAL_GRID terrain and projects
        # This applies effects from Water, Road, House, Factory, etc. to each cell
        self._apply_initial_terrain_effects()

        self.individual_rewards_list = {agent: 0 for agent in self.agents}
        self.common_reward_value = 0
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Starting resources from config (already set in player init)
        for player in self.players.values():
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

    def _apply_initial_terrain_effects(self):
        """Apply initial terrain and project effects from the grid layout."""
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                cell_id = self.grid_layout[x][y]
                # Find the terrain/project type for this cell
                for name, data in self.TERRAIN_AND_PROJECTS.items():
                    if data["id"] == cell_id:
                        effect = data["effect"]
                        # Apply effects to this cell (G, V, D, A, S, F)
                        self.grid[x][y][0] += effect["G"]
                        self.grid[x][y][1] += effect["V"]
                        self.grid[x][y][2] += effect["D"]
                        self.grid[x][y][3] += effect["A"]
                        self.grid[x][y][4] += effect["S"]
                        self.grid[x][y][5] += effect["F"]

                        # Apply neighbor effects if they exist
                        if "neighbors" in data:
                            neighbors = data["neighbors"]
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                                    self.grid[nx][ny][0] += neighbors["G"]
                                    self.grid[nx][ny][1] += neighbors["V"]
                                    self.grid[nx][ny][2] += neighbors["D"]
                                    self.grid[nx][ny][3] += neighbors["A"]
                                    self.grid[nx][ny][4] += neighbors["S"]
                                    self.grid[nx][ny][5] += neighbors["F"]
                        break

    def _is_buildable(self, x, y):
        """Check if a cell is buildable (Empty terrain type)."""
        cell_id = self.grid_layout[x][y]
        # Find if this cell is buildable (Empty terrain)
        for name, data in self.TERRAIN_AND_PROJECTS.items():
            if data["id"] == cell_id:
                return data["is_buildable"] and self.buildings[x][y] is None
        return False

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
                building_data = self.TERRAIN_AND_PROJECTS[building_type]
                building_cost = building_data["cost"]
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

                    # Update grid with new 6-parameter system (G, V, D, A, S, F)
                    building_effect = building_data["effect"]
                    self.grid[x][y][0] += building_effect["G"]  # Greenery
                    self.grid[x][y][1] += building_effect["V"]  # Vitality
                    self.grid[x][y][2] += building_effect["D"]  # Density
                    self.grid[x][y][3] += building_effect["A"]  # Adaptability
                    self.grid[x][y][4] += building_effect["S"]  # Sustainability
                    self.grid[x][y][5] += building_effect["F"]  # Flood_Resistance

                    # Update neighbors score
                    if "neighbors" in building_data:
                        neighbors = building_data["neighbors"]
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                                self.grid[nx][ny][0] += neighbors["G"]
                                self.grid[nx][ny][1] += neighbors["V"]
                                self.grid[nx][ny][2] += neighbors["D"]
                                self.grid[nx][ny][3] += neighbors["A"]
                                self.grid[nx][ny][4] += neighbors["S"]
                                self.grid[nx][ny][5] += neighbors["F"]

                    building_utility = building_data["utility"]
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
                    b_utility = self.TERRAIN_AND_PROJECTS[b_type]["utility"]
                    self.players[agent].resources["money"] += b_utility["money"]
                    self.players[agent].resources["reputation"] += b_utility[
                        "reputation"
                    ]
                    self.players[agent].self_score += (
                        b_utility["money"] + b_utility["reputation"]
                    )

        # Calculate environment scores using new 6-parameter system
        env_scores = self.calculate_environment_score()
        self.env_score = env_scores["env_score"]
        gvd_score = env_scores["gvd_score"]
        asf_score = env_scores["asf_score"]

        logger.debug(f"environment: env_score after step: {self.env_score}")

        # Update player state using new utility function with GVD and ASF scores
        self.players[agent].update_state(reward, info_resources, gvd_score, asf_score)

        logger.debug(
            f"environment: Player {agent} - Self score: {self.players[agent].self_score}, "
            f"Integrated score: {self.players[agent].integrated_score} "
            f"(α={self.players[agent].alpha}, β={self.players[agent].beta}, γ={self.players[agent].gamma})"
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
        G_avg = np.mean(self.grid[:, :, 0])  # Greenery
        V_avg = np.mean(self.grid[:, :, 1])  # Vitality
        D_avg = np.mean(self.grid[:, :, 2])  # Density
        A_avg = np.mean(self.grid[:, :, 3])  # Adaptability
        S_avg = np.mean(self.grid[:, :, 4])  # Sustainability
        F_avg = np.mean(self.grid[:, :, 5])  # Flood_Resistance

        # Calculate GVD score (traditional urban metrics)
        gvd_score = (G_avg + V_avg + D_avg) / 3
        # Calculate ASF score (resilience metrics)
        asf_score = (A_avg + S_avg + F_avg) / 3
        # Overall environment score
        env_score = (G_avg + V_avg + D_avg + A_avg + S_avg + F_avg) / 6

        logger.debug(
            f"environment: calculate_environment_score G={G_avg:.1f}, V={V_avg:.1f}, D={D_avg:.1f}, A={A_avg:.1f}, S={S_avg:.1f}, F={F_avg:.1f}"
        )
        logger.debug(
            f"environment: GVD_score={gvd_score:.1f}, ASF_score={asf_score:.1f}, env_score={env_score:.1f}"
        )

        return {
            "G_avg": G_avg,
            "V_avg": V_avg,
            "D_avg": D_avg,
            "A_avg": A_avg,
            "S_avg": S_avg,
            "F_avg": F_avg,
            "gvd_score": gvd_score,
            "asf_score": asf_score,
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
            "grid_layout": self.grid_layout.copy(),
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
                        row += (
                            "[O]"  # Solar grid uses O to avoid conflict with Shop's S
                        )
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
