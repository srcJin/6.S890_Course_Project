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
    INITIAL_GRID,
    INCOME_LIFECYCLE,
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
        self, grid_x=12, grid_y=12, common_reward=False, reward_alpha=0.5, reward_beta=0.5
    ):
        super().__init__()
        self.TERRAIN_AND_PROJECTS = TERRAIN_AND_PROJECTS
        self.BUILDING_TYPES = BUILDING_TYPES
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
                        low=-50,
                        high=250,
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
            import random
            random.seed(seed)

        # Initialize grid with baseline values for 6 parameters
        # These will be modified by terrain and project effects from INITIAL_GRID
        self.grid = np.empty((self.grid_x, self.grid_y, 6), dtype=np.int32)
        self.grid[:, :, 0] = 50  # G - Greenery baseline
        self.grid[:, :, 1] = 50  # V - Vitality baseline
        self.grid[:, :, 2] = 50  # D - Density baseline
        self.grid[:, :, 3] = 50  # A - Adaptability baseline
        self.grid[:, :, 4] = 50  # S - Sustainability baseline
        self.grid[:, :, 5] = 50  # F - Flood_Resistance baseline

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

        # Initialize pre-built buildings with lifecycle data
        self._initialize_prebuilt_buildings()

        self.individual_rewards_list = {agent: 0 for agent in self.agents}
        self.common_reward_value = 0
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Starting resources from config (reset to initial values)
        for player in self.players.values():
            player.self_score = 0
            player.integrated_score = 0
            player.final_score = 0
            player.environmental_impact_score = 0
            # Reset resources to initial values for fair episode restart
            player.resources = {
                "money": 100,
                "reputation": 70,
            }

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
        """Check if a cell is buildable (Empty terrain or replaceable building)."""
        cell_id = self.grid_layout[x][y]

        # Check if terrain is buildable
        for name, data in self.TERRAIN_AND_PROJECTS.items():
            if data["id"] == cell_id:
                if data["is_buildable"]:
                    # Empty cell or replaceable building
                    if self.buildings[x][y] is None:
                        return True
                    elif self.buildings[x][y].get("is_replaceable", False):
                        return True
                return False
        return False

    def _calculate_building_income(self, building, building_type):
        """
        Calculate building income multiplier based on age and lifecycle.

        Income pattern:
        - Turn 1 (age 0): No income (construction turn)
        - Turn 2 (age 1): Start with 100% income
        - Each subsequent turn: Income decays by decay_rate
        - After duration: No income
        """
        age = building["age"]
        duration = INCOME_LIFECYCLE["duration"]
        start_delay = INCOME_LIFECYCLE["start_delay"]

        # No income during construction and start delay
        if age <= start_delay:
            return 0.0

        # No income after building expires
        if age > duration:
            return 0.0

        # Income age counts producing turns: 1 means the first producing turn
        income_age = age - start_delay

        # Linear decline: producing income starts at 1.0 and decreases linearly
        # so that at income_age == duration the multiplier is ~1/duration and
        # after duration it is 0 (handled above).
        income_multiplier = 1.0 - (income_age - 1) / float(duration)
        return max(0.0, income_multiplier)

    def _initialize_prebuilt_buildings(self):
        """Initialize pre-built buildings from INITIAL_GRID with lifecycle data."""
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                cell_id = self.grid_layout[x][y]

                # Check if this cell contains a pre-built building
                for name, data in self.TERRAIN_AND_PROJECTS.items():
                    if data["id"] == cell_id and data["type"] == "project":
                        # Initialize pre-built building with lifecycle data
                        # Start with random age to simulate existing city
                        import random

                        random_age = 10  # Pre-built buildings start young to avoid immediate replacement

                        self.buildings[x][y] = {
                            "type": name,
                            "turn_built": -random_age,  # Negative to indicate pre-built
                            "age": random_age,
                            # Replacement is only allowed after a project fully expires (no income)
                            "is_replaceable": random_age > INCOME_LIFECYCLE["duration"],
                        }

                        # Mark as built by "system" (no specific agent)
                        self.builders[x][y] = -1  # -1 indicates pre-built

                        if name in BUILDING_TYPES:
                            self.building_types[x][y] = BUILDING_TYPES.index(name)

                        logger.debug(
                            f"Initialized pre-built {name} at ({x},{y}) with age {random_age}"
                        )
                        break

    def _age_all_buildings(self):
        """Age all buildings by 1 turn and update their replaceability status."""
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                if self.buildings[x][y] is not None:
                    building = self.buildings[x][y]
                    building["age"] += 1
                    # Update replaceability status: allow replacement ONLY after expiry
                    duration = INCOME_LIFECYCLE["duration"]
                    if building["age"] > duration:
                        if not building.get("is_replaceable", False):
                            building["is_replaceable"] = True
                            logger.debug(
                                f"Building {building['type']} at ({x},{y}) age {building['age']} becomes replaceable (expired)"
                            )

    def _remove_building_effects(self, x, y, building_type):
        """Remove the grid effects of a building when it's replaced."""
        building_data = self.TERRAIN_AND_PROJECTS[building_type]
        effect = building_data["effect"]

        # Remove direct effects
        self.grid[x][y][0] -= effect["G"]  # Greenery
        self.grid[x][y][1] -= effect["V"]  # Vitality
        self.grid[x][y][2] -= effect["D"]  # Density
        self.grid[x][y][3] -= effect["A"]  # Adaptability
        self.grid[x][y][4] -= effect["S"]  # Sustainability
        self.grid[x][y][5] -= effect["F"]  # Flood_Resistance

        # Remove neighbor effects
        if "neighbors" in building_data:
            neighbors = building_data["neighbors"]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                    self.grid[nx][ny][0] -= neighbors["G"]
                    self.grid[nx][ny][1] -= neighbors["V"]
                    self.grid[nx][ny][2] -= neighbors["D"]
                    self.grid[nx][ny][3] -= neighbors["A"]
                    self.grid[nx][ny][4] -= neighbors["S"]
                    self.grid[nx][ny][5] -= neighbors["F"]

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
            elif self.buildings[x][y] is not None and not self.buildings[x][y].get(
                "is_replaceable", False
            ):
                build_on_occupied_penalty = -999999999999999999
                reward += build_on_occupied_penalty
                logger.debug(
                    f"environment: Agent {agent} tried to build on a non-replaceable building at ({x},{y}). Penalty: {build_on_occupied_penalty}."
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

                    # Remove old building effects if replacing
                    if self.buildings[x][y] is not None:
                        old_building = self.buildings[x][y]
                        old_type = old_building["type"]
                        logger.debug(
                            f"Replacing {old_type} with {building_type} at ({x},{y})"
                        )
                        self._remove_building_effects(x, y, old_type)

                    # Update buildings and builders
                    self.buildings[x][y] = {
                        "type": building_type,
                        "turn_built": self.num_moves,
                        "age": 0,  # Building age for income calculation
                        "is_replaceable": False,  # Can be replaced when income is low
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

        # Age all buildings once per full round (only when first agent acts)
        if self.agents.index(agent) == 0:
            self._age_all_buildings()

        # Update utilities based on buildings - only for buildings owned by current agent
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                if self.buildings[gx][gy] is not None and self.builders[gx][
                    gy
                ] == self.agents.index(agent):
                    building = self.buildings[gx][gy]
                    b_type = building["type"]

                    # Calculate income based on age and lifecycle
                    income = self._calculate_building_income(building, b_type)

                    # Apply income if building is producing
                    if income > 0:
                        b_utility = self.TERRAIN_AND_PROJECTS[b_type]["utility"]
                        # Apply decay multiplier
                        actual_money = b_utility["money"] * income
                        actual_reputation = b_utility["reputation"] * income

                        self.players[agent].resources["money"] += actual_money
                        self.players[agent].resources["reputation"] += actual_reputation
                        self.players[agent].self_score += (
                            actual_money + actual_reputation
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
        # Avoid printing full arrays which is very slow; log shapes and key stats instead
        try:
            logger.debug(
                "environment: observe | agent=%s | grid.shape=%s grid_layout.shape=%s builders.shape=%s building_types=%s resources_keys=%s",
                agent,
                getattr(observation["grid"], "shape", None),
                getattr(observation["grid_layout"], "shape", None),
                getattr(observation["builders"], "shape", None),
                (
                    list(observation["building_types"].keys())
                    if isinstance(observation["building_types"], dict)
                    else type(observation["building_types"]).__name__
                ),
                (
                    list(observation["resources"].keys())
                    if isinstance(observation["resources"], dict)
                    else type(observation["resources"]).__name__
                ),
            )
        except Exception:
            # Best-effort debug logging only
            pass
        return observation

    def render(self, mode="human"):
        display_grid = ""
        for x in range(self.grid_x):
            row = ""
            for y in range(self.grid_y):
                # If there is a player-built or prebuilt building, show its symbol
                if self.buildings[x][y] is not None:
                    b_type = self.buildings[x][y]["type"]
                    symbol = self.TERRAIN_AND_PROJECTS.get(b_type, {}).get(
                        "symbol", "?"
                    )
                    row += f"[{symbol}]"
                else:
                    # Otherwise show the base grid layout (terrain or prebuilt project symbol)
                    cell_id = int(self.grid_layout[x][y])
                    symbol = "?"
                    for name, data in self.TERRAIN_AND_PROJECTS.items():
                        if data.get("id") == cell_id:
                            symbol = str(data.get("symbol", "?"))[:1]
                            break
                    row += f"[{symbol}]"
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
