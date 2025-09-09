# src/envs/simcity_scale_up_koto_wrapper.py

import numpy as np
import torch as th
from envs.multiagentenv import MultiAgentEnv
from envs.simcity_scale_up_koto import (
    SimCityScaleUpEnv,
    BalancedPlayer,
    InterestDrivenPlayer,
    AltruisticPlayer,
    EnvironmentalFocusedPlayer,
)
from utils.logging import get_logger

logger = get_logger(log_file_path="simulation_scale_up_koto.log")


class SimCityScaleUpKotoWrapper(MultiAgentEnv):
    def __init__(self, grid_x=12, grid_y=12, **kwargs):
        logger.debug(
            "simcity_scale_up_koto_wrapper: Initializing SimCityScaleUpKotoWrapper"
        )

        self.env = SimCityScaleUpEnv(
            grid_x=grid_x,
            grid_y=grid_y,
            common_reward=kwargs.get("common_reward", False),
        )
        self.n_agents = len(self.env.agents)
        # Episode limit based on buildable spaces (41 for 8x8 grid)
        buildable_spaces = sum(
            1
            for x in range(grid_x)
            for y in range(grid_y)
            if self.env._is_buildable(x, y)
        )
        self.episode_limit = kwargs.get("time_limit", buildable_spaces)
        self.current_step = 0

        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            # Calculate actual observation size
            flat_obs = self._flatten_observation(single_obs)
            self.obs_size = len(flat_obs)
        else:
            self.obs_size = 0
            logger.warning(
                "simcity_scale_up_wrapper: No agents present during initialization."
            )

        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0

        # Calculate state size after obs_size is determined
        self.state_size = self.obs_size * self.n_agents

        logger.debug(
            f"simcity_scale_up_wrapper: Obs size={self.obs_size}, State size={self.state_size}, Actions={self.n_actions}"
        )

        # Validate by actually checking the observation size
        obs, _ = self.reset()
        if obs.shape != (self.n_agents, self.obs_size):
            logger.warning(
                f"Observation shape mismatch: expected ({self.n_agents}, {self.obs_size}), got {obs.shape}"
            )
            # Update obs_size to match actual size
            self.obs_size = obs.shape[1]
            self.state_size = self.obs_size * self.n_agents
            logger.info(
                f"Updated obs_size to {self.obs_size}, state_size to {self.state_size}"
            )

        logger.debug("simcity_scale_up_wrapper: Initialization completed successfully")

    def reset(self):
        logger.debug("simcity_scale_up_wrapper: Resetting environment")
        self.current_step = 0

        # Reset the PettingZoo environment
        obs_dict, info = self.env.reset()

        # Convert observations to format expected by MARL framework
        obs = []
        for agent in self.env.agents:
            agent_obs = self.env.observe(agent)
            flat_obs = self._flatten_observation(agent_obs)
            obs.append(flat_obs)

        obs = np.array(obs, dtype=np.float32)
        logger.debug(
            f"simcity_scale_up_wrapper: Reset complete, obs shape: {obs.shape}"
        )
        return obs, info

    def step(self, actions):
        """Execute actions for all agents"""
        logger.debug(
            f"simcity_scale_up_wrapper: Step {self.current_step}, actions: {actions}"
        )

        # Convert actions to list if needed
        if hasattr(actions, "tolist"):
            actions_list = actions.tolist()
        else:
            actions_list = actions if isinstance(actions, list) else [actions]

        logger.debug(f"simcity_scale_up_wrapper: Actions list: {actions_list}")

        # Execute actions for each agent
        for idx, agent in enumerate(self.env.agents):
            logger.debug(f"simcity_scale_up_wrapper step: Processing agent {agent}.")
            if self.env.terminations[agent] or self.env.truncations[agent]:
                self.env.step(None)
                logger.debug(
                    f"simcity_scale_up_wrapper: Agent {agent} is terminated or truncated; stepping with None."
                )
                continue

            action = actions_list[idx] if idx < len(actions_list) else 0
            if action < 0 or action >= self.n_actions:
                logger.warning(
                    f"simcity_scale_up_wrapper: Invalid action {action} for agent {agent}. Forcing No-op."
                )
                action = 0  # default to no-op

            self.env.step(action)
            logger.debug(
                f"simcity_scale_up_wrapper: Agent {agent} took action {action}."
            )

            if all(self.env.terminations.values()) or all(
                self.env.truncations.values()
            ):
                logger.debug(
                    "simcity_scale_up_wrapper: Game ended during multi-agent step loop."
                )
                break

        obs = self.get_obs()

        # Handle rewards like the original wrapper
        if self.env.common_reward:
            # Return a single scalar reward
            rewards = float(self.env.common_reward_value)
        else:
            # Return a vector of length n_agents
            rewards = np.array(
                [self.env.individual_rewards_list[agent] for agent in self.env.agents],
                dtype=np.float32,
            )

        # Determine termination flags
        done = all(self.env.terminations.values()) or all(self.env.truncations.values())
        terminated_flags = [self.env.terminations[agent] for agent in self.env.agents]
        truncated_flags = [self.env.truncations[agent] for agent in self.env.agents]

        terminated = any(terminated_flags)
        truncated = any(truncated_flags)

        self.current_step += 1
        if self.current_step >= self.episode_limit:
            terminated = True
            truncated = True
            logger.debug(
                "simcity_scale_up_wrapper: Episode limit reached, terminating."
            )

        # Info should record the environment score, each agent's reward, and common reward
        info = {
            "env_score": self.env.env_score,
            "common_reward_value": self.env.common_reward_value,
        }
        # Add individual rewards as separate numeric keys
        for agent, reward in self.env.individual_rewards_list.items():
            info[f"{agent}_reward"] = reward

        # Add resource info as separate numeric keys instead of nested dict
        for agent in self.env.agents:
            player_resources = self.env.players[agent].resources
            info[f"{agent}_money"] = player_resources["money"]
            info[f"{agent}_reputation"] = player_resources["reputation"]

        logger.debug(
            f"simcity_scale_up_wrapper: Step result: obs shape={obs.shape}, rewards={rewards}, terminated={terminated}, truncated={truncated}, info={info}"
        )

        return obs, rewards, terminated, truncated, info

    def get_obs(self):
        """Get current observations for all agents"""
        logger.debug(
            "simcity_scale_up_wrapper: Collecting observations for all agents."
        )
        observations = []

        for agent in self.env.agents:
            agent_obs = self.env.observe(agent)
            flat_obs = self._flatten_observation(agent_obs)
            observations.append(flat_obs)

        logger.debug(
            f"simcity_scale_up_wrapper: Collected {len(observations)} observations"
        )

        obs_array = np.array(observations, dtype=np.float32)[np.newaxis]
        logger.debug(
            f"simcity_scale_up_wrapper: Aggregated observations shape={obs_array.shape}"
        )
        return obs_array

    def get_obs_agent(self, agent_id):
        """Get observation for specific agent"""
        agent = self.env.agents[agent_id]
        agent_obs = self.env.observe(agent)
        return self._flatten_observation(agent_obs)

    def get_obs_size(self):
        """Return the size of observations"""
        return self.obs_size

    def get_state(self):
        """Get global state (concatenated observations)"""
        logger.debug("simcity_scale_up_wrapper: Fetching global state.")
        obs = self.get_obs()
        state = obs.reshape(1, -1)
        logger.debug(f"simcity_scale_up_wrapper: Global state shape={state.shape}")
        return state

    def get_state_size(self):
        """Return the size of the global state"""
        return self.state_size

    def get_avail_actions(self):
        """Get available actions for all agents"""
        logger.debug(
            "simcity_scale_up_wrapper: Fetching available actions for all agents."
        )
        avail_actions = np.ones((1, self.n_agents, self.n_actions), dtype=np.float32)
        for agent_id, agent in enumerate(self.env.agents):
            agent_avail = self.get_avail_agent_actions(agent_id)
            avail_actions[0, agent_id] = agent_avail
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Get available actions for specific agent"""
        logger.debug(
            f"simcity_scale_up_wrapper: Fetching available actions for agent {agent_id}."
        )
        agent = self.env.agents[agent_id]
        if self.env.terminations[agent] or self.env.truncations[agent]:
            avail_actions = np.zeros(self.n_actions, dtype=np.float32)
            avail_actions[0] = 1.0  # No-op action
        else:
            # Start with all actions unavailable, then enable NO-OP and valid build targets
            avail_actions = np.zeros(self.n_actions, dtype=np.float32)
            # Always allow NO-OP
            avail_actions[0] = 1.0

            # Resource-aware: only allow building types the agent can afford now
            player_res = self.env.players[agent].resources
            affordable_types = []
            for t, bname in enumerate(self.env.BUILDING_TYPES):
                cost = self.env.TERRAIN_AND_PROJECTS[bname]["cost"]
                can_afford = player_res.get("money", 0) >= cost.get(
                    "money", 0
                ) and player_res.get("reputation", 0) >= cost.get("reputation", 0)
                affordable_types.append(can_afford)

            # The discrete action layout is: 0 = NO-OP, then for each building type t and each cell c
            num_cells = self.env.grid_x * self.env.grid_y
            # Iterate all cells and mark buildable ones for all affordable project types
            for x in range(self.env.grid_x):
                for y in range(self.env.grid_y):
                    if self.env._is_buildable(x, y):
                        cell_index = x * self.env.grid_y + y
                        for t in range(len(self.env.BUILDING_TYPES)):
                            if not affordable_types[t]:
                                continue
                            aidx = 1 + t * num_cells + cell_index
                            if 0 <= aidx < self.n_actions:
                                avail_actions[aidx] = 1.0

        # Avoid logging full availability arrays; log counts only
        try:
            logger.debug(
                "simcity_scale_up_wrapper: avail_actions | agent=%s | enabled=%d/%d",
                agent_id,
                int(np.sum(avail_actions)),
                self.n_actions,
            )
        except Exception:
            pass
        return avail_actions

    def get_total_actions(self):
        """Return total number of actions"""
        return self.n_actions

    def get_env_info(self):
        """Return environment information"""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def render(self):
        """Render the environment"""
        return self.env.render()

    def close(self):
        """Close the environment"""
        self.env.close()

    def seed(self, seed):
        """Set random seed"""
        np.random.seed(seed)

    def _flatten_observation(self, obs_dict):
        """Convert observation dictionary to flat array"""
        flat_parts = []

        # Flatten grid (G, V, D, A, S, F parameters for 12x12 grid) - now 6 parameters
        flat_parts.append(obs_dict["grid"].flatten())

        # Add resources (money, reputation) as individual elements
        resources = obs_dict["resources"]
        flat_parts.append(np.array([resources["money"]], dtype=np.float32))
        flat_parts.append(np.array([resources["reputation"]], dtype=np.float32))

        # Flatten builders and building_types
        flat_parts.append(obs_dict["builders"].flatten())
        flat_parts.append(obs_dict["building_types"].flatten())

        # Add terrain layout information (simplified - just the grid layout IDs)
        # The environment now uses TERRAIN_AND_PROJECTS with unified structure
        terrain_matrix = obs_dict.get("grid_layout", np.zeros((12, 12), dtype=np.int32))
        flat_parts.append(terrain_matrix.flatten())

        return np.concatenate(flat_parts).astype(np.float32)

    def get_episode_records(self):
        """Get episode records for visualization"""
        # This would need to be implemented to store episode data
        # For now, return empty structure
        return {"episode_records": []}
