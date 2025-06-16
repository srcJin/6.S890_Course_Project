# src/envs/simcity_scale_up_wrapper.py

import numpy as np
import torch as th
from envs.multiagentenv import MultiAgentEnv
from envs.simcity_scale_up import (
    SimCityScaleUpEnv,
    BalancedPlayer,
    InterestDrivenPlayer,
    AltruisticPlayer,
)
from utils.logging import get_logger

logger = get_logger(log_file_path="simulation_scale_up.log")


class SimCityScaleUpWrapper(MultiAgentEnv):
    def __init__(self, grid_x=8, grid_y=8, **kwargs):
        logger.debug("simcity_scale_up_wrapper: Initializing SimCityScaleUpWrapper")

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
            self.obs_size = (
                single_obs["grid"].size
                + len(single_obs["resources"])
                + single_obs["builders"].size
                + single_obs["building_types"].size
            )
        else:
            self.obs_size = 0
            logger.warning(
                "simcity_scale_up_wrapper: No agents present during initialization."
            )

        self.state_size = self.obs_size * self.n_agents
        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0
        logger.debug(
            f"simcity_scale_up_wrapper: Obs size={self.obs_size}, State size={self.state_size}, Actions={self.n_actions}"
        )

        obs, _ = self.reset()
        assert obs.shape == (
            self.n_agents,
            self.obs_size,
        ), f"Expected obs shape ({self.n_agents}, {self.obs_size}), got {obs.shape}"

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

        rewards = []
        dones = []
        infos = []

        # Execute actions for each agent
        for i, agent in enumerate(self.env.agents):
            if not self.env.terminations[agent] and not self.env.truncations[agent]:
                if self.env.agent_selection == agent:
                    action = actions[i] if i < len(actions) else 0
                    self.env.step(action)

                    # Get reward for this agent
                    if self.env.common_reward:
                        reward = self.env.common_reward_value
                    else:
                        reward = self.env.individual_rewards_list.get(agent, 0)

                    rewards.append(reward)
                    dones.append(
                        self.env.terminations[agent] or self.env.truncations[agent]
                    )
                    infos.append(self.env.infos.get(agent, {}))
                else:
                    # Agent is not active this step
                    rewards.append(0)
                    dones.append(
                        self.env.terminations[agent] or self.env.truncations[agent]
                    )
                    infos.append({})
            else:
                # Agent is already done
                rewards.append(0)
                dones.append(True)
                infos.append({})

        # Get updated observations
        obs = []
        for agent in self.env.agents:
            agent_obs = self.env.observe(agent)
            flat_obs = self._flatten_observation(agent_obs)
            obs.append(flat_obs)

        obs = np.array(obs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        self.current_step += 1

        # Check if episode is done
        episode_done = all(dones) or self.current_step >= self.episode_limit

        logger.debug(
            f"simcity_scale_up_wrapper: Step complete, rewards: {rewards}, done: {episode_done}"
        )
        return rewards, obs, dones, infos, episode_done

    def get_obs(self):
        """Get current observations for all agents"""
        obs = []
        for agent in self.env.agents:
            agent_obs = self.env.observe(agent)
            flat_obs = self._flatten_observation(agent_obs)
            obs.append(flat_obs)
        return np.array(obs, dtype=np.float32)

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
        obs = self.get_obs()
        return obs.flatten()

    def get_state_size(self):
        """Return the size of the global state"""
        return self.state_size

    def get_avail_actions(self):
        """Get available actions for all agents"""
        avail_actions = []
        for agent in self.env.agents:
            if self.env.terminations[agent] or self.env.truncations[agent]:
                # Agent is done, only no-op available
                agent_avail = np.zeros(self.n_actions)
                agent_avail[0] = 1  # No-op action
            else:
                # All actions are available for active agents
                agent_avail = np.ones(self.n_actions)
            avail_actions.append(agent_avail)
        return np.array(avail_actions, dtype=np.float32)

    def get_avail_agent_actions(self, agent_id):
        """Get available actions for specific agent"""
        agent = self.env.agents[agent_id]
        if self.env.terminations[agent] or self.env.truncations[agent]:
            avail_actions = np.zeros(self.n_actions)
            avail_actions[0] = 1  # No-op action
        else:
            avail_actions = np.ones(self.n_actions)
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
        
        # Flatten grid (S, W, R, C parameters for 8x8 grid)
        flat_parts.append(obs_dict["grid"].flatten())
        
        # Add resources (money, reputation) as individual elements
        resources = obs_dict["resources"]
        flat_parts.append(np.array([resources["money"]], dtype=np.float32))
        flat_parts.append(np.array([resources["reputation"]], dtype=np.float32))
        
        # Flatten builders and building_types
        flat_parts.append(obs_dict["builders"].flatten())
        flat_parts.append(obs_dict["building_types"].flatten())
        
        return np.concatenate(flat_parts).astype(np.float32)

    def get_episode_records(self):
        """Get episode records for visualization"""
        # This would need to be implemented to store episode data
        # For now, return empty structure
        return {"episode_records": []}
