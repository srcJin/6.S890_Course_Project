# src/envs/simcity_wrapper.py

import numpy as np
import torch as th
from envs.multiagentenv import MultiAgentEnv
from envs.simcity import SimCityEnv, BalancedPlayer, InterestDrivenPlayer, AltruisticPlayer
from utils.logging import get_logger
from envs.simcity.environment import NO_OP

logger = get_logger(log_file_path="simulation.log")

class SimCityWrapper(MultiAgentEnv):
    def __init__(self, grid_x=4, grid_y=4, **kwargs):
        logger.debug("simcity_wrapper: Initializing SimCityWrapper")

        self.env = SimCityEnv(grid_x=grid_x, grid_y=grid_y, common_reward=kwargs.get("common_reward", False))
        self.n_agents = len(self.env.agents)
        # @todo is it true? 
        self.episode_limit = kwargs.get("time_limit", grid_x * grid_y)
        self.current_step = 0

        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = (
                single_obs["grid"].size +
                len(single_obs["resources"])+
                single_obs["builders"].size +
                single_obs["building_types"].size
            )
        else:
            self.obs_size = 0
            logger.warning("simcity_wrapper: No agents present during initialization.")

        self.state_size = self.obs_size * self.n_agents
        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0
        logger.debug(f"simcity_wrapper: Obs size={self.obs_size}, State size={self.state_size}, Actions={self.n_actions}")

        obs, _ = self.reset()
        assert obs.shape == (1, self.n_agents, self.obs_size), (
            f"simcity_wrapper: Observation shape mismatch: {obs.shape} vs {(1, self.n_agents, self.obs_size)}"
        )

        if self.state_size > 0:
            state = self.get_state()
            assert state.shape == (1, self.state_size), (
                f"simcity_wrapper: State shape mismatch: {state.shape} vs {(1, self.state_size)}"
            )

    def get_obs(self):
        logger.debug("simcity_wrapper: Collecting observations for all agents.")
        observations = []

        for agent in self.env.agents:
            obs = self.env.observe(agent)
            flat_obs = np.concatenate([
                obs["grid"].flatten(),
                list(obs["resources"].values()),
                obs["builders"].flatten(), obs["building_types"].flatten()
            ])
            observations.append(flat_obs)
        
        logger.debug(f"Grid shape: {obs['grid'].shape}")
        logger.debug(f"Resources: {obs['resources']}")
        logger.debug(f"Builders shape: {obs['builders'].shape}")
        logger.debug(f"Building types shape: {obs['building_types'].shape}")
        logger.debug(f"Flattened observation: {flat_obs}")

        obs_array = np.array(observations, dtype=np.float32)[np.newaxis]
        logger.debug(f"simcity_wrapper: Aggregated observations shape={obs_array.shape}")
        return obs_array

    def get_obs_agent(self, agent_id):
        logger.debug(f"simcity_wrapper: Fetching observation for agent {agent_id}.")
        agent = self.env.agents[agent_id]
        obs = self.env.observe(agent)
        agent_obs = np.concatenate([
            obs["grid"].flatten(),
            list(obs["resources"].values()),
            obs["builders"].flatten(), obs["building_types"].flatten()
        ]).astype(np.float32)
        return agent_obs

    def get_state(self):
        logger.debug("simcity_wrapper: Fetching global state.")
        obs = self.get_obs()
        state = obs.reshape(1, -1)
        logger.debug(f"simcity_wrapper: Global state shape={state.shape}")
        return state

    # def get_avail_actions(self):
    #     logger.debug("simcity_wrapper: Fetching available actions for all agents.")
    #     return np.ones((1, self.n_agents, self.n_actions), dtype=np.float32)

    # def get_avail_agent_actions(self, agent_id):
    #     logger.debug(f"simcity_wrapper: Fetching available actions for agent {agent_id}.")
    #     return [1] * self.n_actions

    def get_avail_agent_actions(self, agent_id):
        logger.debug(f"simcity_wrapper: Fetching available actions for agent {agent_id}.")
        agent = self.env.agents[agent_id]
        avail_actions = np.zeros(self.n_actions, dtype=np.float32)

        for action in range(self.n_actions):
            building_type, x, y = self.env.decode_action(action)

            # No-op is always available
            if action == NO_OP:
                avail_actions[action] = 1.0
                continue

            # Check if coordinates are within bounds
            if not (0 <= x < self.env.grid_x and 0 <= y < self.env.grid_y):
                continue  # Invalid coordinates

            # Check if the cell is already occupied
            if self.env.buildings[x][y] is not None:
                continue  # Cell occupied

            # Check if the agent has enough resources
            building_cost = self.env.BUILDING_COSTS[building_type]
            player_resources = self.env.players[agent].resources
            if (player_resources["money"] >= building_cost["money"] and
                player_resources["reputation"] >= building_cost["reputation"]):
                avail_actions[action] = 1.0  # Action is available

        logger.debug(f"simcity_wrapper: Available actions for agent {agent_id}: {avail_actions}")
        return avail_actions.tolist()

    def get_avail_actions(self):
        logger.debug("simcity_wrapper: Fetching available actions for all agents.")
        avail_actions = np.ones((1, self.n_agents, self.n_actions), dtype=np.float32)
        for agent_id, agent in enumerate(self.env.agents):
            agent_avail = self.get_avail_agent_actions(agent_id)
            avail_actions[0, agent_id] = agent_avail
        return avail_actions

    def step(self, actions):
        logger.debug(f"simcity_wrapper: Step called with actions: {actions}")
        actions_list = actions.squeeze(0).cpu().numpy().tolist()
        logger.debug(f"simcity_wrapper: Actions list: {actions_list}")



        for idx, agent in enumerate(self.env.agents):
            if self.env.terminations[agent] or self.env.truncations[agent]:
                self.env.step(None)
                logger.debug(f"simcity_wrapper: Agent {agent} is terminated or truncated; stepping with None.")
                continue

            action = actions_list[idx]
            if action < 0 or action >= self.n_actions:
                logger.warning(f"simcity_wrapper: Invalid action {action} for agent {agent}. Forcing No-op.")
                action = 0  # default to no-op

            self.env.step(action)
            logger.debug(f"simcity_wrapper: Agent {agent} took action {action}.")

            if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                logger.debug("simcity_wrapper: Game ended during multi-agent step loop.")
                break

        obs = self.get_obs()


        if self.env.common_reward:
            logger.debug(f"simcity_wrapper, common reward mode, rewards = {self.env.common_reward_value}")
            rewards = self.env.common_reward_value
        else:
            logger.debug(f"simcity_wrapper, individual reward mode, rewards = {self.env.individual_reward_list}")
            rewards = self.env.individual_reward_list

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
            logger.debug("simcity_wrapper: Episode limit reached, terminating.")

        # info should record the environment score, each agent's reward, and common reward
        info = {"env_score": self.env.env_score, "agent_1_individual_reward":self.env.individual_rewards_list["P1"], "agent_2_individual_reward":self.env.individual_rewards_list["P2"], "agent_1_individual_reward":self.env.individual_rewards_list["P3"], "common_reward_value": self.env.common_reward_value}

        logger.debug(f"simcity_wrapper: Step result: obs shape={obs.shape}, rewards={rewards}, terminated={terminated}, truncated={truncated}, info={info}")

        return obs, rewards, terminated, truncated, info

    def reset(self, seed=None, options=None):
        logger.debug("simcity_wrapper: Reset called.")
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_step = 0

        self.n_agents = len(self.env.agents)
        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = single_obs["grid"].size + 2 + single_obs["builders"].size + single_obs["building_types"].size
        else:
            self.obs_size = 0

        self.state_size = self.obs_size * self.n_agents
        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0
        logger.debug(f"simcity_wrapper: Reset results - n_agents={self.n_agents}, obs_size={self.obs_size}, state_size={self.state_size}, n_actions={self.n_actions}")
        return self.get_obs(), info

    def get_env_info(self):
        logger.debug("simcity_wrapper: Fetching environment info.")
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        logger.debug(f"simcity_wrapper: Environment info: {env_info}")
        return env_info

    def get_stats(self):
        logger.debug("simcity_wrapper: Fetching stats (empty implementation).")
        return {}

    def get_obs_size(self):
        logger.debug("simcity_wrapper: Fetching observation size.")
        return self.obs_size

    def get_state_size(self):
        logger.debug("simcity_wrapper: Fetching state size.")
        return self.state_size

    def get_total_actions(self):
        logger.debug("simcity_wrapper: Fetching total number of actions.")
        return self.n_actions

    def render(self):
        logger.debug("simcity_wrapper: Rendering environment.")
        return self.env.render()

    def close(self):
        logger.debug("simcity_wrapper: Closing environment.")
        self.env.close()

    def seed(self, seed=None):
        logger.debug(f"simcity_wrapper: Seeding environment with seed: {seed}")
        return self.reset(seed=seed)[0]
