# src/envs/simcity_wrapper.py

import numpy as np
import torch as th
from envs.multiagentenv import MultiAgentEnv
from envs.simcity import SimCityEnv, BalancedPlayer
from utils.logging import get_logger

logger = get_logger(log_file_path="simulation.log")

class SimCityWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        logger.debug("simcity_wrapper: Initializing SimCityWrapper")
        self.env = SimCityEnv(common_reward=kwargs.get("common_reward", False))
        self.env.players = {
            "P1": BalancedPlayer("P1"),
            "P2": BalancedPlayer("P2"),
            "P3": BalancedPlayer("P3"),
        }
        self.n_agents = len(self.env.agents)
        self.episode_limit = kwargs.get("time_limit", self.env.grid_size * self.env.grid_size)
        self.current_step = 0

        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = (
                single_obs["grid"].size +
                2 +
                single_obs["builders"].size
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
                [obs["resources"]["money"], obs["resources"]["reputation"]],
                obs["builders"].flatten()
            ])
            observations.append(flat_obs)
        obs_array = np.array(observations, dtype=np.float32)[np.newaxis]
        logger.debug(f"simcity_wrapper: Aggregated observations shape={obs_array.shape}")
        return obs_array

    def get_obs_agent(self, agent_id):
        logger.debug(f"simcity_wrapper: Fetching observation for agent {agent_id}.")
        agent = self.env.agents[agent_id]
        obs = self.env.observe(agent)
        agent_obs = np.concatenate([
            obs["grid"].flatten(),
            [obs["resources"]["money"], obs["resources"]["reputation"]],
            obs["builders"].flatten()
        ]).astype(np.float32)
        return agent_obs

    def get_state(self):
        logger.debug("simcity_wrapper: Fetching global state.")
        obs = self.get_obs()
        state = obs.reshape(1, -1)
        logger.debug(f"simcity_wrapper: Global state shape={state.shape}")
        return state

    def get_avail_actions(self):
        logger.debug("simcity_wrapper: Fetching available actions for all agents.")
        return np.ones((1, self.n_agents, self.n_actions), dtype=np.float32)

    def get_avail_agent_actions(self, agent_id):
        logger.debug(f"simcity_wrapper: Fetching available actions for agent {agent_id}.")
        return [1] * self.n_actions

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
            total_reward = sum(self.env._cumulative_rewards.values())
            reward = total_reward
            logger.debug(f"simcity_wrapper: Total reward (common_reward): {total_reward}")
        else:
            reward = [self.env._cumulative_rewards[agent] for agent in self.env.agents]
            logger.debug(f"simcity_wrapper: Individual rewards: {reward}")

        self.env._cumulative_rewards = {agent: 0 for agent in self.env.agents}

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

        info = {"env_score": self.env.env_score}

        logger.debug(f"simcity_wrapper: Step result: obs shape={obs.shape}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        logger.debug("simcity_wrapper: Reset called.")
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_step = 0

        self.n_agents = len(self.env.agents)
        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = single_obs["grid"].size + 2 + single_obs["builders"].size
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
