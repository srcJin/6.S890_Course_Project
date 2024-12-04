# simcity_wrapper.py
from envs.multiagentenv import MultiAgentEnv
from gymnasium.spaces import flatdim
import numpy as np
from envs.simcity import SimCityEnv, BalancedPlayer
from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")

class SimCityWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        metadata = {"render_modes": ["human"]}

        logger.debug("simcity_wrapper Initializing SimCityWrapper")

        self.env = SimCityEnv(common_reward=kwargs.get("common_reward", False))
        
        # Initialize players
        self.env.players = {
            "P1": BalancedPlayer("P1"),
            "P2": BalancedPlayer("P2"),
            "P3": BalancedPlayer("P3"),
        }
        logger.debug("simcity_wrapper Players initialized: P1, P2, P3")
        
        self.n_agents = len(self.env.agents)
        logger.debug(f"simcity_wrapper Number of agents: {self.n_agents}")
        
        # Set episode limit from env_args if provided, otherwise use default
        self.episode_limit = kwargs.get("time_limit", self.env.grid_size * self.env.grid_size)
        logger.debug(f"simcity_wrapper Episode limit set to: {self.episode_limit}")
        
        # Calculate observation size
        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = (
                single_obs["grid"].size +  # grid observations
                2 +  # money and reputation
                single_obs["builders"].size  # builders information
            )
            logger.debug(f"simcity_wrapper Observation size per agent: {self.obs_size}")
        else:
            self.obs_size = 0  # Handle cases with zero agents
            logger.warning("simcity_wrapper No agents present during initialization.")
        
        # Calculate state size (full observation for all agents)
        self.state_size = self.obs_size * self.n_agents
        logger.debug(f"simcity_wrapper State size: {self.state_size}")
        
        # Get number of actions
        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0
        logger.debug(f"simcity_wrapper Number of actions per agent: {self.n_actions}")
        
        # Initialize the environment
        logger.debug("simcity_wrapper Resetting environment during initialization.")
        obs, _ = self.reset()
        
        # Ensure shapes are correct
        try:
            assert obs.shape == (1, self.n_agents, self.obs_size), (
                f"simcity_wrapper Observation shape mismatch: {obs.shape} vs expected {(1, self.n_agents, self.obs_size)}"
            )
            logger.debug("simcity_wrapper Initial observation shape is correct.")
        except AssertionError as e:
            logger.error(e)
            raise
    
        if self.state_size > 0:
            state = self.get_state()
            try:
                assert state.shape == (1, self.state_size), (
                    f"simcity_wrapper State shape mismatch: {state.shape} vs expected {(1, self.state_size)}"
                )
                logger.debug("simcity_wrapper Initial state shape is correct.")
            except AssertionError as e:
                logger.error(e)
                raise
    
    def get_obs(self):
        """Returns all agent observations in a list"""
        observations = []
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            flat_obs = np.concatenate([
                obs["grid"].flatten(),
                [obs["resources"]["money"], obs["resources"]["reputation"]],
                obs["builders"].flatten()
            ])
            observations.append(flat_obs)
        # Convert to numpy array with correct shape [1, n_agents, obs_size]
        obs_array = np.array(observations, dtype=np.float32)[np.newaxis]
        logger.debug(f"simcity_wrapper Generated aggregated observations with shape: {obs_array.shape}")
        return obs_array
    
    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        agent = self.env.agents[agent_id]
        obs = self.env.observe(agent)
        agent_obs = np.concatenate([
            obs["grid"].flatten(),
            [obs["resources"]["money"], obs["resources"]["reputation"]],
            obs["builders"].flatten()
        ]).astype(np.float32)
        logger.debug(f"simcity_wrapper Observation for agent {agent_id}: {agent_obs.shape}")
        return agent_obs
    
    def get_state(self):
        """Returns the global state"""
        obs = self.get_obs()
        state = obs.reshape(1, -1)
        logger.debug(f"simcity_wrapper Generated global state with shape: {state.shape}")
        return state
    
    def get_avail_actions(self):
        """Returns the available actions of all agents"""
        avail_actions = []
        for _ in range(self.n_agents):
            avail_actions.append([1] * self.n_actions)  # All actions available
        avail_actions_array = np.array(avail_actions, dtype=np.float32)[np.newaxis]
        logger.debug(f"simcity_wrapper Available actions shape: {avail_actions_array.shape}")
        return avail_actions_array
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        avail_actions = [1] * self.n_actions
        logger.debug(f"simcity_wrapper Available actions for agent {agent_id}: {avail_actions}")
        return avail_actions
    
    def step(self, actions):
        """Execute actions and return reward, terminated, info"""
        logger.debug(f"Executing step with actions: {actions}")
        rewards = []
        for idx, agent in enumerate(self.env.agents):
            logger.debug(f"simcity_wrapper Agent {agent} taking action {actions[idx]}")
            self.env.step(actions[idx])
            agent_reward = self.env._cumulative_rewards.get(agent, 0)
            rewards.append(agent_reward)
            logger.debug(f"simcity_wrapper Agent {agent} received reward: {agent_reward}")
        
        # Aggregate rewards if common_reward is True
        if self.env.common_reward:
            total_reward = sum(rewards)
            logger.debug(f"simcity_wrapper Aggregated total reward (common_reward=True): {total_reward}")
        else:
            total_reward = np.array(rewards)
            logger.debug(f"simcity_wrapper Individual rewards: {total_reward}")
        
        # Get termination status
        # terminated = all(self.env.terminations.values())
        # truncated = all(self.env.truncations.values())

        terminated = [False for _ in range(self.n_agents)]
        truncated = [False for _ in range(self.n_agents)]
        logger.debug(f"simcity_wrapper Termination status - Terminated: {terminated}, Truncated: {truncated}")
        
        # Log current number of agents
        current_agents = len(self.env.agents)
        logger.debug(f"simcity_wrapper Number of active agents after step: {current_agents}")
        
        return self.get_obs(), total_reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        logger.debug("simcity_wrapper Resetting the underlying SimCityEnv.")
        _, info = self.env.reset(seed=seed, options=options)
        
        # Update number of agents in case it changes
        self.n_agents = len(self.env.agents)
        logger.debug(f"simcity_wrapper Number of agents after reset: {self.n_agents}")
        
        # Recalculate observation and state sizes if necessary
        if self.n_agents > 0:
            single_obs = self.env.observe(self.env.agents[0])
            self.obs_size = (
                single_obs["grid"].size +  # grid observations
                2 +  # money and reputation
                single_obs["builders"].size  # builders information
            )
            logger.debug(f"simcity_wrapper Updated observation size per agent: {self.obs_size}")
        else:
            self.obs_size = 0  # Handle cases with zero agents
            logger.warning("No agents present after reset.")
        
        self.state_size = self.obs_size * self.n_agents
        logger.debug(f"simcity_wrapper Updated state size: {self.state_size}")
        
        # Get number of actions
        self.n_actions = self.env.action_spaces["P1"].n if self.n_agents > 0 else 0
        logger.debug(f"simcity_wrapper Updated number of actions per agent: {self.n_actions}")
        
        # Generate aggregated observations using get_obs()
        obs = self.get_obs()
        logger.debug(f"simcity_wrapper Observation shape after reset: {obs.shape}")
        
        # Ensure the shapes are correct
        if self.n_agents > 0:
            expected_obs_shape = (1, self.n_agents, self.obs_size)
        else:
            expected_obs_shape = (1, 0, self.obs_size)
        
        try:
            assert obs.shape == expected_obs_shape, (
                f"simcity_wrapper Observation shape mismatch: {obs.shape} vs expected {expected_obs_shape}"
            )
            logger.debug("simcity_wrapper Reset observation shape is correct.")
        except AssertionError as e:
            logger.error(e)
            raise
        
        if self.state_size > 0:
            state = self.get_state()
            logger.debug(f"State shape after reset: {state.shape}")
            try:
                assert state.shape == (1, self.state_size), (
                    f"simcity_wrapper State shape mismatch: {state.shape} vs expected {(1, self.state_size)}"
                )
                logger.debug("simcity_wrapper Reset state shape is correct.")
            except AssertionError as e:
                logger.error(e)
                raise
        
        return obs, info
    
    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        logger.debug(f"simcity_wrapper Environment info: {env_info}")
        return env_info
    
    def get_stats(self):
        return {}
    
    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.obs_size
    
    def get_state_size(self):
        """Returns the shape of the state"""
        return self.state_size
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.n_actions
    
    def render(self):
        logger.debug("simcity_wrapper Rendering environment.")
        return self.env.render()
    
    def close(self):
        logger.debug("simcity_wrapper Closing environment.")
        self.env.close()
    
    def seed(self, seed=None):
        """Seeds the environment"""
        logger.debug(f"simcity_wrapper Seeding environment with seed: {seed}")
        return self.reset(seed=seed)[0]
