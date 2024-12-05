# episode_runner.py
import logging
from functools import partial
import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2

from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")

class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()
    
        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
    
        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
    
            logger.debug(f"Pre-transition data at time step {self.t}: {pre_transition_data}")
    
            self.batch.update(pre_transition_data, ts=self.t)
    
            # Select actions
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            logger.debug(f"Selected actions: {actions}")
    
            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            logger.debug(f"Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            #
            if terminated or truncated:
                logger.debug(f"Episode ended due to termination: {terminated}, truncation: {truncated}")
            # Reset logic

            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward
            logger.debug(f"Episode return so far: {episode_return}")
    
            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]
    
            logger.debug(f"Post-transition data at time step {self.t}: {post_transition_data}")
    
            self.batch.update(post_transition_data, ts=self.t)
    
            self.t += 1
    
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        logger.debug(f"Last data before episode ends: {last_data}")
    
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)
    
        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        logger.debug(f"Selected actions for last state: {actions}")
    
        self.batch.update({"actions": actions}, ts=self.t)
    
        # Update stats
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
    
        if not test_mode:
            self.t_env += self.t
    
        cur_returns.append(episode_return)
    
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env
    
        return self.batch


    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
            logger.debug(f"{prefix}return_mean: {np.mean(returns)}, {prefix}return_std: {np.std(returns)}")
        else:
            for i in range(self.args.n_agents):
                agent_mean = np.array(returns)[:, i].mean()
                agent_std = np.array(returns)[:, i].std()
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    agent_mean,
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    agent_std,
                    self.t_env,
                )
                logger.debug(f"{prefix}agent_{i}_return_mean: {agent_mean}, {prefix}agent_{i}_return_std: {agent_std}")
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
            logger.debug(f"{prefix}total_return_mean: {total_returns.mean()}, {prefix}total_return_std: {total_returns.std()}")
        
        returns.clear()
    
        for k, v in stats.items():
            if k != "n_episodes":
                mean_val = v / stats["n_episodes"]
                self.logger.log_stat(
                    prefix + k + "_mean", mean_val, self.t_env
                )
                logger.debug(f"{prefix}{k}_mean: {mean_val}")
        stats.clear()
