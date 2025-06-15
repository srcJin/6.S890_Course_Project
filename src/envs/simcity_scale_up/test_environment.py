#!/usr/bin/env python3
"""
Test script for the Urban Resilience SimCity Scale-Up Environment
"""
import sys
import os

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

import numpy as np

# Direct import to avoid envs.__init__ issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "simcity_scale_up_env", os.path.join(os.path.dirname(__file__), "environment.py")
)
env_module = importlib.util.module_from_spec(spec)

# We need to add the required dependencies to sys.modules first
sys.modules["simcity_scale_up_env"] = env_module

# Import required modules manually
from pettingzoo.utils import AECEnv, agent_selector
from gymnasium import spaces

# Now import the config and other modules
import importlib.util


def load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load required modules
config_module = load_module(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
)
players_module = load_module(
    "players", os.path.join(os.path.dirname(__file__), "players.py")
)
log_module = load_module("log", os.path.join(os.path.dirname(__file__), "log.py"))

# Load utils.logging
utils_logging_path = os.path.join(project_root, "src", "utils", "logging.py")
utils_logging = load_module("utils.logging", utils_logging_path)

# Now load the environment
spec.loader.exec_module(env_module)
SimCityScaleUpEnv = env_module.SimCityScaleUpEnv


def test_environment():
    """Test the scaled-up SimCity environment"""
    print("Testing Urban Resilience SimCity Scale-Up Environment")
    print("=" * 60)

    # Initialize environment
    env = SimCityScaleUpEnv(grid_x=8, grid_y=8)

    print(f"Environment initialized with {len(env.agents)} agents: {env.agents}")
    print(f"Grid size: {env.grid_x}x{env.grid_y}")
    print(f"Building types: {env.BUILDING_TYPES}")
    print(f"Total actions: {env.total_actions}")

    # Reset environment
    obs, info = env.reset()
    print(f"\nEnvironment reset. First agent: {env.agent_selection}")

    # Display initial grid
    print("\nInitial Grid Layout:")
    env.render()

    # Test a few actions
    print("\nTesting actions...")
    for i in range(10):
        agent = env.agent_selection

        # Choose a random valid action
        action = np.random.randint(0, env.total_actions)

        print(f"Step {i+1}: Agent {agent} takes action {action}")

        # Take action
        env.step(action)

        # Show current state
        current_obs = env.observe(agent)
        print(f"  Agent {agent} resources: {current_obs['resources']}")

        # Check if game is over
        if env.terminations[agent] or env.truncations[agent]:
            print("Game ended!")
            break

    # Final grid display
    print("\nFinal Grid Layout:")
    env.render()

    # Display final scores
    print("\nFinal Scores:")
    for agent in env.agents:
        player = env.players[agent]
        print(
            f"Agent {agent}: Self Score: {player.self_score:.2f}, "
            f"Integrated Score: {player.integrated_score:.2f}"
        )

    print(f"\nEnvironment Score: {env.env_score:.2f}")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_environment()
