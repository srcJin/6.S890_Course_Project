# server_scale_up_simple.py - Simple inference server for SimCity Scale-Up Environment

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import random
import os, sys
import logging

# Ensure src/ is in sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import environment wrapper for scale-up environment
from envs.simcity_scale_up_wrapper import SimCityScaleUpWrapper

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------------
# 1. Global Initialization
# --------------------------

# Initialize scaled-up environment (8x8 grid, 4 players, 5 building types)
env = SimCityScaleUpWrapper(grid_x=8, grid_y=8, common_reward=False)
env_info = env.get_env_info()
n_agents = env.n_agents
n_actions = env_info.get("n_actions", env.n_actions)

logger.info(
    "Scale-up environment initialized: n_agents=%d, n_actions=%d, obs_size=%d, episode_limit=%d",
    n_agents,
    n_actions,
    env.obs_size,
    env.episode_limit,
)

# Global variables for episode tracking
current_episode = None
episode_counter = 0

# --------------------------
# 2. API Endpoints
# --------------------------


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "environment": "SimCity Scale-Up",
            "grid_size": "8x8",
            "n_agents": int(n_agents),
            "n_actions": int(n_actions),
            "building_types": 5,
            "obs_size": int(env.obs_size),
        }
    )


@app.route("/reset", methods=["POST"])
def reset_environment():
    """Reset the environment and return initial observations"""
    global current_episode, episode_counter

    try:
        logger.info("Resetting scale-up environment")

        # Reset environment
        obs, info = env.reset()

        # Create new episode tracking
        episode_counter += 1
        current_episode = {
            "episode_id": episode_counter,
            "observations": [],
            "actions": [],
            "rewards": [],
            "step_count": 0,
        }

        # Get initial observations and available actions
        avail_actions = env.get_avail_actions()

        # Store initial observation
        current_episode["observations"].append(obs.tolist())

        response = {
            "status": "success",
            "episode_id": episode_counter,
            "observation": obs.tolist(),
            "avail_actions": avail_actions.tolist(),
            "info": info,
            "current_agent": (
                env.env.agent_selection if hasattr(env.env, "agent_selection") else "P1"
            ),
        }

        logger.info(f"Environment reset successful, episode {episode_counter}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in reset: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/step", methods=["POST"])
def step_environment():
    """Execute one step in the environment"""
    global current_episode

    try:
        data = request.get_json()
        human_action = data.get("action", 0)  # Default to no-op
        agent_id = data.get("agent_id", 0)  # Which agent is taking the action

        logger.info(
            f"Executing step with human action: {human_action} for agent {agent_id}"
        )

        if current_episode is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Environment not initialized. Call /reset first.",
                    }
                ),
                400,
            )

        # Use random actions for other agents (simple AI)
        actions = []
        for i in range(n_agents):
            if i == agent_id:
                actions.append(human_action)
            else:
                # Random action from valid range [0, n_actions-1]
                actions.append(random.randint(0, n_actions - 1))

        logger.info(f"Combined actions: {actions}")

        # Execute step
        rewards, next_obs, dones, infos, episode_done = env.step(actions)

        # Update episode tracking
        current_episode["actions"].append(actions)
        current_episode["rewards"].append(rewards.tolist())
        current_episode["observations"].append(next_obs.tolist())
        current_episode["step_count"] += 1

        # Get updated available actions
        next_avail_actions = env.get_avail_actions()

        # Prepare response
        response = {
            "status": "success",
            "observation": next_obs.tolist(),
            "rewards": rewards.tolist(),
            "dones": dones.tolist(),
            "episode_done": bool(episode_done),
            "avail_actions": next_avail_actions.tolist(),
            "actions_taken": actions,
            "info": {
                "step_count": current_episode["step_count"],
                "human_action": human_action,
                "infos": infos,
            },
            "current_agent": (
                env.env.agent_selection if hasattr(env.env, "agent_selection") else "P1"
            ),
        }

        # If episode is done, store it
        if episode_done:
            logger.info(
                f"Episode {current_episode['episode_id']} completed with {current_episode['step_count']} steps"
            )
            current_episode = None

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in step: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def simulate_full_episode():
    """Simulate a complete episode with random AI agents only"""
    try:
        logger.info("Starting full random AI simulation")

        # Reset environment
        obs, info = env.reset()

        episode_records = []
        step_count = 0
        max_steps = env.episode_limit

        while step_count < max_steps:
            # Get current state
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            # Generate random actions for all agents
            actions = [random.randint(0, n_actions - 1) for _ in range(n_agents)]

            # Store step data
            step_data = {
                "t_env": step_count,
                "observation": obs.tolist(),
                "actions_taken": actions,
                "avail_actions": avail_actions.tolist(),
            }

            # Execute step
            rewards, next_obs, dones, infos, episode_done = env.step(actions)

            step_data.update(
                {"rewards": rewards.tolist(), "dones": dones.tolist(), "info": infos}
            )

            episode_records.append(step_data)
            step_count += 1

            if episode_done:
                break

        logger.info(f"Simulation completed with {step_count} steps")

        return jsonify(
            {
                "status": "success",
                "episode_records": episode_records,
                "total_steps": step_count,
            }
        )

    except Exception as e:
        logger.error(f"Error in simulate: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/env_info", methods=["GET"])
def get_env_info():
    """Get environment information"""
    try:
        env_info_dict = env.get_env_info()
        env_info_dict.update(
            {
                "building_types": env.env.BUILDING_TYPES,
                "grid_layout": env.env.grid_layout.tolist(),
                "terrain_map": list(env.env.terrain_map.items()),
                "infrastructure_map": list(env.env.infrastructure_map.items()),
            }
        )

        return jsonify({"status": "success", "env_info": env_info_dict})

    except Exception as e:
        logger.error(f"Error in env_info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/render", methods=["GET"])
def render_environment():
    """Get a text rendering of the current environment state"""
    try:
        # Capture the render output
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        env.render()

        sys.stdout = old_stdout
        render_text = captured_output.getvalue()

        return jsonify({"status": "success", "render": render_text})

    except Exception as e:
        logger.error(f"Error in render: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --------------------------
# 3. Main Function
# --------------------------

if __name__ == "__main__":
    logger.info("Starting SimCity Scale-Up Simple Inference Server")
    logger.info(
        f"Environment: 8x8 grid, {n_agents} agents, {n_actions} actions, {env.obs_size} obs_size"
    )
    logger.info("Server will run on http://127.0.0.1:5888")

    # Run Flask app
    app.run(host="127.0.0.1", port=5888, debug=False)
