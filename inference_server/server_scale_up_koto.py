# server_scale_up_koto.py - Inference server for SimCity Scale-Up Koto Environment

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import numpy as np
import os, sys
from argparse import Namespace
import logging
import warnings

# Ensure src/ is in sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# Temporarily disable FutureWarning for torch.load
warnings.simplefilter("ignore", FutureWarning)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from components.episode_buffer import EpisodeBatch

# Import environment wrapper for koto environment
from envs.simcity_scale_up_koto_wrapper import SimCityScaleUpKotoWrapper

# Import multi-agent controller
from controllers.basic_controller import BasicMAC

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------------
# 1. Global Initialization
# --------------------------

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# Initialize koto environment (12x12 grid, 4 players, 8 building types)
env = SimCityScaleUpKotoWrapper(grid_x=12, grid_y=12, common_reward=False)
env_info = env.get_env_info()
n_agents = env.n_agents
n_actions = env_info.get("n_actions", env.n_actions)

logger.info(
    "Koto environment initialized: n_agents=%d, n_actions=%d, obs_size=%d, episode_limit=%d",
    n_agents,
    n_actions,
    env.obs_size,
    env.episode_limit,
)

# Create args object for configuration
args = Namespace(
    action_selector="soft_policies",
    mask_before_softmax=True,
    runner="parallel",
    buffer_size=10,
    batch_size_run=10,
    batch_size=10,
    target_update_interval_or_tau=0.01,
    lr=0.0005,
    hidden_dim=128,
    obs_agent_id=True,
    obs_last_action=False,
    obs_individual_obs=False,
    agent_output_type="pi_logits",
    learner="actor_critic_learner",
    entropy_coef=0.01,
    use_rnn=True,
    standardise_returns=False,
    standardise_rewards=True,
    q_nstep=5,
    critic_type="cv_critic",
    name="maa2c",
    t_max=20050000,
    n_agents=n_agents,
    n_actions=n_actions,
    device=device,
    gamma=0.99,
    grad_norm_clip=10.0,
    agent="rnn",
)

# Define scheme for EpisodeBatch
scheme = {
    "obs": {"vshape": int(env.obs_size), "group": "agents"},
    "avail_actions": {"vshape": int(n_actions), "group": "agents"},
}

groups = {"agents": n_agents}

# Initialize MAC (Multi-Agent Controller)
mac = BasicMAC(scheme, groups, args)

# Global variables for episode tracking
episode_data = []
current_episode = None
episode_counter = 0

# --------------------------
# 2. Model Loading Function
# --------------------------


def load_model(model_path):
    """Load trained model from directory containing model files"""
    global mac
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            # Check if it's a directory with separate model files
            if os.path.isdir(model_path):
                mac.load_models(model_path)
                logger.info("Model loaded successfully from directory")
            else:
                # Try loading as a single checkpoint file
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False
                )
                mac.load_state(checkpoint)
                logger.info("Model loaded successfully from checkpoint")
            return True
        else:
            logger.warning(f"Model file/directory not found: {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


# Try to load default koto model if available
default_model_path = os.path.join(os.path.dirname(__file__), "saved_models_scale_up_koto")

# Load the koto models
if os.path.exists(default_model_path):
    try:
        logger.info(f"Loading koto models from: {default_model_path}")
        mac.load_models(default_model_path)
        logger.info("Koto models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading koto models: {e}")
        logger.info("Continuing with random policy")
else:
    logger.warning(f"Koto models directory not found: {default_model_path}")
    logger.info("Using random policy for testing")

# --------------------------
# 3. API Endpoints
# --------------------------


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "environment": "SimCity Scale-Up Koto",
            "grid_size": "12x12",
            "n_agents": int(n_agents),
            "n_actions": int(n_actions),
            "building_types": 8,
            "parameters": ["G", "V", "D", "A", "S", "F"],
        }
    )


@app.route("/reset", methods=["POST"])
def reset_environment():
    """Reset the environment and return initial observations"""
    global current_episode, episode_counter

    try:
        logger.info("Resetting koto environment")

        # Reset environment
        obs, info = env.reset()

        # Reset MAC hidden states
        mac.init_hidden(batch_size=1)

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

        # Get actions for all agents using the trained models
        obs = env.get_obs()
        avail_actions = env.get_avail_actions()

        # Convert to PyTorch tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        avail_actions_tensor = torch.tensor(
            avail_actions, dtype=torch.float32, device=device
        )

        # Create batch for MAC
        batch = EpisodeBatch(scheme, groups, 1, 1)
        batch.update(
            {
                "obs": obs_tensor.reshape(1, 1, n_agents, -1),
                "avail_actions": avail_actions_tensor.reshape(1, 1, n_agents, -1),
            },
            bs=None,
            ts=0,
            mark_filled=True,
        )

        # Get actions from MAC (trained models)
        ai_actions = mac.select_actions(
            batch, t_ep=0, t_env=current_episode["step_count"], test_mode=True
        )
        ai_actions = ai_actions.cpu().numpy().flatten()

        # Override the human player's action
        actions = ai_actions.copy()
        actions[agent_id] = human_action

        logger.info(f"Combined actions: {actions}")

        # Execute step
        next_obs, rewards, terminated, truncated, env_info = env.step(actions)
        episode_done = terminated or truncated

        # Handle rewards - ensure they are in list format
        if hasattr(rewards, "tolist"):
            rewards_list = rewards.tolist()
        else:
            rewards_list = [float(rewards)] * n_agents

        # Update episode tracking
        current_episode["actions"].append(actions)
        current_episode["rewards"].append(rewards_list)
        current_episode["observations"].append(next_obs.tolist())
        current_episode["step_count"] += 1

        # Get updated available actions
        next_avail_actions = env.get_avail_actions()

        # Prepare response
        response = {
            "status": "success",
            "observation": next_obs.tolist(),
            "rewards": rewards_list,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_done": bool(episode_done),
            "avail_actions": next_avail_actions.tolist(),
            "actions_taken": actions,
            "info": {
                "step_count": current_episode["step_count"],
                "human_action": human_action,
                "env_info": env_info,
            },
            "current_agent": (
                env.env.agent_selection if hasattr(env.env, "agent_selection") else "P1"
            ),
        }

        # If episode is done, store it
        if episode_done:
            episode_data.append(current_episode.copy())
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
    """Simulate a complete episode with AI agents only"""
    try:
        logger.info("Starting full AI simulation")

        # Reset environment
        obs, info = env.reset()
        mac.init_hidden(batch_size=1)

        episode_records = []
        step_count = 0
        max_steps = env.episode_limit

        while step_count < max_steps:
            # Get current state
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            # Convert to PyTorch tensors
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            avail_actions_tensor = torch.tensor(
                avail_actions, dtype=torch.float32, device=device
            )

            # Create batch for MAC
            batch = EpisodeBatch(scheme, groups, 1, 1)
            batch.update(
                {
                    "obs": obs_tensor.reshape(1, 1, n_agents, -1),
                    "avail_actions": avail_actions_tensor.reshape(1, 1, n_agents, -1),
                },
                bs=None,
                ts=0,
                mark_filled=True,
            )

            # Get actions from MAC
            actions = mac.select_actions(
                batch, t_ep=0, t_env=step_count, test_mode=True
            )
            actions = actions.cpu().numpy().flatten()

            # Store step data
            step_data = {
                "t_env": step_count,
                "observation": [obs.tolist()],  # Wrapped in list for consistency
                "actions_taken": actions.tolist(),
                "avail_actions": avail_actions.tolist(),
            }

            # Execute step
            next_obs, rewards, terminated, truncated, env_info = env.step(actions)
            episode_done = terminated or truncated

            step_data.update(
                {
                    "rewards": (
                        rewards.tolist()
                        if hasattr(rewards, "tolist")
                        else [float(rewards)] * n_agents
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": env_info,
                }
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


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Load a trained model"""
    try:
        data = request.get_json()
        model_path = data.get("model_path")

        if not model_path:
            return (
                jsonify({"status": "error", "message": "model_path is required"}),
                400,
            )

        success = load_model(model_path)

        if success:
            return jsonify(
                {"status": "success", "message": f"Model loaded from {model_path}"}
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to load model from {model_path}",
                    }
                ),
                500,
            )

    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/env_info", methods=["GET"])
def get_env_info():
    """Get environment information"""
    try:
        env_info_dict = env.get_env_info()
        env_info_dict.update(
            {
                "building_types": getattr(env.env, 'BUILDING_TYPES', {}),
                "grid_layout": env.env.grid.tolist(),
                "terrain_and_projects": getattr(env.env, 'terrain_and_projects', {}),
                "parameters": ["G", "V", "D", "A", "S", "F"],
                "parameter_names": {
                    "G": "Greenery",
                    "V": "Vitality", 
                    "D": "Density",
                    "A": "Adaptability",
                    "S": "Sustainability",
                    "F": "Flood_Resistance"
                }
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
# 4. Main Function
# --------------------------

if __name__ == "__main__":
    logger.info("Starting SimCity Scale-Up Koto Inference Server")
    logger.info(f"Environment: 12x12 grid, {n_agents} agents, {n_actions} actions")
    logger.info("Server will run on http://127.0.0.1:5889")

    # Run Flask app
    app.run(host="127.0.0.1", port=5889, debug=False)