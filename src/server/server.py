# app/server.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch as th
import numpy as np
from envs.simcity_wrapper import SimCityWrapper
from envs.simcity.players import BalancedPlayer
from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")

app = Flask(__name__)
CORS(app)


env = None

def initialize_environment():
    global env
    logger.debug("Initializing SimCityWrapper for Flask server.")
    env = SimCityWrapper(common_reward=False)
    env.reset()
    logger.debug("SimCityWrapper initialized and environment reset successfully.")

@app.route("/reset", methods=["POST"])
def reset_env():
    logger.debug("Received /reset request")
    try:
        data = request.get_json()
        if data and 'common_reward' in data:
            common_reward = data['common_reward']
            env.env.common_reward = common_reward
            logger.debug(f"Set common_reward to {common_reward}")
        obs = env.reset()[0].tolist()
        state = env.get_state().tolist()
        logger.debug("Environment reset complete.")
        return jsonify({"status": "success", "obs": obs, "state": state})
    except Exception as e:
        logger.error(f"Error resetting environment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/step", methods=["POST"])
def step_env():
    logger.debug("Received /step request")
    try:
        data = request.get_json()
        if not data or 'actions' not in data:
            logger.warning("No actions provided.")
            return jsonify({"status": "error", "message": "No actions provided."}), 400

        user_actions = data['actions']
        if set(user_actions.keys()) != set(env.env.agents):
            logger.warning("Actions for all agents must be provided.")
            return jsonify({"status": "error", "message": "Actions for all agents must be provided."}), 400

        actions = []
        for agent in env.env.agents:
            action = user_actions.get(agent, -1)
            if not isinstance(action, int) or action < 0 or action >= env.n_actions:
                logger.warning(f"Invalid action for agent {agent}: {action}")
                return jsonify({"status": "error", "message": f"Invalid action for agent {agent}."}), 400
            actions.append(action)

        logger.debug(f"Final actions for step: {actions}")
        actions_array = np.array(actions, dtype=np.int32)
        actions_tensor = th.tensor(actions_array).unsqueeze(0)

        obs, reward, terminated, truncated, info = env.step(actions_tensor)
        response = {
            "status": "success",
            "obs": obs.tolist(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        logger.debug("Step execution successful.")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error during step execution: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/state", methods=["GET"])
def get_state():
    try:
        state = env.get_state().tolist()
        return jsonify({"status": "success", "state": state})
    except Exception as e:
        logger.error(f"Error fetching state: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/obs", methods=["GET"])
def get_obs():
    try:
        obs = env.get_obs().tolist()
        return jsonify({"status": "success", "obs": obs})
    except Exception as e:
        logger.error(f"Error fetching obs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/info", methods=["GET"])
def get_info():
    try:
        info = {"env_score": env.env.env_score}
        return jsonify({"status": "success", "info": info})
    except Exception as e:
        logger.error(f"Error fetching info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/render", methods=["GET"])
def render_env():
    try:
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        env.render()
        sys.stdout = old_stdout
        render_output = mystdout.getvalue()
        return jsonify({"status": "success", "render": render_output})
    except Exception as e:
        logger.error(f"Error rendering environment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    try:
        initialize_environment()
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        raise
