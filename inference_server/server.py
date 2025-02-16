# server.py

from flask import Flask, request, jsonify
import torch
import numpy as np
import os, sys
from argparse import Namespace
import logging

# 配置日志（可根据需要调整日志级别）
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure src/ is in sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# 导入环境封装类
from envs.simcity_wrapper import SimCityWrapper

# 导入多智能体控制器 BasicMAC（该模块内部会调用已注册的 RNNAgent）
from controllers.basic_controller import BasicMAC

app = Flask(__name__)

# --------------------------
# 1. 全局初始化
# --------------------------

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# 初始化环境
env = SimCityWrapper(grid_x=4, grid_y=4, common_reward=False)
env_info = env.get_env_info()  # 包含 state_shape、obs_shape、n_actions、n_agents 等信息
n_agents = env.n_agents
# 如果 env_info 中没有 n_actions 字段，则使用 env.n_actions
n_actions = env_info.get("n_actions", env.n_actions)
logger.info(
    "Environment initialized: n_agents=%d, n_actions=%d, obs_size=%d",
    n_agents,
    n_actions,
    env.obs_size,
)

# 创建一个 args 对象（模拟配置文件中的参数）
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
    agent="rnn_agent",  # 这里的 "rnn_agent" 必须与你在 modules/agents/rnn_agent.py 中注册的 key 保持一致
)

# 定义一个 minimal scheme 用于构造 MAC
scheme = {
    "obs": {"vshape": env.obs_size},
    # 如果 obs_last_action 为 True，则需要添加 "actions_onehot" 字段；此处为 False，可省略
}

# groups 参数在这里不做特殊处理，传空字典即可
groups = {}

# 初始化多智能体控制器（MAC）
mac = BasicMAC(scheme, groups, args)
mac.to(device)
# 初始化 MAC 隐状态（batch_size=1 表示单个 episode 推理）
mac.init_hidden(batch_size=1)
logger.info("MAC initialized.")

# 加载训练好的 agent 模型参数
model_save_path = os.path.join("saved_models")
agent_model_path = os.path.join(model_save_path, "agent.th")
if os.path.exists(agent_model_path):
    mac.load_models(model_save_path)
    logger.info("Loaded trained agent model from %s", model_save_path)
else:
    logger.warning(
        "Trained model not found at %s. Running with untrained model.", model_save_path
    )

# 全局环境时间计数器
t_env = 0

# --------------------------
# 2. Flask 接口定义
# --------------------------


@app.route("/reset", methods=["POST"])
def reset_env():
    """
    重置环境，同时重置 MAC 隐状态，并返回初始观测和环境信息。
    """
    global t_env
    obs, info = env.reset()
    t_env = 0
    # 重置 MAC 隐状态
    mac.init_hidden(batch_size=1)
    logger.info("Environment reset. obs shape: %s", np.array(obs).shape)
    return jsonify(
        {
            "observation": np.array(obs).tolist(),
            "info": info,
            "message": "Environment has been reset successfully.",
        }
    )


@app.route("/step", methods=["POST"])
def step_env():
    """
    执行一步交互：
      1. 获取当前环境观测及可用动作。
      2. 利用 MAC 计算动作（允许前端通过 user_actions 指定部分 agent 的动作）。
      3. 调用 env.step(actions) 更新环境。
      4. 返回新的观测、奖励、终止标志等信息。
    """
    global t_env
    data = request.get_json()
    # 用户可以传入部分 agent 的动作，格式如：{"0": 2} 表示 agent0 由用户指定动作 2
    user_actions = data.get("user_actions", {})

    # 获取当前观测和可用动作
    obs = env.get_obs()  # 形状: (1, n_agents, obs_size)
    avail_actions = env.get_avail_actions()  # 形状: (1, n_agents, n_actions)

    # 构造一个 dummy batch，用于调用 MAC.select_actions
    # 注意：MAC.select_actions 期望 batch["obs"] 的形状为 (batch_size, episode_length, n_agents, obs_dim)
    batch_obs = np.reshape(obs, (1, 1, n_agents, env.obs_size))
    batch_avail = np.reshape(avail_actions, (1, 1, n_agents, n_actions))
    dummy_batch = {
        "obs": batch_obs,
        "avail_actions": batch_avail,
        "batch_size": 1,
        "device": device,
    }

    # 使用 MAC 选取动作，t_ep 固定为 0（单步推理），t_env 为全局步数
    mac_actions = mac.select_actions(dummy_batch, t_ep=0, t_env=t_env, test_mode=True)
    mac_actions = mac_actions.tolist()  # 转为列表，长度为 n_agents

    # 如果前端指定了部分 agent 的动作，则覆盖 MAC 输出
    final_actions = []
    for i in range(n_agents):
        if str(i) in user_actions:
            final_actions.append(int(user_actions[str(i)]))
        else:
            final_actions.append(mac_actions[i])
    final_actions = np.array(final_actions)
    logger.debug("Actions chosen: %s", final_actions.tolist())

    # 将动作传递给环境，执行一步
    obs_next, rewards, terminated, truncated, info = env.step(final_actions)
    t_env += 1

    response = {
        "actions_taken": final_actions.tolist(),
        "next_observation": np.array(obs_next).tolist(),
        "rewards": (
            np.array(rewards).tolist() if isinstance(rewards, np.ndarray) else rewards
        ),
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "t_env": t_env,
    }
    return jsonify(response)


@app.route("/simulate", methods=["POST"])
def simulate_episode():
    """
    运行一个完整的 self play episode（所有 agent 均由模型控制），直至游戏结束，
    并返回每一步的记录（包括动作、观测、奖励等）。
    """
    global t_env
    episode_records = []

    # 重置环境和 MAC 隐状态
    obs, info = env.reset()
    t_env = 0
    mac.init_hidden(batch_size=1)

    terminated = False
    while not terminated:
        # 获取当前状态和可用动作
        obs = env.get_obs()  # (1, n_agents, obs_size)
        avail_actions = env.get_avail_actions()  # (1, n_agents, n_actions)

        # 构造 dummy batch
        batch_obs = np.reshape(obs, (1, 1, n_agents, env.obs_size))
        batch_avail = np.reshape(avail_actions, (1, 1, n_agents, n_actions))
        dummy_batch = {
            "obs": batch_obs,
            "avail_actions": batch_avail,
            "batch_size": 1,
            "device": device,
        }

        # 使用 MAC 计算动作（所有 agent 均由模型控制）
        mac_actions = mac.select_actions(
            dummy_batch, t_ep=0, t_env=t_env, test_mode=True
        )
        mac_actions = mac_actions.tolist()
        actions = np.array(mac_actions)
        logger.debug("t_env=%d, actions=%s", t_env, actions.tolist())

        # 执行环境步
        obs_next, rewards, terminated, truncated, info = env.step(actions)
        t_env += 1

        step_record = {
            "t_env": t_env,
            "actions": actions.tolist(),
            "observation": np.array(obs_next).tolist(),
            "rewards": (
                np.array(rewards).tolist()
                if isinstance(rewards, np.ndarray)
                else rewards
            ),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        episode_records.append(step_record)

        if terminated or truncated:
            break

    return jsonify(
        {"episode_records": episode_records, "message": "Episode simulation completed."}
    )


# --------------------------
# 3. 启动 Flask 服务
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
