# SimCity Scale-Up Inference Server Guide

## Overview

The Scale-Up Inference Server is an enhanced version of the SimCity multi-agent reinforcement learning environment that provides a larger, more complex simulation with 4 agents operating on an 8x8 grid instead of the standard 4x4 grid.

## 🏗️ Architecture

### Environment Specifications
- **Grid Size**: 8x8 (vs 4x4 in standard version)
- **Number of Agents**: 4 (vs 3 in standard version)
- **Action Space**: 321 actions per agent (vs 49 in standard version)
- **Observation Space**: 514-dimensional observations (vs 82 in standard version)
- **Algorithm**: MAPPO (Multi-Agent Proximal Policy Optimization)

### Key Features
- **Individual Rewards**: Each agent receives individual rewards for their actions
- **Larger Action Space**: More building types and placement options
- **Complex Interactions**: More sophisticated multi-agent coordination required
- **Trained Models**: Pre-trained MAPPO models for intelligent agent behavior

## 🚀 Setup Instructions

### Prerequisites
1. **Conda Environment**: Ensure you have conda installed
2. **Python 3.8**: Required for compatibility
3. **Dependencies**: All EPyMARL requirements must be installed

### Step-by-Step Setup

#### 1. Create and Activate Environment
```bash
cd /path/to/6.S890_Course_Project
conda create -n epymarl python=3.8 -y
conda activate epymarl
```

#### 2. Install Dependencies
```bash
# Install basic requirements
pip install einops
pip install -r requirements.txt
pip install -r env_requirements.txt
pip install -r pac_requirements.txt

# Install PettingZoo (required for scale-up environment)
pip install pettingzoo
```

#### 3. Verify Environment
```bash
python -c "import sys; sys.path.insert(0, 'src'); from envs.simcity_scale_up_wrapper import SimCityScaleUpWrapper; env = SimCityScaleUpWrapper(grid_x=8, grid_y=8, common_reward=False); print('✅ Environment loaded successfully')"
```

## 🎮 Running the Server

### Start the Scale-Up Server

First, ensure no other server is running on port 5888:
```bash
# Kill any existing inference servers
pkill -f "inference_server"

# Or specifically kill processes using port 5888
lsof -ti:5888 | xargs kill -9 2>/dev/null || true
```

Then start the scale-up server:
```bash
conda activate epymarl
python inference_server/inference_server_scale_up.py
```

The server will start on port 5888 by default with the following output:
```
[INFO] Using device: cpu
[INFO] Environment initialized: n_agents=4, n_actions=321, obs_size=514
[INFO] MAC initialized.
[INFO] Loaded trained agent model from inference_server/saved_models_scale_up
```

### Verify Server is Running
```bash
curl -X POST http://127.0.0.1:5888/reset
```

## 📡 API Endpoints

### 1. Reset Environment
Resets the environment and returns initial observations.

**Endpoint**: `POST /reset`
```bash
curl -X POST http://127.0.0.1:5888/reset
```

**Response**:
```json
{
  "observation": [/* 4 agents x 514-dimensional observations */],
  "info": {},
  "message": "Environment has been reset successfully."
}
```

### 2. Step Environment
Executes one step in the environment with optional user-specified actions.

**Endpoint**: `POST /step`
```bash
# Auto-generated actions (AI-controlled)
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:5888/step

# Mixed control (user controls agent 0, AI controls others)
curl -X POST -H "Content-Type: application/json" -d '{
  "user_actions": {"0": 42}
}' http://127.0.0.1:5888/step
```

**Response**:
```json
{
  "actions_taken": [42, 156, 203, 87],
  "next_observation": [/* updated observations */],
  "rewards": [25.3, 18.7, 22.1, 19.5],
  "terminated": false,
  "truncated": false,
  "info": {
    "P1_reward": 25.3,
    "P2_reward": 18.7,
    "P3_reward": 22.1,
    "P4_reward": 19.5,
    "env_score": 85.6,
    "player_resources": {
      "P1": {"money": 120, "reputation": 85},
      "P2": {"money": 95, "reputation": 78},
      "P3": {"money": 110, "reputation": 82},
      "P4": {"money": 88, "reputation": 75}
    }
  },
  "t_env": 1
}
```

### 3. Simulate Full Episode
Runs a complete episode with all agents controlled by AI.

**Endpoint**: `POST /simulate`
```bash
curl -X POST http://127.0.0.1:5888/simulate
```

**Response**:
```json
{
  "episode_records": [
    {
      "t_env": 1,
      "actions": [42, 156, 203, 87],
      "observation": [/* step observations */],
      "rewards": [25.3, 18.7, 22.1, 19.5],
      "terminated": false,
      "truncated": false,
      "info": {/* step info */}
    },
    /* ... more steps until episode ends ... */
  ],
  "message": "Episode simulation completed."
}
```

## 🧠 Model Details

### Training Algorithm: MAPPO
- **Algorithm**: Multi-Agent Proximal Policy Optimization
- **Network**: RNN-based agents with 128 hidden dimensions
- **Training**: Episode-based runner with batch size 1
- **Optimization**: Adam optimizer with learning rate 0.0003

### Model Files
Located in `inference_server/saved_models_scale_up/`:
- `agent.th`: Trained agent neural network weights
- `agent_opt.th`: Agent optimizer state
- `critic.th`: Trained critic neural network weights  
- `critic_opt.th`: Critic optimizer state

### Configuration Parameters
```python
args = Namespace(
    action_selector="soft_policies",
    runner="episode",
    lr=0.0003,
    hidden_dim=128,
    learner="ppo_learner",
    entropy_coef=0.001,
    use_rnn=True,
    gamma=0.99,
    agent="rnn"
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'pettingzoo'
```bash
conda activate epymarl
pip install pettingzoo
```

#### 2. Port Already in Use
```bash
# Kill existing servers
pkill -f "inference_server"

# Check port usage
netstat -an | grep 5888

# Restart server
python inference_server/inference_server_scale_up.py
```

#### 3. Model Not Found
Ensure the model files exist:
```bash
ls -la inference_server/saved_models_scale_up/
# Should show: agent.th, agent_opt.th, critic.th, critic_opt.th
```

#### 4. Environment Import Error
Verify Python path:
```bash
export PYTHONPATH=/path/to/6.S890_Course_Project/src:$PYTHONPATH
```

### Debug Mode
Run the server with debug output:
```bash
conda activate epymarl
python inference_server/inference_server_scale_up.py
```

Check server logs for detailed information about initialization and request processing.

## 🔄 Integration with Frontend

### Next.js Frontend Configuration
Update your frontend to connect to the scale-up server:

```javascript
const API_BASE_URL = 'http://127.0.0.1:5888';

// Reset environment
const resetEnvironment = async () => {
  const response = await fetch(`${API_BASE_URL}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  return response.json();
};

// Send actions
const sendActions = async (userActions = {}) => {
  const response = await fetch(`${API_BASE_URL}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_actions: userActions })
  });
  return response.json();
};
```

### Agent Mapping
- **Agent 0 (P1)**: First player
- **Agent 1 (P2)**: Second player  
- **Agent 2 (P3)**: Third player
- **Agent 3 (P4)**: Fourth player

### Action Space
The scale-up environment has 321 possible actions per agent, including:
- Building placements (various types and locations)
- Resource management actions
- Strategic coordination moves

## 📊 Performance Monitoring

### Server Health Check
```bash
# Check if server is responding
curl -I http://127.0.0.1:5888/reset

# Monitor server process
ps aux | grep inference_server_scale_up.py
```

### Resource Usage
```bash
# Monitor CPU and memory usage
top -p $(pgrep -f inference_server_scale_up.py)
```

## 🎯 Use Cases

### Research Applications
- **Multi-Agent Coordination**: Study complex agent interactions in larger environments
- **Scalability Testing**: Evaluate algorithm performance with increased complexity
- **Strategic Planning**: Analyze long-term strategic decision making

### Development Testing
- **Frontend Integration**: Test UI with more complex multi-agent scenarios
- **API Performance**: Evaluate server response times with larger state spaces
- **Model Evaluation**: Compare performance across different environment scales

## 📝 Version Information

- **Environment**: SimCity Scale-Up v2.0
- **Algorithm**: MAPPO
- **Framework**: EPyMARL
- **Python**: 3.8+
- **PyTorch**: 2.4.1+

## 🤝 Contributing

To contribute improvements to the scale-up server:

1. Ensure all tests pass with the scale-up environment
2. Update documentation for any API changes
3. Maintain backward compatibility where possible
4. Test with both individual and batch operations

---

**Last Updated**: June 2025  
**Maintainer**: 6.S890 Course Project Team 