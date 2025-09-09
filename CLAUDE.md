# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is an Extended Python Multi-Agent Reinforcement Learning (EPyMARL) framework specifically adapted for SimCity environments. The codebase includes both a standard 4x4 grid SimCity environment and a scaled-up 8x8 grid version with more agents and complexity.

### Core Components

- **`src/main.py`**: Primary entry point using Sacred framework for experiment management
- **`src/run.py`**: Main execution logic that orchestrates training/evaluation
- **`src/envs/`**: Environment definitions including SimCity variants
  - `simcity/`: Original 4x4 grid environment (3 agents)
  - `simcity_scale_up/`: Enhanced 8x8 grid environment (4 agents)
  - `simcity_scale_up_koto/`: Experimental variant
- **`src/learners/`**: MARL algorithm implementations (MAPPO, QMIX, PAC, etc.)
- **`src/controllers/`**: Agent control logic and multi-agent coordination
- **`src/modules/`**: Neural network architectures (agents, critics, mixers)
- **`inference_server/`**: Flask-based API servers for trained model inference

### Environment Specifications

**SimCity Standard**: 4x4 grid, 3 agents, 49 actions per agent, 82-dim observations
**SimCity Scale-Up**: 8x8 grid, 4 agents, 321 actions per agent, 514-dim observations

### Algorithm Support

- **Common reward algorithms**: QMIX, VDN, COMA, QTRAN
- **Individual reward algorithms**: MAPPO, IA2C, IPPO, MAA2C, PAC
- **Individual vs common rewards**: Configurable via `common_reward=True/False`

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n epymarl python=3.8
conda activate epymarl

# Install dependencies
pip install einops
pip install -r requirements.txt
pip install -r env_requirements.txt
pip install -r pac_requirements.txt
```

### Training Commands

#### Basic Training
```bash
# Standard SimCity with QMIX
python src/main.py --config=qmix --env-config=simcity

# Scale-up environment with MAPPO
python src/main.py --config=mappo --env-config=simcity_scale_up
```

#### Advanced Training Options
```bash
# Individual rewards (for compatible algorithms)
python src/main.py --config=mappo --env-config=simcity_scale_up with common_reward=False

# Custom training parameters
python src/main.py --config=mappo --env-config=simcity_scale_up \
    with env_args.time_limit=100 \
    t_max=2000000 \
    save_model=True \
    save_model_interval=10000
```

#### Pre-configured Experiment Scripts
```bash
# Scale-up training with MAPPO
./train_scale_up_mappo.sh

# Batch experiments
./simcity_experiments_all.sh
./simcity_large_experiments_all.sh
```

### Inference Server

#### Standard Environment Server
```bash
python inference_server/server.py
# Runs on port 5000
```

#### Scale-up Environment Server
```bash
python inference_server/inference_server_scale_up.py  
# Runs on port 5888
```

#### API Endpoints
- `POST /reset`: Reset environment
- `POST /step`: Execute actions (supports mixed human/AI control)
- `POST /simulate`: Run full AI episode

### Analysis and Visualization

#### Plot Training Results
```bash
python plot_results.py --path results
```

#### Model Evaluation
```bash
# Load and evaluate saved model
python src/main.py --config=mappo --env-config=simcity_scale_up \
    with checkpoint_path=results/models/[model_dir] \
    evaluate=True
```

### Configuration System

- **Algorithm configs**: `src/config/algs/*.yaml` (mappo.yaml, qmix.yaml, etc.)
- **Environment configs**: `src/config/envs/*.yaml` (simcity.yaml, simcity_scale_up.yaml, etc.)
- **Default parameters**: `src/config/default.yaml`

### Key Configuration Parameters

- `t_max`: Total training timesteps
- `test_interval`: Evaluation frequency
- `save_model_interval`: Model checkpoint frequency  
- `use_cuda`: GPU acceleration
- `common_reward`: Individual vs shared rewards
- `batch_size`: Training batch size
- `lr`: Learning rate

## Testing

No specific test framework is configured. Environment validation can be done via:

```bash
# Test environment loading
python -c "import sys; sys.path.insert(0, 'src'); from envs.simcity_scale_up_wrapper import SimCityScaleUpWrapper; print('Environment OK')"

# Simple test runs
python src/envs/simcity_scale_up/simple_test.py
```