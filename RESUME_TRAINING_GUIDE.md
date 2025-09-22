# Enhanced Training Resume Guide

## 概述

增强版训练恢复脚本能够**完全继承**原始训练的所有设置，包括种子、学习率、批量大小等所有超参数，确保训练的连续性和一致性。

## 主要改进

### 相比原始 `resume_training.py` 的优势：

1. **完整配置继承**: 自动从 Sacred 配置文件读取所有原始训练参数
2. **种子保持**: 确保使用相同的随机种子，保证结果可重现
3. **参数验证**: 显示所有继承的关键参数供确认
4. **智能匹配**: 通过种子自动匹配对应的 Sacred 配置文件

### 原始脚本的局限性：
- 只恢复模型权重，不继承训练超参数
- 使用默认或命令行指定的参数，可能与原始训练不一致
- 无法保证训练的真正连续性

## 文件说明

- `resume_training_enhanced.py`: 核心增强版恢复脚本
- `resume_koto_training.sh`: 便捷的包装脚本
- `train_scale_up_koto_mappo.sh`: 原始训练脚本（参考）

## 使用方法

### 1. 交互式模式（推荐）

```bash
# 显示所有可用的训练运行，包括种子和配置信息
python resume_training_enhanced.py

# 或使用便捷脚本
./resume_koto_training.sh
```

### 2. 自动化模式

```bash
# 自动恢复运行索引 0 的最新检查点
python resume_training_enhanced.py --auto --run-index 0

# 恢复特定步数
python resume_training_enhanced.py --auto --run-index 0 --step 50000

# 使用自定义 t_max
python resume_training_enhanced.py --auto --run-index 0 --t-max 200000000

# 使用便捷脚本
./resume_koto_training.sh 0           # 运行索引 0
./resume_koto_training.sh 0 50000     # 运行索引 0，步数 50000
./resume_koto_training.sh 0 50000 200000000  # 自定义 t_max
```

### 3. 仅查看命令（不执行）

```bash
# 查看将要执行的完整命令
python resume_training_enhanced.py --dry-run --auto --run-index 0
```

## 输出示例

```
Available runs:
Idx | Env  | Ckpts | Latest  | Seed       | Config | Origin   | Directory
----+------+-------+---------+------------+--------+----------+----------------------------------------------
  0 | koto |    33 | 2663921 | 996602412  |      ✓ | models   | mappo_seed996602412_simcity_scale_up_koto_...

Resume configuration:
  Run directory : /path/to/models/mappo_seed996602412_simcity_scale_up_koto_...
  Checkpoint    : 2663921
  t_max target  : 100000000
  Sacred config : /path/to/sacred/mappo/simcity_scale_up_koto/1/config.json
  Original seed : 996602412
  Learning rate : 0.0003
  Batch size    : 192
```

## 继承的参数

脚本会自动继承以下所有原始训练参数：

### 核心训练参数
- `seed`: 随机种子
- `lr`: 学习率
- `batch_size`: 批量大小
- `batch_size_run`: 运行批量大小
- `epochs`: 训练轮数

### 环境配置
- `env_args.time_limit`: 时间限制
- `env_args.grid_x`, `env_args.grid_y`: 网格大小

### 算法参数
- `entropy_coef`: 熵系数
- `eps_clip`: PPO 裁剪参数
- `gamma`: 折扣因子
- `grad_norm_clip`: 梯度裁剪

### 网络配置
- `hidden_dim`: 隐藏层维度
- `use_rnn`: 是否使用 RNN

### 日志和保存
- `log_interval`: 日志间隔
- `save_model_interval`: 模型保存间隔
- `test_interval`: 测试间隔

## 重要说明

1. **Sacred 配置匹配**: 脚本通过种子自动匹配对应的 Sacred 配置文件
2. **参数覆盖**: 某些参数（如 checkpoint_path, load_step, t_max）会被恰当覆盖
3. **兼容性**: 与原始 `train_scale_up_koto_mappo.sh` 完全兼容
4. **验证**: 训练开始前会验证检查点文件的完整性

## 故障排除

### 找不到 Sacred 配置
如果找不到 Sacred 配置文件，脚本会降级为基本恢复模式：
```
Warning: No Sacred config found for run ...
Falling back to basic resume without full config inheritance.
```

### 检查点验证失败
如果检查点文件缺失或损坏，脚本会显示错误并列出可用步数：
```
Validation failed: Checkpoint ... missing files: agent.th, critic.th
Available steps: [10000, 20000, 30000, ...]
```

## 与原始训练脚本的对应关系

增强版恢复脚本会完全继承 `train_scale_up_koto_mappo.sh` 中的所有设置：

| 原始脚本参数 | 继承状态 | 说明 |
|-------------|---------|------|
| `env_args.time_limit=150` | ✅ 自动继承 | 从 Sacred 配置读取 |
| `t_max=100000000` | 🔄 可覆盖 | 可通过 --t-max 指定新值 |
| `batch_size=192` | ✅ 自动继承 | 保持原始批量大小 |
| `save_model_interval=10000` | ✅ 自动继承 | 保持原始保存间隔 |
| 所有其他参数 | ✅ 自动继承 | 完全保持一致 |

这确保了训练能够真正无缝继续，就像从未中断过一样。