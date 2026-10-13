# M20-Piper 轮足移动操作训练

本适配基于 DeepRobotics 的 M20-Piper USD，加载时将底盘关节坐标系和限位对齐官方 M20，无需改用 X30。支持平地移动底座与六轴 Piper 末端位姿的联合训练，不包含抓取物体、夹爪策略、复杂地形、MuJoCo 部署或实机控制。

## 复用当前环境

本机使用已有 `rl-training`：Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128、RSL-RL 5.0.1。无需创建环境、重装 Isaac Lab 或升级 PyTorch。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-training
cd ~/LeggedManip_Lab
unset DEEP_ROBOTICS_MODEL_DIR
python -m pip install -e source/LeggedManip_Lab --no-deps --no-build-isolation
```

本机已完成项目的 editable 安装，后续无需重复安装。必须先 `conda activate`，以执行当前环境的动态库初始化钩子。

本项目已内置 `deep_robotics_model/M20` 和 `deep_robotics_model/M20_Piper` 的完整模型资源（USD、URDF、MJCF 及网格），普通 `git clone` 即可取得，不需要额外克隆模型仓库、初始化子模块或安装 `rl_training` Python 包。模型快照版本及原始许可证见 [`deep_robotics_model/README.md`](../deep_robotics_model/README.md)。

环境变量未设置时，默认读取项目根目录下的 `deep_robotics_model/M20_Piper/usd/M20_Piper.usd`，并以 `deep_robotics_model/M20/usd/M20.usd` 为方向参考。上面的 `unset` 用来清除之前指向外部模型的设置。若确实需要其他模型副本，仍可设置 `DEEP_ROBOTICS_MODEL_DIR=/absolute/path/deep_robotics_model` 覆盖默认目录；该目录必须同时包含 M20 和 M20_Piper 的完整资源。

加载时以 M20 的 16 个底盘关节的轴、两侧局部旋转和角度限位为基准修正 M20-Piper。修正发生在场景 USD 中，且先于多环境克隆；不覆盖内置或外部的官方文件，不改变底盘几何零位、质量、机械臂或夹爪。直接在其他程序中打开原始 `M20_Piper.usd` 不会包含此修正。

训练配置使用本地地面材质和无纹理光源，不依赖远程 Nucleus 地面/天空素材。Isaac Sim 仍需能访问 GPU；沙箱内无法访问 CUDA 不表示宿主机没有 GPU。

## 启动

只查看修正后的模型，不启动训练；窗口暂停在官方 M20 默认腿姿，前腿髋/膝为 `-0.3/+0.6`，后腿为 `+0.3/-0.6` rad：

```bash
python scripts/preview_m20_piper.py
```

```bash
python scripts/list_envs.py --keyword M20
python scripts/check_m20_piper_actions.py --headless --task M20-PIPER-WBC
python scripts/smoke_m20_piper.py --headless --task M20-PIPER-Flat
python scripts/smoke_m20_piper.py --headless --task M20-PIPER-WBC
python scripts/rsl_rl/train.py --task M20-PIPER-Flat --headless --num_envs 16 --max_iterations 2
python scripts/rsl_rl/train.py --task M20-PIPER-WBC --headless --num_envs 32 --max_iterations 10
```

正式训练示例（不是已训练好的策略）：

```bash
python scripts/rsl_rl/train.py --task M20-PIPER-Flat --headless --num_envs 1024 --max_iterations 10000
python scripts/rsl_rl/train.py --task M20-PIPER-WBC --headless --num_envs 1024 --max_iterations 10000
```

两种任务分别写入 `logs/rsl_rl/m20_piper_flat` 和 `logs/rsl_rl/m20_piper_wbc`。根据显存和吞吐调整并行数，不要同时执行两条长训练命令。

回放自己训练的 checkpoint：

```bash
python scripts/rsl_rl/play.py --task M20-PIPER-Flat-Play --num_envs 4 --checkpoint /absolute/path/model_9999.pt
python scripts/rsl_rl/play.py --task M20-PIPER-WBC-Play --headless --num_envs 4 --checkpoint /absolute/path/model_9999.pt --max_steps 200
```

省略 `--headless` 可打开窗口；`--max_steps` 用于有界验证。回放会同时导出 JIT/ONNX。已有 GO2-PIPER checkpoint 的动作/观测及坐标定义不同，不能直接用来控制 M20-Piper。

## 轮足适配

模型具有 24 个活动关节，策略输出 22 维动作：

| 动作索引 | 顺序 | 控制方式 |
| --- | --- | --- |
| 0–11 | `fl, fr, hl, hr`，各自 `hipx, hipy, knee` | 默认角度 + 位置增量；hipx 缩放 0.125，其余 0.25 rad |
| 12–17 | `joint1` … `joint6` | 默认角度 + 0.25 × 动作，单位 rad |
| 18–21 | `fl, fr, hl, hr` 的 `wheel_joint` | 5 × 动作，单位 rad/s；零位置刚度 |

`joint7/8` 是两侧夹指，由独立隐式 PD 保持闭合，不作为策略动作。RSL-RL 包装器将原始动作裁剪至 ±5。

对齐官方 M20 后，`hipx` 的实际轴为 `-X`，`hipy/knee/wheel` 为 `-Y`。轮速动作仍保持官方的正比例映射 `5 × action`；直立姿态下向前滚动应给负轮速。修正前的短测 checkpoint 不能直接继续使用，需重新训练。

策略输出的是关节动作，不是底盘速度命令。正的前进速度命令仍表示机身沿 `+X` 行驶，PPO 通过速度跟踪奖励学习负轮速；不能为了让轮速数值为正而再次反转底盘命令或关节观测。

- 轮角可无限累积，因此位置观测只含腿和手臂的 18 个关节；速度/力矩使用 22 个受控关节。策略观测 234 维（三帧历史），critic 114 维。
- 接触检测改为四个 `*_wheel`，机身改为 `base_link`。不再使用足底滑动、腾空时间、足式镜像等奖励；轮子不参加位置限位和关节加速度惩罚。
- 初始机身高度 0.58 m，仿真 200 Hz、控制 50 Hz、episode 20 s。初始底座前后速度/偏航范围 ±0.2，课程上限分别 ±1 m/s、±0.8 rad/s；横向速度命令固定为零，先学习滚动而非侧向跨步。
- 保留物理、质量、惯量、质心和执行器增益随机化，以及机身触地/倾倒终止。未训练的固定姿态开环动作仍可能失稳，不能用零动作长时间站立作为已学会平衡的证明。

## 末端坐标与任务

官方模型没有 GO2 配置中的 `link0` / `end_effector`。这里采用 `arm_base_link` 为机械臂参考系，`gripper_base` 为跟踪刚体（与末端法兰重合），**不是两指之间的抓取 TCP**。

- `M20-PIPER-Flat`：位置和姿态目标都相对 `arm_base_link`。
- `M20-PIPER-WBC`：XY 相对 `arm_base_link`，Z 为世界绝对高度；姿态仍相对 `arm_base_link`。
- 默认手臂角度为 `[0, 1.0, -1.2, 0, 0.2, 0]` rad。名义法兰相对位置约 `[0.205, 0, 0.481]` m；Piper 法兰坐标轴使初始目标 pitch 约为 1.4836 rad，而不是照搬 GO2 的零姿态。
- 初始目标在默认姿态附近取样，随后按跟踪奖励扩展课程。Flat 的 Z 上限范围是 0.20–0.50 m，WBC 是世界高度 0.80–1.17 m；这些是仿真训练初值，并非经过完整逆运动学/碰撞验证的工作空间。

## 参数来源与假设

参考 [官方模型仓库](https://github.com/DeepRoboticsLab/deep_robotics_model) 中 `M20_Piper/usd/M20_Piper.usd`、`M20_Piper/urdf/M20_Piper.urdf` 和 `M20/usd/M20.usd`；项目内置模型固定为已验证的快照 `549eb9ac08ab0674ad388202f0a276a1b140833b`。加载 USD 后对齐底盘关节坐标系和限位，不修改原始模型文件，也不重新转换 URDF。

底座电机、初始腿姿和动作尺度参考 [官方 RL Training](https://github.com/DeepRoboticsLab/rl_training) 中：

- `source/rl_training/rl_training/assets/deeprobotics.py` 的 `DEEPROBOTICS_M20_CFG`。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/rough_env_cfg.py`。
- 本次对齐时核对的官方训练快照：`3d05d3b97f7a2390f003547f23a02b3b42eea195`。

| 参数 | 腿 | 轮 |
| --- | --- | --- |
| 力矩限幅 | 76.4 Nm | 21.6 Nm |
| 速度限幅 | 22.4 rad/s | 79.3 rad/s |
| Kp / Kd | 80 / 2 | 0 / 0.6 |
| armature | 0 | 0.00243216 |
| 延迟 | 0–1 个物理步 | 0–1 个物理步 |

**Piper 电机没有在上述 locomotion 配置中提供厂商确认的控制参数。** 当前手臂增益参考本项目已有 GO2-Piper 配置，力矩限幅使用其注释中的仿真候选值 `[20, 20, 15, 7, 5, 5]` Nm，Kp `[50, 50, 80, 30, 30, 20]`，Kd `[3, 2, 3, 3, 2.5, 1]`，延迟 0–4 个物理步。速度限幅依据模型的 5 rad/s（joint1–5）、3 rad/s（joint6）。这些不是已确认的实机额定参数；部署前必须重新核实电机约束、摩擦、延迟、负载和碰撞安全性。

## 验证边界

2026-10-12 对齐底盘方向后的验证：

- 无窗口运动学校验覆盖零位、默认位及 16 个关节单独正向扰动。M20-Piper 与同角度 URDF 的最大位置误差约 `5.4e-7 m`，方向误差约 `3.2e-7 rad`。
- Flat、WBC 各通过 4 环境 × 200 步检查；所有环境的 16 个底盘关节轴、局部旋转、限位与官方 M20 一致，机械臂和夹爪的关节定义保持原样。
- `-5 rad/s` 短时轮速测试中，两种任务的机身平均前向速度约 `0.596 m/s`；这不是稳态速度跟踪精度。两种任务的动作/观测、有限数值及超时复位检查通过。
- `check_m20_piper_actions.py` 在 44 个固定底座、关闭重力/扰动的诊断环境中，通过训练使用的 RSL-RL 包装器逐一测试 22 维动作的正负输入。位置目标、轮速目标、夹爪隔离、实际关节响应、观测符号和 ±5 动作裁剪全部通过；执行器延迟保留，先用零动作预热。另检查了真实未训练网络的 `234 → 22` 输出到控制目标的完整映射。此诊断不改正式训练配置，也不代表策略已学会任务。

以下是本机 2026-10-12 在 NVIDIA RTX5880-Ada-48Q 上、底盘方向修正前的历史验证结果，不代表修正后的策略表现；旧 checkpoint 仅供历史记录，不能用于修正后的模型：

| 项目 | 结果 |
| --- | --- |
| 任务注册 | Flat、WBC 及各自 Play，共 4 个任务可列出 |
| Flat / WBC 仿真 | 各 4 环境 × 200 步；数值、关节映射、末端奖励及超时复位检查通过 |
| 轮速方向 | 5 rad/s 短时开环轮速指令下，机身平均前向速度约 0.145 m/s；不是速度跟踪精度测试 |
| Flat PPO | 16 环境 × 2 迭代，共 768 transitions，保存 `model_1.pt` |
| WBC PPO | 32 环境 × 10 迭代，共 7680 transitions，保存 `model_9.pt` |
| Checkpoint 回放 | 两种 Play 任务各 4 环境 × 100 步，正常退出 |
| 导出 | 两种策略均生成 JIT / ONNX；JIT 的 234 → 22 维有限输出和 ONNX 结构校验通过 |

验证产物保存在 `logs/rsl_rl/m20_piper_flat/2026-10-12_14-15-56_m20_smoke/` 和 `logs/rsl_rl/m20_piper_wbc/2026-10-12_14-21-08_m20_smoke/`。这里只保留了短测 checkpoint，不是收敛策略，没有启动后台长训练。

`smoke_m20_piper.py` 检查所有环境的底盘关节方向/限位与官方 M20 一致、关节覆盖与动作映射、观测维度、轮速方向、精确目标时的位置/姿态奖励、200 步有限数值以及 episode 超时后的轮角清零和自动复位。前 20 步零动作、接着 40 步给轮子 -5 rad/s，之后返回零动作；报告开环跌倒数量，不把它当作训练收敛测试。

短程 PPO 只用于确认采样、反向传播和 checkpoint 保存链路。完整任务成功率、跟踪精度、碰撞安全和长期稳定性需要正式训练后另行评估。本适配没有启用自碰撞，也没有提供实机部署控制器。

M20 的 agent 配置使用 RSL-RL 5 原生 actor/critic 字典，避开本机 Isaac Lab 2.3.2 缺少 `RslRlMLPModelCfg` 的问题。兼容入口不修改已安装的 Isaac Lab/RSL-RL 包；其他机器人原有配置是否支持旧 Isaac Lab，不在本适配的验证范围内。
