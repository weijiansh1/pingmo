# 常规滚转外围与品模中心控制器：集成协议 v1

本次只修改中心控制器之外的接口与外围回路。研究主线仍是基于飞行品模的飞控大模型：随机采样 Gθ → 品模闭环交互训练逐对象 TD3 教师 → 条件学生蒸馏 → DHC-6 小样本权重微调。未重新训练或改变任何既有权重。

## 控制边界

- 输入：坡度角 φ_c [rad] 或机体滚转率 p_c [rad/s]，二选一；估计状态含时间戳、φ、p，并可提供俯仰角、q、r。
- 外环：最短坡度角误差 / 2 s；扣除 `tan(pitch)*(q*sin(phi)+r*cos(phi))` 后得到机体 p 给定；坡度角指令限 ±30°、p 给定限 ±30°/s。
- 中心：保留已训练网络、35 维特征、8 维归一化 θ、二阶期望响应、误差积分与动作历史；20 ms 固定时钟。
- 输出：副翼偏角请求，DHC-6 标定 `delta_a_req = -10 deg * a_req`；位置限 ±10°、速率限 40°/s（0.8°/20 ms）。这些范围源自旧实验的 ±22 N / 88 N/s 映射，是实验假设，不是已核实的 DHC-6 机械规格。
- 理想位置跟随；没有增加未辨识的伺服滞后。旧虚拟力坐标仅保留在中心特征适配器内。
- 姿态/角速度是状态估计接口，并未实现或接入 IMU/EKF。近 90° 俯仰的欧拉角奇点被拒绝；该接口不是适航飞控软件。

## 固定数据与权重

- 两组既有中心权重：源域冻结 W0 与 `dhc6_finetune/unified/b30/weights/full.pt`。
- 两者使用相同 b30 θ；8 个适配条件来自 30 s 前缀，4 个留出条件仍采用既定线性映射。
- 12 个独立评测 ARX 沿用 `unified/shared/eval_models.json`，由 chirp_A10 拟合，并在 prbs_A10 / step3211_A10 验证；不在本次重拟合。
- 输入模型为实际副翼偏角 da_act [deg]，输出 p [deg/s]。外部接口采用 SI 单位，在边界转换。
- 具体权重、θ、模型及旧 manifest 的路径和 SHA-256 见 `manifest.json`；本次运行前后均核对。

## 新坡度角协议

每条 20 s、50 Hz，初始状态为零；12 状态 × 3 指令 × 2 权重 = 72 工况。固定参数先于评测，没有根据结果调整外环增益。

1. 1 s 时，坡度角指令由 0 跳到 +20° 并保持。
2. 1/8/15 s 时，指令依次 0→+15°→−15°→0。
3. 1 s 起，幅值 10°、频率 0.08 Hz 的正弦指令。

降阶状态提供器：俯仰角=q=r=0，梯形积分 p 得到 φ。完整运动学补偿在单元测试中核对，但 SISO ARX 不能验证侧滑、转弯协调或多轴响应。

在 t_k 取状态并产生动作，动作保持到 t_(k+1)，记录 t_(k+1) 的状态；误差对照该区间内零阶保持的原始坡度角指令。报告全 20 s RMSE、峰值误差、最大角度/速率、舵面 TV、幅度/速率限制比例；阶跃与反转另报末 1 s 平均绝对误差。正弦的末段误差不作稳态判据。所有工况保留；72 工况内有限与未超过 30° 不是稳定性或品质等级证明。

## 角速度接口等价回归

旁路坡度角外环，使用既有 6 条回归 + 6 条先导留出速率指令、每条 5 s；12 状态 × 12 指令 × 2 权重 = 288 工况。与 `fine_tune.arx_rollout_numpy` 独立实现比较响应、参考和实际舵面；预设响应容差 0.001 deg/s。逐工况差异见 `rate_parity.json`。

## 品模与品质要求的边界

源域教师在 `SpecialistTrackingEnv` 中实例化参数化滚转品模 `PChannel(record.parameters)` 并与之闭环交互，这就是通过飞行品模训练。期望响应模型 M⋆ 是定义训练误差的另一对象，不等同于 Gθ。现有奖励含跟踪、操纵能量、请求动作跳变；并未直接优化全部 GJB 条款。外围集成使用既有成果，不新增“按品质重新训练”的结论。

## 复现

在 `pingmo/` 工作目录，使用安装项目依赖的 Python：

```powershell
python -B scripts/43_evaluate_roll_autopilot.py --out ../roll_autopilot/new_run
```

默认输出目录已归档，脚本拒绝覆盖。该命令只评测，不训练。绘图脚本 `dhc6_blog_assets/make_roll_autopilot_figures.py` 读取本次 20261019 归档；页面由 `dhc6_html_theory.py` 读取归档生成。

部署加载接口现在同时支持源域与 DHC-6 统一学生 checkpoint：

```python
from src.student.controller import load_student_controller
from src.controllers.roll_autopilot import PingmoRateController, RollAutopilot, RollStateEstimate

policy, metadata = load_student_controller(checkpoint_path, normalized_theta)
loop = RollAutopilot(PingmoRateController(policy.predict))
state = RollStateEstimate(time_s=0.0, roll_rad=0.0, p_rad_s=0.0)
output = loop.step(state, roll_setpoint_rad=0.2)
# output.surface.commanded_rad 是受位置/速率限制的副翼指令。
# 本仿真中 applied_rad 采用理想跟随，不能当作真实舵面测量。
```

网页随附 `code/` 为本次实现副本，`trajectories.zip` 为 72 条 NPZ 轨迹；原始目标 MAT 与原 checkpoint 保留在原路径。
