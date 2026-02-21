# SHWS-ASM 技术实现文档

## 1. 项目目标与复现边界

本项目面向夏克-哈特曼波前传感器（SHWS）动态范围软件扩展，核心目标是在统一仿真与统一评价口径下，复现 ASM（Adaptive Spot Matching）相对基线方法的动态范围优势，并得到可复核的参数化结论。项目只做软件算法与仿真链路实现，不调整光学硬件结构。

本项目将“基本与论文效果相同”定义为以下三条同时满足：

1. 在统一测试协议下，ASM 相对基线方法的动态范围增益 `range_gain >= 14.0`。
2. 在缺斑场景（10\%、30\%、50\%）下，ASM 重构成功率显著高于基线，且在 30\% 缺斑时仍可保持稳定重构。
3. 在动态范围扩展的同时，ASM 的重构误差满足预设精度阈值，不允许出现仅提升范围但重构失真不可用的结果。

不在本项目范围内的内容：

1. 深度学习端到端反演训练与部署。
2. 新硬件设计、实际平台联调、闭环控制实机验证。
3. 论文全部图表逐点数值重建。

## 2. Conda 环境与依赖管理

### 2.1 环境策略

使用 `conda` 管理环境，要求环境、依赖和随机种子可追溯。环境仅允许一个主执行环境，禁止混用系统 Python。

### 2.2 创建与激活

```bash
conda create -n shws-asm python=3.10 -y
conda activate shws-asm
```

### 2.3 依赖安装

```bash
conda install -y numpy scipy pandas matplotlib pyyaml tqdm
conda install -y -c conda-forge scikit-image
```

如需最小依赖运行当前量化脚本，可仅保留 `python` 标准库；但完整实验与可视化阶段按本节依赖执行。

### 2.4 环境导出与复现

```bash
conda env export --no-builds > environment.yml
conda env create -f environment.yml
```

### 2.5 随机性控制

所有实验固定随机种子，至少控制以下源：

1. `numpy.random.seed(seed)`
2. Python `random.seed(seed)`
3. PSO 初始化粒子位置与速度
4. 缺斑采样索引

推荐默认种子：`20260210`。

## 3. 工程目录与模块职责

建议工程目录如下：

```text
/Users/lilin/Desktop/小雨毕设
├── configs/
│   ├── base.yaml
│   ├── exp_dynamic_range.yaml
│   ├── exp_missing_spot.yaml
│   └── exp_param_scan.yaml
├── src/
│   ├── sim/
│   │   ├── wavefront.py
│   │   ├── lenslet.py
│   │   ├── imaging.py
│   │   └── noise.py
│   ├── recon/
│   │   ├── baseline_extrap_nn.py
│   │   ├── asm_objective.py
│   │   ├── asm_pso.py
│   │   └── zernike.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── protocol.py
│   │   └── statistics.py
│   └── cli/
│       ├── simulate.py
│       ├── run_baseline.py
│       ├── run_asm.py
│       ├── evaluate.py
│       └── plot.py
├── outputs/
│   ├── raw/
│   ├── tables/
│   └── figures/
├── shws_dynamic_range_model.py
├── shws_dynamic_range_results.csv
└── 开题报告/SHWS_ASM_技术实现文档.md
```

说明：现有 `shws_dynamic_range_model.py` 作为参数关系快速解析模型保留，不替代完整 ASM 仿真与重构链路。

## 4. SHWS 前向仿真模型

### 4.1 模型输入

输入参数由配置文件统一管理，至少包含：

1. 光学参数：`wavelength_nm`, `pitch_um`, `focal_mm`, `aperture_fill_factor`
2. 探测器参数：`pixel_um`, `sensor_width_px`, `sensor_height_px`
3. 波前参数：`zernike_order`, `coeff_vector`, `amplitude_grid`
4. 噪声参数：`read_noise_sigma`, `background_level`, `centroid_noise_px`

### 4.2 前向链路

前向模型按如下顺序执行：

1. 生成波前：由 Zernike 系数生成连续波前 `W(x, y)`。
2. 子孔径采样：在每个微透镜子孔径上估计局部斜率 `(theta_x, theta_y)`。
3. 位移映射：根据 `Δx = f * theta_x`, `Δy = f * theta_y` 计算焦斑偏移。
4. 成像与采样：将焦斑投影至探测器并离散采样。
5. 噪声注入：叠加读出噪声、背景噪声和定位误差。
6. 质心提取：输出观测焦斑点集 `G = {(x_i, y_i)}`。

### 4.3 物理一致性约束

1. 子孔径边界约束：若焦斑越界，允许进入“缺斑/串扰候选”集合。
2. 传感器边界约束：超出探测器范围记为缺失点。
3. 光斑有效性约束：低信噪比点可被标记为不可靠。

## 5. 基线算法实现规范（外推 + 最近邻）

### 5.1 输入输出

输入：观测焦斑点集 `G`、标称参考点集 `R0`、外推初值。输出：匹配点集 `M_baseline`、重构系数 `c_baseline`、误差指标。

### 5.2 算法流程

1. 从中心区域高置信点建立初始匹配。
2. 根据已匹配邻域做局部外推，预测下一子孔径焦斑位置。
3. 在门限半径 `tau_nn` 内做最近邻匹配。
4. 迭代传播直至全孔径或无可用候选。
5. 由匹配结果重构 Zernike 系数并计算误差。

### 5.3 失效判据

基线方法出现以下任一情况视为失效：

1. 连续 `k_fail` 个子孔径无可用匹配。
2. 匹配冲突率超过阈值 `conflict_ratio_max`。
3. 重构 RMSE 超过阈值 `rmse_max`。

## 6. ASM 算法实现规范

### 6.1 参数化与前向映射

令待估参数为 Zernike 系数向量 `c ∈ R^d`，通过前向模型得到估计点集 `E(c)`。观测点集为 `G`。

### 6.2 目标函数

采用集合级匹配目标，定义：

`J(c) = d_H(E(c), G) + λ_dup * P_dup(c) + λ_out * P_out(c) + λ_reg * ||c||_2^2`

其中：

1. `d_H` 为双向 Hausdorff 距离。
2. `P_dup` 为多对一退化匹配惩罚。
3. `P_out` 为越界或无效点惩罚。
4. `λ_dup`, `λ_out`, `λ_reg` 为权重超参数。

### 6.3 PSO 搜索

PSO 默认参数建议：

1. 粒子数 `n_particles = 100`
2. 最大迭代 `n_iter = 200`
3. 惯性权重 `w: 0.9 -> 0.4` 线性衰减
4. 学习因子 `c1 = 1.8`, `c2 = 1.8`
5. 参数边界按像差先验设定 `c_min`, `c_max`

### 6.4 收敛与终止

满足以下任一条件停止：

1. 全局最优目标值在 `patience` 轮内改进小于 `eps_obj`。
2. 达到最大迭代次数。
3. 结果满足精度阈值且稳定性通过。

## 7. 实验设计

### 7.1 总体原则

所有对比必须共享以下条件：同一波前样本集合、同一噪声模型、同一重构评价函数、同一统计口径。禁止不同算法使用不同数据分布。

### 7.2 动态范围极限实验

1. 逐步增大输入波前幅值（建议按 `PV` 或斜率幅值网格扫描）。
2. 每个幅值点重复多次随机采样。
3. 记录基线与 ASM 的成功重构上限。
4. 计算 `range_gain = DR_ASM / DR_baseline`。

### 7.3 缺斑鲁棒性实验

1. 缺斑率设为 `10%`, `30%`, `50%`。
2. 对每个缺斑率重复随机删除操作。
3. 统计成功率、RMSE 和失败模式。

### 7.4 参数扫描实验

围绕微透镜参数进行二维或多维扫描，至少覆盖：

1. `focal_mm`
2. `pitch_um`
3. 必要时加入 `pixel_um` 与噪声水平 `σ`

输出参数关系模型：`R = F(p, f, d, s, σ)`，其中 `R` 为误差约束下可用动态范围。

## 8. 评价指标与统计方法

### 8.1 指标定义

1. 动态范围 `DR`：在约束 `RMSE <= rmse_max` 且成功率 `>= sr_min` 条件下的最大可用输入幅值。
2. 动态范围增益 `range_gain = DR_ASM / DR_baseline`。
3. 重构误差：`RMSE(phase)`。
4. 成功率：`success_rate = N_success / N_total`。
5. 运行开销：单样本耗时与总实验耗时。

### 8.2 推荐阈值

1. `rmse_max = 0.15 λ`
2. `sr_min = 0.95`（极限缺斑实验可放宽至 `0.85`，需单独标注）
3. 核心结论验收：`range_gain >= 14.0`

### 8.3 统计策略

1. 每个实验点至少 `N=20` 次独立重复。
2. 输出均值、标准差与 `95%` 置信区间。
3. 对关键结论点提供原始结果表与可追溯随机种子。

## 9. 结果对齐与验收标准

### 9.1 验收分级

1. 通过：满足核心结论 `range_gain >= 14.0`，并完成缺斑鲁棒性结论。
2. 临界通过：`range_gain` 接近阈值（例如 `13.0~14.0`）但趋势一致，需补充误差分析与参数解释。
3. 不通过：`range_gain < 13.0` 或缺斑实验无法体现 ASM 优势。

### 9.2 结果一致性检查

1. 检查是否存在“仅范围提升、精度明显恶化”的伪提升。
2. 检查与当前解析模型趋势是否一致（大 `pitch`、短 `focal` 倾向更大范围）。
3. 检查不同随机种子下结论稳定性。

### 9.3 与论文结论对齐说明

本项目按“核心结论复现”执行，不要求逐图逐点复刻论文数值。若出现偏差，优先从噪声模型、目标函数权重和参数边界解释差异来源。

## 10. 风险与排错手册

### 10.1 优化不收敛

可能原因：PSO 参数不当、搜索边界过宽、目标函数尺度不平衡。

处理顺序：

1. 缩小参数边界。
2. 对目标项做归一化。
3. 提高粒子数或迭代数。
4. 调整 `w`, `c1`, `c2`。

### 10.2 退化匹配严重

可能原因：惩罚项过弱、缺斑过多、噪声过高。

处理顺序：

1. 提高 `λ_dup`。
2. 增加有效点置信筛选。
3. 在高缺斑场景增加稳健先验。

### 10.3 指标异常波动

可能原因：随机种子未固定、样本量不足、实验条件不一致。

处理顺序：

1. 固定随机性。
2. 增加重复次数。
3. 统一配置并记录版本。

### 10.4 参数爆炸与越界

可能原因：约束缺失、目标函数可辨识性不足。

处理顺序：

1. 增强 `λ_reg` 正则。
2. 强制参数边界。
3. 增加先验物理约束。

## 11. 可复现实验清单

### 11.1 一次完整运行顺序

```bash
conda activate shws-asm

# 1) 快速参数趋势校核（现有脚本）
python shws_dynamic_range_model.py --out-csv shws_dynamic_range_results.csv

# 2) 生成统一仿真样本
python -m src.cli.simulate --config configs/base.yaml

# 3) 跑基线方法
python -m src.cli.run_baseline --config configs/exp_dynamic_range.yaml

# 4) 跑 ASM 方法
python -m src.cli.run_asm --config configs/exp_dynamic_range.yaml

# 5) 缺斑鲁棒性实验
python -m src.cli.run_baseline --config configs/exp_missing_spot.yaml
python -m src.cli.run_asm --config configs/exp_missing_spot.yaml

# 6) 评价与汇总
python -m src.cli.evaluate --config configs/base.yaml
python -m src.cli.plot --config configs/base.yaml
```

### 11.2 产出文件要求

1. `outputs/raw/*.csv`：逐样本原始结果。
2. `outputs/tables/summary_metrics.csv`：核心指标汇总。
3. `outputs/figures/*.png`：动态范围曲线、缺斑鲁棒性曲线、参数热图。
4. `outputs/tables/repro_manifest.json`：版本、种子、配置哈希、环境信息。

### 11.3 最终交付判定

满足以下条件即可判定交付完成：

1. 文档、代码、配置可独立复现实验。
2. 核心结论满足 `range_gain >= 14.0`。
3. 缺斑鲁棒性对比结论清晰。
4. 结果可追溯，可解释，与开题报告技术路线一致。

## 附录 A：配置接口约定（YAML）

```yaml
experiment:
  name: "exp_dynamic_range"
  seed: 20260210

optics:
  wavelength_nm: 633.0
  pitch_um: 150.0
  focal_mm: 6.0
  fill_factor: 0.95

sensor:
  pixel_um: 10.0
  width_px: 2048
  height_px: 2048

noise:
  read_sigma: 0.01
  background: 0.0
  centroid_noise_px: 0.05

zernike:
  order: 15
  coeff_bound: 1.0

asm:
  lambda_dup: 1.0
  lambda_out: 1.0
  lambda_reg: 1e-3
  pso_particles: 100
  pso_iter: 200
  w_start: 0.9
  w_end: 0.4
  c1: 1.8
  c2: 1.8

evaluation:
  rmse_max_lambda: 0.15
  success_rate_min: 0.95
  required_range_gain: 14.0
```

## 附录 B：数据接口约定（CSV）

最小字段集合：

1. `sample_id`
2. `method`（`baseline` 或 `asm`）
3. `pv_level`
4. `missing_ratio`
5. `rmse`
6. `success`
7. `runtime_ms`
8. `range_gain`
9. `seed`

说明：`range_gain` 在逐样本层可留空，汇总表必须给出。

## 附录 C：现有脚本兼容说明

`shws_dynamic_range_model.py` 当前输出字段包括：

1. `pitch_um`, `focal_mm`, `f_number`
2. `airy_radius_um`, `max_disp_um`
3. `theta_max_mrad`, `theta_max_soft_mrad`
4. `slope_resolution_urad`, `local_opd_pv_waves`
5. `dynamic_to_resolution`, `valid`

该脚本可用于项目早期参数区间筛选和趋势验证，不替代 ASM 完整重构实验。
