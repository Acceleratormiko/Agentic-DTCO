# DTCO Tools 工具文档

## 概述

本目录包含 **DTCO (Design-Technology Co-Optimization)** 半导体器件与电路协同优化工具链。该工具链实现了从 TCAD 器件仿真到电路级评估的完整流程，支持 FinFET 和 GAA (Gate-All-Around) 纳米片 FET 器件的自动化优化。

## 目录结构

```
tools/
├── common.py                    # 共享函数和常量
├── tcad_tool.py                 # TCAD 仿真工具
├── compact_model_train_tool.py  # 紧凑模型训练工具
├── hspice_sim_tool.py           # 电路仿真工具
├── plugins/                     # 插件模块
│   ├── circuit_eval/            # 电路评估模块
│   ├── compact_model/           # 紧凑模型模块
│   ├── dtco_controller/         # 评分控制器
│   └── verilog_a/               # Verilog-A 导出模块
└── test/                        # 测试文件
```

## 核心工具

### 1. TCADTool (`tcad_tool.py`)

**功能**：执行 TCAD (Technology Computer-Aided Design) 仿真，生成器件结构并提取电学特性。

**主要能力**：
- 生成 SDE (Structure Device Editor) 脚本
- 运行 SDEVICE Id-Vg 和 C-V 仿真
- 提取器件指标：Ion (开态电流)、Ioff (关态电流)、SS (亚阈值摆幅)
- 支持 N-type、P-type 或 both 模式

**输入参数**：
```python
{
    "device_type": "n-type" | "p-type" | "both",
    "ntype_variables": {  # N-type 器件设计变量
        "Lg": 0.018,         # 栅极长度 (um)
        "Lext": 0.005,       # 扩展长度 (um)
        "Lsd": 0.014,        # 源漏长度 (um)
        "Fh": 0.035,         # 鳍片高度 (um)
        "Fwt": 0.006,        # 鳍片宽度 (um)
        "Fsp": 0.03,         # 鳍片间距 (um)
        "theta": 0.82,       # 倾斜角度 (degree)
        "ToxLK": 5e-4,       # 低K氧化层厚度 (um)
        "ToxHK": 1.13e-3,    # 高K氧化层厚度 (um)
        "SD_Doping": 1e20,   # 源漏掺杂浓度 (cm-3)
        "SDE_Doping": 1e18,  # 源漏扩展掺杂 (cm-3)
        "C_Doping": 5e17,    # 沟道掺杂 (cm-3)
        "ntype_work_function": 4.5  # 功函数 (eV)
    },
    "ptype_variables": {...},  # P-type 器件设计变量
    "ntype_work_function": 4.5,
    "ptype_work_function": 4.7
}
```

**输出**：
```python
{
    "work_dir": "/tmp/tcad_xxx",
    "device_type": "both",
    "ntype_result": {
        "metrics": {"Ion": 1.2e-4, "Ioff": 1e-10, "SS": 68.5, "VDD": 0.75}
    },
    "ptype_result": {...}
}
```

---

### 2. DTCOCompactModelTrainTool (`compact_model_train_tool.py`)

**功能**：基于 TCAD 仿真数据训练神经网络紧凑模型，并导出为 Verilog-A 格式供 SPICE 仿真使用。

**主要能力**：
- 从 TCAD 数据构建训练数据集
- 训练 BSIM-NN 风格的神经网络模型
- 支持 I-V 和 C-V 联合训练
- 物理感知损失函数（包含 gm/gds 导数拟合）
- 导出 Verilog-A 紧凑模型

**神经网络架构**：
```
输入层 (8 neurons): [l, nfin, eot_lk, eot_hk, vg, vd, vs, vb]
    ↓ ISRU 激活
隐藏层1 (10 neurons)
    ↓ ISRU 激活
隐藏层2 (10 neurons)
    ↓ 线性
输出层 (1-2 neurons): [y1=ln(Id/Vds)] 或 [y1, log(Cgg)]
```

**损失函数**：
```
L_total = L_Id + λ_gm * L_gm + λ_gds * L_gds + w_cv * L_cv

其中:
- L_Id: Id 损失（线性域 + 对数域 MSE）
- L_gm: 归一化 dIds/dVgs 损失
- L_gds: 归一化 dIds/dVds 损失
- L_cv: C-V RMS 误差
```

**输入参数**：
```python
{
    "work_dir": "/tmp/tcad_xxx"  # TCAD 仿真输出目录
}
```

**输出**：
```python
{
    "ntype": {
        "model_dir": "compact_model_nn_ntype",
        "verilog_a_path": "integrated_ntype.va"
    },
    "ptype": {...}
}
```

---

### 3. DTCOHspiceSimTool (`hspice_sim_tool.py`)

**功能**：使用训练好的 Verilog-A 紧凑模型运行 HSPICE 电路仿真，评估电路性能并计算综合评分。

**支持电路类型**：
- `inverter` / `inv`: 反相器
- `ring_oscillator` / `ro`: 环形振荡器
- `dff` / `flipflop`: D 触发器
- `sram` / `sram_6t`: 6T SRAM 单元

**评估指标**：
| 指标 | 说明 | 单位 |
|------|------|------|
| tpHL_ps | 高到低传播延迟 | ps |
| tpLH_ps | 低到高传播延迟 | ps |
| F04_delay_ps | FO4 延迟 | ps |
| avg_power_mw | 平均功耗 | mW |
| energy_per_transition_fj | 每次翻转能耗 | fJ |

**输入参数**：
```python
{
    "work_dir": "/tmp/tcad_xxx",
    "circuit_type": "inverter"  # 可选
}
```

**输出**：
```python
{
    "work_dir": "/tmp/tcad_xxx",
    "circuit_metrics": {
        "tpHL_ps": 2.3,
        "tpLH_ps": 2.1,
        "avg_power_mw": 0.45,
        "energy_per_transition_fj": 4.2
    },
    "score": 75.5,
    "score_components": {...}
}
```

---

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DTCO 优化工作流                                │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  设计变量输入     │
    │  (Lg, Fh, Fwt,   │
    │   ToxLK, 掺杂等)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   TCADTool       │
    │  ──────────────  │
    │  1. 生成 SDE 脚本 │
    │  2. 运行器件仿真  │
    │  3. 提取 I-V 曲线 │
    │  4. 提取 C-V 曲线 │
    │  5. 计算 Ion/Ioff │
    └────────┬─────────┘
             │
             ├── device_metrics.json
             ├── result_Sat_*.plt (I-V 数据)
             └── cv_data.json (C-V 数据)
             │
             ▼
    ┌──────────────────────────────┐
    │  DTCOCompactModelTrainTool   │
    │  ────────────────────────────│
    │  1. 读取 TCAD 仿真数据        │
    │  2. 构建训练数据集            │
    │  3. 计算数值导数 (gm, gds)    │
    │  4. 训练神经网络模型          │
    │  5. 导出 Verilog-A 紧凑模型   │
    └────────┬─────────────────────┘
             │
             ├── fet_neural_network_model.pth
             ├── integrated_ntype.va
             └── integrated_ptype.va
             │
             ▼
    ┌──────────────────┐
    │  DTCOHspiceSimTool│
    │  ──────────────  │
    │  1. 加载 Verilog-A│
    │  2. 生成电路网表  │
    │  3. 运行 HSPICE   │
    │  4. 解析仿真结果  │
    │  5. 计算综合评分  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │    评估结果       │
    │  ──────────────  │
    │  • 电路性能指标   │
    │  • 综合评分 (0-100)│
    │  • 优化建议       │
    └──────────────────┘
```

---

## 评分机制

### 评分公式
```
Score = 100 × Σ(weight_i × subscore_i)
```

### 权重配置
| 指标 | 权重 | 说明 |
|------|------|------|
| delay | 0.25 | 传播延迟（越小越好）|
| power | 0.15 | 平均功耗（目标 ≤ 0.5mW）|
| energy | 0.10 | 翻转能耗（目标 ≤ 5fJ）|
| ion | 0.20 | 开态电流（对数刻度，越大越好）|
| ioff | 0.15 | 关态电流（对数刻度，越小越好）|
| ss | 0.15 | 亚阈值摆幅（目标 ≤ 70mV/dec）|

### 目标值
```python
# 延迟目标
DELAY_BEST_PS = 0.5    # 最优延迟
DELAY_WORST_PS = 5.0   # 最差延迟

# Ion 目标
ION_TARGET_A_PER_UM = 1e-4   # 最小可接受值
ION_BEST_A_PER_UM = 5e-3     # 最优值

# Ioff 目标
IOFF_TARGET_A_PER_UM = 1e-8  # 目标上限
IOFF_WORST_A_PER_UM = 1e-5   # 最差值

# SS 目标
SS_TARGET_MV_PER_DEC = 70.0   # 目标值
SS_WORST_MV_PER_DEC = 100.0   # 最差值
```

---

## 设计空间

### FinFET 7nm 设计空间

| 参数 | 最小值 | 最大值 | 单位 | 说明 |
|------|--------|--------|------|------|
| Lg | 0.014 | 0.028 | μm | 栅极长度 |
| Lext | 0.004 | 0.1 | μm | 扩展长度 |
| Lsd | 0.01 | 0.022 | μm | 源漏长度 |
| Fh | 0.03 | 0.06 | μm | 鳍片高度 |
| Fwt | 0.005 | 0.009 | μm | 鳍片宽度 |
| Fsp | 0.025 | 0.045 | μm | 鳍片间距 |
| theta | 0.5 | 4.0 | deg | 倾斜角度 |
| ToxLK | 4e-4 | 1.5e-3 | μm | 低K氧化层厚度 |
| ToxHK | 1e-3 | 1.5e-3 | μm | 高K氧化层厚度 |
| SD_Doping | 6e19 | 6e21 | cm⁻³ | 源漏掺杂 |
| SDE_Doping | 6e17 | 6e20 | cm⁻³ | 扩展掺杂 |
| C_Doping | 1e16 | 1e18 | cm⁻³ | 沟道掺杂 |
| ntype_work_function | 4.15 | 4.65 | eV | N-type功函数 |
| ptype_work_function | 4.65 | 5.15 | eV | P-type功函数 |

---

## 插件模块

### circuit_eval

电路评估模块，负责：
- 生成 HSPICE 网表 (`netlist_generator.py`)
- 解析 HSPICE 输出 (`parse_hspice_output.py`)
- 定义电路配置和指标 (`schemas.py`)

### compact_model

紧凑模型模块，负责：
- 神经网络训练和数据处理 (`nn_fitting.py`)
- BSIM-NN 风格模型架构
- 物理感知损失函数

### verilog_a

Verilog-A 导出模块，负责：
- 将 PyTorch 模型转换为 Verilog-A 代码
- DC (I-V) 和 AC (C-V) 模型集成
- N-type 和 P-type 模型分别导出

### dtco_controller

评分控制器，负责：
- 计算综合评分 (`scoring.py`)
- 定义优化目标和权重
- 生成评分报告

---

## 依赖环境

### 必需软件
- **Synopsys Sentaurus TCAD**: SDE, SDEVICE
- **Synopsys HSPICE**: 电路仿真器
- **Python 3.8+**: 运行环境

### Python 依赖
```
numpy
pandas
torch
scikit-learn
scipy
```

---

## 使用示例

### 完整流程调用

```python
import asyncio
from tools.tcad_tool import TCADTool
from tools.compact_model_train_tool import DTCOCompactModelTrainTool
from tools.hspice_sim_tool import DTCOHspiceSimTool

async def run_dtco_optimization():
    # 1. TCAD 仿真
    tcad_tool = TCADTool(config, tool_schema)
    tcad_result, _, _ = await tcad_tool.execute(
        instance_id="test",
        parameters={
            "device_type": "both",
            "ntype_variables": {...},
            "ptype_variables": {...}
        }
    )

    # 2. 紧凑模型训练
    train_tool = DTCOCompactModelTrainTool(config, tool_schema)
    train_result, _, _ = await train_tool.execute(
        instance_id="test",
        parameters={"work_dir": tcad_result["work_dir"]}
    )

    # 3. 电路仿真
    sim_tool = DTCOHspiceSimTool(config, tool_schema)
    sim_result, _, _ = await sim_tool.execute(
        instance_id="test",
        parameters={
            "work_dir": tcad_result["work_dir"],
            "circuit_type": "inverter"
        }
    )

    return sim_result

# 运行
result = asyncio.run(run_dtco_optimization())
print(f"Score: {result['score']}")
```

---

## 文件输出结构

```
work_dir/
├── ntype/
│   ├── device_metrics.json          # N-type 器件指标
│   ├── cv_data.json                  # C-V 仿真数据
│   ├── result_Sat_*.plt              # I-V 仿真数据
│   ├── compact_model_nn_ntype/
│   │   ├── fet_neural_network_model.pth  # 训练好的模型
│   │   ├── integrated_ntype.va           # Verilog-A 紧凑模型
│   │   └── nn_fit_metrics.json           # 训练指标
│   └── circuit_eval/
│       └── inverter/
│           ├── inverter.sp           # HSPICE 网表
│           ├── metrics.csv           # 电路指标
│           └── sim.log               # 仿真日志
├── ptype/
│   └── ...                           # 同上
└── circuit_eval/                     # 电路级评估结果
```

---

## 注意事项

1. **工具依赖**：确保 `sde`、`sdevice`、`hspice` 可执行文件在系统 PATH 中
2. **许可证**：Synopsys 工具需要有效许可证
3. **并发控制**：使用 semaphore 限制并发仿真数量，避免资源竞争
4. **临时文件**：仿真完成后会自动清理临时目录

---

## 版本信息

- 文档版本: 1.0
- 最后更新: 2025-01
- 目标工艺: FinFET 7nm / GAA 2nm
