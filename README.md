# KuTARA

English | [简体中文](#kutara-中文)

KuTARA is the open-source core implementation of the paper on **"KuTARA: Topology-Aware Attention-Enhanced Reinforcement Learning for Coordinated Microservice Autoscaling in Kubernetes"**.

It provides the main reusable modules behind KuTARA:

- Collect service-level runtime states from Kubernetes and Prometheus.
- Learn topology-aware microservice representations with microservice topology-aware attention models.
- Build a Gymnasium-compatible autoscaling environment for reinforcement learning.
- Train or load RL policies with Stable-Baselines3.
- Apply scaling actions to Kubernetes Deployments.

## Motivation

Microservice autoscaling is difficult because service performance depends not only on the load of a single component, but also on topology dependencies, upstream/downstream interactions, and delayed scaling effects.

KuTARA addresses this problem by combining:

- **Topology-aware analysis**: extract dependency-aware service embeddings and workload forecasts.
- **Prediction-enhanced RL state**: inject future-aware signals into the policy observation.
- **Safe Kubernetes execution**: constrain scaling actions with replica bounds and rate limits.

The goal is to improve SLA satisfaction, resource efficiency, and scaling stability without relying on aggressive over-provisioning.

## Features

- Kubernetes and Prometheus based service monitoring.
- Topology-aware analyzer with graph encoding and temporal forecasting.
- Gymnasium autoscaling environment for online or offline RL experiments.
- Support for SAC through Stable-Baselines3.
- Continuous-to-discrete and flattened discrete action wrappers.
- Kubernetes executor with bounded replica control.
- Modular design: `monitor`, `analyze`, `planning`, and `execution`.

## Repository Layout

```text
kutara-open/
├── monitor/          # Kubernetes and Prometheus state collection
├── analyze/          # Topology-aware analyzer and forecasting models
├── planning/         # RL environment, planner, and action wrappers
├── execution/        # Kubernetes scaling executor
├── requirements.txt  # Python dependencies
└── README.md
```

## Architecture

Typical KuTARA workflow:

1. `monitor` collects service metrics, replica states, latency, and pending pod information.
2. `analyze` reads the service topology and produces topology-aware embeddings and forecasts.
3. `planning` builds the RL observation, computes reward, and selects a scaling action.
4. `execution` applies the selected scaling action to Kubernetes Deployments under safety bounds.

```text
Kubernetes + Prometheus
          │
          ▼
      monitor
          │
          ▼
      analyze  ── topology embeddings / forecasts
          │
          ▼
      planning ── RL policy decision
          │
          ▼
      execution ── bounded Kubernetes scaling
```

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need GPU acceleration, install the appropriate PyTorch build for your platform before installing the remaining dependencies.

## Quickstart

`main.py` connects monitoring, topology analysis, RL planning, and Kubernetes
execution. It defaults to dry-run mode, so it computes actions without changing
Deployments:

```bash
python main.py \
  --topology topology/service_graph.json \
  --services frontend,checkoutservice,paymentservice \
  --namespace online-boutique \
  --kubeconfig ~/.kube/config \
  --prometheus-url http://localhost:9090 \
  --timesteps 10000 \
  --output models/kutara_sac_policy
```

The topology `nodes` order must exactly match `--services`. Add `--execute`
only after validating the policy and replica bounds. Run `python main.py --help`
for all options.

## Module Overview

### `monitor`

Collects runtime states from Kubernetes and Prometheus.

Main entry points:

```python
from monitor import Monitor, MonitorConfig
```

### `analyze`

Implements topology-aware representation learning and forecasting.

Main entry points:

```python
from analyze import Analyzer, AnalyzeConfig, AnalyzeResult
```

### `planning`

Provides the RL autoscaling environment and policy utilities.

Main entry points:

```python
from planning import (
    KuTARAEnv,
    PlanningConfig,
    RewardWeights,
    ServiceReplicaBounds,
    Planner,
    wrap_for_sac,
    wrap_for_dqn,
)
```

### `execution`

Applies bounded scaling actions to Kubernetes Deployments.

Main entry point:

```python
from execution import K8sExecutor
```

## Topology File

KuTARA expects a service topology file that describes service nodes and directed dependencies.

Example:

```json
{
  "nodes": ["frontend", "checkoutservice", "paymentservice"],
  "adjacency": [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ]
}
```

If your topology uses a different schema, adapt `analyze/loader.py`.

## Notes

- Service IDs are expected to match Kubernetes Deployment names by default.
- Prometheus queries may need adaptation for different service meshes or metric exporters.
- The executor is intentionally bounded, but production usage should add stronger policy guards, authentication, rollout checks, and failure recovery.
- This release focuses on reproducible research and module reuse rather than a turnkey production autoscaler.

## Citation

If you use this code in academic work, please cite the KuTARA paper. (BibTeX to be added.)

## License

Please see the `LICENSE` file in the repository root.

---

# KuTARA 中文

[English](#kutara) | 简体中文

KuTARA 是配套论文 **《KuTARA: Topology-Aware Attention-Enhanced Reinforcement Learning for Coordinated Microservice Autoscaling in Kubernetes》** 发布的核心开源实现。

本仓库提供 KuTARA 的主要可复用模块：

- 从 Kubernetes 和 Prometheus 采集服务运行状态。
- 利用图结构和注意力模型学习微服务拓扑表征。
- 构建兼容 Gymnasium 的微服务自动扩缩容强化学习环境。
- 基于 Stable-Baselines3 训练或加载 RL 策略。
- 将策略动作安全地转换为 Kubernetes Deployment 扩缩容操作。

## 研究动机

微服务自动扩缩容的难点在于，服务性能不仅取决于单个组件的负载，还受到服务依赖关系、上下游传播效应以及扩缩容延迟的影响。

KuTARA 通过以下方式解决这一问题：

- **拓扑感知分析**：提取依赖感知的服务嵌入和工作负载预测信号。
- **预测增强 RL 状态**：将未来信息注入策略观测空间，而不是直接把预测值映射为扩缩容动作。
- **安全 Kubernetes 执行**：通过副本上下限和动作速率限制约束扩缩容行为。

KuTARA 的目标是在不依赖激进资源过度配置的情况下，同时提升 SLA 满足率、资源效率和扩缩容稳定性。

## 功能特性

- 基于 Kubernetes 和 Prometheus 的服务状态采集。
- 拓扑感知分析器，支持图编码和时间序列预测。
- 用于在线或离线 RL 实验的 Gymnasium 自动扩缩容环境。
- 通过 Stable-Baselines3 支持 SAC、PPO、DQN 和 Recurrent PPO。
- 支持连续动作到离散扩缩容动作的映射，以及离散动作展平包装。
- 带有副本边界约束的 Kubernetes 执行器。
- 模块化组织：`monitor`、`analyze`、`planning`、`execution`。

## 仓库结构

```text
kutara-open/
├── monitor/          # Kubernetes 和 Prometheus 状态采集
├── analyze/          # 拓扑感知分析器与预测模型
├── planning/         # RL 环境、规划器与动作包装器
├── execution/        # Kubernetes 扩缩容执行器
├── requirements.txt  # Python 依赖
└── README.md
```

## 架构流程

KuTARA 的典型工作流程如下：

1. `monitor` 采集服务指标、副本状态、延迟和 pending pod 信息。
2. `analyze` 读取服务拓扑，生成拓扑感知嵌入和预测结果。
3. `planning` 构造 RL 观测、计算 reward，并选择扩缩容动作。
4. `execution` 在安全边界内将动作应用到 Kubernetes Deployment。

```text
Kubernetes + Prometheus
          │
          ▼
      monitor
          │
          ▼
      analyze  ── topology embeddings / forecasts
          │
          ▼
      planning ── RL policy decision
          │
          ▼
      execution ── bounded Kubernetes scaling
```

## 安装

创建 Python 环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要 GPU 加速，请先根据你的平台安装合适版本的 PyTorch，再安装其余依赖。

## 快速开始

`main.py` 会串联监控、拓扑分析、RL 规划和 Kubernetes 执行模块。默认是
dry-run，只计算扩缩容动作而不修改 Deployment：

```bash
python main.py \
  --topology topology/service_graph.json \
  --services frontend,checkoutservice,paymentservice \
  --namespace online-boutique \
  --kubeconfig ~/.kube/config \
  --prometheus-url http://localhost:9090 \
  --timesteps 10000 \
  --output models/kutara_sac_policy
```

拓扑文件中的 `nodes` 顺序必须与 `--services` 完全一致。确认策略和安全边界后，
增加 `--execute` 才会实际修改 Kubernetes Deployment 副本数。运行
`python main.py --help` 可查看算法、权重、副本边界等全部参数。

## 模块说明

### `monitor`

负责从 Kubernetes 和 Prometheus 采集运行状态。

主要入口：

```python
from monitor import Monitor, MonitorConfig
```

### `analyze`

实现拓扑感知表示学习和预测。

主要入口：

```python
from analyze import Analyzer, AnalyzeConfig, AnalyzeResult
```

### `planning`

提供 RL 自动扩缩容环境和策略工具。

主要入口：

```python
from planning import (
    KuTARAEnv,
    PlanningConfig,
    RewardWeights,
    ServiceReplicaBounds,
    Planner,
    wrap_for_sac,
    wrap_for_dqn,
)
```

### `execution`

负责将扩缩容动作应用到 Kubernetes Deployment。

主要入口：

```python
from execution import K8sExecutor
```

## 拓扑文件

KuTARA 需要一个描述服务节点和有向依赖边的拓扑文件。

示例：

```json
{
  "nodes": ["frontend", "checkoutservice", "paymentservice"],
  "adjacency": [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ]
}
```

如果你的拓扑格式不同，可以修改 `analyze/loader.py` 中的读取逻辑。

## 使用说明

- 默认情况下，服务 ID 应与 Kubernetes Deployment 名称一致。
- Prometheus 查询语句可能需要根据你的 service mesh 或监控系统进行适配。
- 执行器已经包含基本副本边界约束，但生产部署仍应增加更严格的策略保护、认证、发布检查和故障恢复机制。
- 本开源版本主要面向论文复现和模块复用，而不是开箱即用的生产级 autoscaler。

## 引用

如果你在学术工作中使用本项目，请引用KuTARA 论文：《KuTARA: Topology-Aware Attention-Enhanced Reinforcement Learning for Coordinated Microservice Autoscaling in Kubernetes》


## 许可证

请查看仓库根目录下的 `LICENSE` 文件。
