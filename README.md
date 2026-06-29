# KuTARA

English | [简体中文](README-cn.md)

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