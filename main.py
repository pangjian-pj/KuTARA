"""KuTARA end-to-end online training entry point."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from analyze import AnalyzeConfig, Analyzer
from execution import K8sExecutor
from monitor import Monitor, MonitorConfig
from monitor.clients import KubernetesClients, PrometheusClient
from planning import (
    Planner,
    PlanningConfig,
    ServiceReplicaBounds,
    wrap_for_dqn,
    wrap_for_sac,
)


LOGGER = logging.getLogger("kutara.main")


def csv_list(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one service ID is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect monitor, analyzer, planner, and Kubernetes executor."
    )
    parser.add_argument("--topology", required=True, help="Topology JSON path")
    parser.add_argument(
        "--services", required=True, type=csv_list, help="Comma-separated deployment names"
    )
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--kubeconfig", help="Kubeconfig path; defaults to in-cluster/KUBECONFIG")
    parser.add_argument("--prometheus-url", help="Prometheus URL; omit to disable Prometheus queries")
    parser.add_argument("--analyzer-weights", help="Optional analyzer checkpoint")
    parser.add_argument("--encoder", choices=("mta", "gat", "gcn"), default="mta")
    parser.add_argument(
        "--algo", choices=("sac", "ppo", "dqn", "recurrent_ppo"), default="sac"
    )
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--policy", help="Existing policy to continue training")
    parser.add_argument("--output", default="models/kutara_policy")
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=3)
    parser.add_argument("--max-scale-per-step", type=int, default=1)
    parser.add_argument("--step-wait-seconds", type=float, default=0.0)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply scaling actions. Without this flag, actions are dry-run only.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    topology_path = Path(args.topology).expanduser()
    if not topology_path.is_file():
        raise ValueError(f"topology file does not exist: {topology_path}")
    with topology_path.open("r", encoding="utf-8") as file:
        topology = json.load(file)
    nodes = topology.get("nodes")
    adjacency = topology.get("adjacency")
    if nodes != args.services:
        raise ValueError("--services must exactly match topology 'nodes' order")
    if not isinstance(adjacency, list) or len(adjacency) != len(nodes):
        raise ValueError("topology must contain a square 'adjacency' matrix")
    if any(not isinstance(row, list) or len(row) != len(nodes) for row in adjacency):
        raise ValueError("topology 'adjacency' matrix must be square")
    if args.timesteps <= 0 or args.episode_steps <= 0:
        raise ValueError("timesteps and episode-steps must be positive")
    if args.min_replicas < 0 or args.max_replicas < args.min_replicas:
        raise ValueError("invalid replica bounds")


def build_environment(args: argparse.Namespace):
    kubeconfig = os.path.expanduser(args.kubeconfig) if args.kubeconfig else None
    k8s = KubernetesClients(
        in_cluster_preferred=kubeconfig is None,
        kubeconfig_path=kubeconfig,
    )
    prometheus = PrometheusClient(args.prometheus_url) if args.prometheus_url else None
    monitor = Monitor(
        k8s_clients=k8s,
        prom_client=prometheus,
        config=MonitorConfig(namespace=args.namespace),
    )
    analyzer = Analyzer(
        AnalyzeConfig(
            device="cpu",
            weights_path=args.analyzer_weights,
            encoder_type=args.encoder,
        )
    )
    config = PlanningConfig(
        adjacency_json_path=str(Path(args.topology).expanduser()),
        service_ids=args.services,
        namespace=args.namespace,
        replica_bounds={
            service: ServiceReplicaBounds(args.min_replicas, args.max_replicas)
            for service in args.services
        },
        max_scale_per_step=args.max_scale_per_step,
        step_wait_seconds=args.step_wait_seconds,
        max_episode_steps=args.episode_steps,
    )

    executor = K8sExecutor(k8s) if args.execute else None
    executor_fn = executor.scale_service if executor else None
    planner = Planner(algo=args.algo, seed=args.seed)
    env = planner.make_env(monitor, analyzer, config, executor=executor_fn)
    if args.algo == "sac":
        env = wrap_for_sac(env, service_ids=args.services)
    elif args.algo == "dqn":
        env = wrap_for_dqn(env, service_ids=args.services)
    return planner, env


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        validate_args(args)
        planner, env = build_environment(args)
        mode = "execute" if args.execute else "dry-run"
        LOGGER.info("Starting %s training with %s for %d steps", mode, args.algo, args.timesteps)
        if args.policy:
            planner.load(os.path.expanduser(args.policy), env=env)
        else:
            planner.build_model(env)
        planner.train(total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.policy))
        output = os.path.expanduser(args.output)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        planner.save(output)
        LOGGER.info("Training complete; policy saved to %s.zip", output)
        return 0
    except (OSError, ValueError) as exc:
        LOGGER.error("Startup failed: %s", exc)
        return 2
    finally:
        if "env" in locals():
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
