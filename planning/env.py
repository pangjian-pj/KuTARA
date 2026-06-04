import time
import logging
import csv
import os
from datetime import datetime
from dataclasses import dataclass,field
from typing import Callable, Dict, List, Optional, Tuple, Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from monitor.monitor import Monitor
from analysis.analyzer import Analyzer, AnalyzeConfig, AnalyzeResult

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    w_c: float = 0.3
    w_l: float = 0.3
    w_r: float = 0.3
    w_pending: float = 0.1


@dataclass
class ServiceReplicaBounds:
    min_replicas: int
    max_replicas: int


@dataclass
class PlanningConfig:
    adjacency_json_path: str
    service_ids: List[str]
    namespace: str = "default"
    # 奖励权重与阈值
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    latency_threshold_ms: float = 500.0
    # 延迟奖励平滑系数
    alpha: float = 0.01
    # 副本上下限
    replica_bounds: Dict[str, ServiceReplicaBounds] = None
    # 冷却设置（每步最大调整Pod总数）
    max_scale_per_step: int = 5
    # 预测步长与模型配置（透传给Analyzer）
    analyze_config: AnalyzeConfig = AnalyzeConfig(horizon=6, device="cpu")
    # 观察空间包含的开关（未来可扩展）
    include_embedding: bool = True
    include_forecast: bool = True
    # Intermediate transition signals (e.g., pending pods, startup delay)
    include_interm: bool = True

    # How to integrate prediction with RL
    # - state: append forecast to observation (default KuTARA)
    # - action: use forecast to override/shape the executed scaling action (RL weakened)
    # - reward: use forecast to shape reward (prediction affects learning signal)
    pred_integration: Literal["state", "action", "reward"] = "state"

    # ActionPred knobs
    # forecast_h_index: which horizon step of forecast to use (0 = next step)
    actionpred_h_index: int = 0
    # Convert predicted pod demand to delta by (pred_pod - current_pod) * gain
    actionpred_gain: float = 1.0
    # Clamp the ActionPred delta to avoid extreme jumps
    actionpred_max_abs_delta: int = 5

    # RewardPred knobs
    # Penalize predicted demand (or predicted pod shortage/excess) to encourage proactive scaling.
    rewardpred_weight: float = 0.05
    # 步长等待（秒），用于真实集群时步进
    step_wait_seconds: float = 0.0
    # optional tensorboard log directory; if set, env will write per-step metrics
    tensorboard_log_dir: Optional[str] = None
    # episode 配置
    max_episode_steps: int = 200
    # 若非 None，会把每步记录追加到 csv
    log_csv_path: Optional[str] = None


class KuTARAEnv(gym.Env):
    """
    KuTARA 规划环境：
    - 状态：对每个服务拼接 monitor 数值特征 + embedding + forecast
    - 动作：MultiDiscrete([N, len(deltas)])，选择服务与扩缩容步长
    - 奖励：多目标奖励（成本、延迟、资源利用）
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        monitor: Monitor,
        analyzer: Analyzer,
        config: PlanningConfig,
        executor: Optional[Callable[[str, int], None]] = None,
    ):
        super().__init__()
        self.monitor = monitor
        self.analyzer = analyzer
        self.cfg = config
        self.executor = executor  # 可注入实际执行扩缩容的函数：fn(service_id, target_replicas)

        self.service_ids = list(self.cfg.service_ids)
        self.N = len(self.service_ids)
        assert self.N >= 1, "service_ids 不能为空"

        # MultiDiscrete 动作空间
        self.deltas = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5], dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([self.N, len(self.deltas)])

        # 构建一次观测空间形状（需要先拉一次embedding与feature）
        obs = self._build_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        # 维护副本状态（用于目标与边界检查），默认从 monitor.state 的 pod_num 读入
        self.current_replicas = {}
        # episode 与 step 计数
        self.step_count = 0
        self.episode_count = 0
        self._csv_file = None
        self._csv_writer = None
        # episode 累计 reward
        self._episode_reward_accum = 0.0
        # 内存缓冲 CSV 行，最后在 episode 结束时一次性落盘
        self._csv_buffer = []
        # 如果配置了 csv 路径，延迟打开（在 reset 时打开）
        # cache for last analyze result (used by get_metrics)
        self._last_analyze_result = None
        # keep tensorboard writer attributes but default to None so env doesn't write itself
        self._tb_writer = None
        self._tb_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # increment episode counter and reset per-episode state
        self.episode_count += 1
        self.step_count = 0
        self._episode_reward_accum = 0.0
        # open csv if configured
        if self.cfg.log_csv_path and self._csv_file is None:
            os.makedirs(os.path.dirname(self.cfg.log_csv_path), exist_ok=True)
            need_header = True
            if os.path.exists(self.cfg.log_csv_path) and os.path.getsize(self.cfg.log_csv_path) > 0:
                need_header = False
            self._csv_file = open(self.cfg.log_csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            if need_header:
                header = [
                    "timestamp", "episode", "step", "service_id", "target_replicas", "current_replicas",
                    "pod_num", "pending_pod_num", "avg_startup_delay_ms", "response_time_ms",
                    "reward_total", "r_c", "r_l", "r_r", "p_pending"
                ]
                self._csv_writer.writerow(header)
                self._csv_file.flush()

        self._refresh_replicas()
        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action):
        service_idx, delta_idx = int(action[0]), int(action[1])
        service_id = self.service_ids[service_idx]
        delta = int(self.deltas[delta_idx])

        # If ActionPred variant is enabled, override delta using forecast.
        if getattr(self.cfg, "pred_integration", "state") == "action":
            try:
                # Ensure analyze result exists (step() calls _build_observation later, but we want forecast now)
                if self._last_analyze_result is None:
                    result: AnalyzeResult = self.analyzer.analyze_topology(
                        adjacency_json_path=self.cfg.adjacency_json_path,
                        monitor=self.monitor,
                        history_node_features=None,
                    )
                    self._last_analyze_result = result

                h = int(getattr(self.cfg, "actionpred_h_index", 0))
                h = max(0, h)
                pred_pod = 0.0
                if self._last_analyze_result is not None and h < len(self._last_analyze_result.forecast.future_node_pod_num):
                    pred_pod = float(self._last_analyze_result.forecast.future_node_pod_num[h].get(service_id, 0.0))
                current = self.current_replicas.get(service_id, 0)
                gain = float(getattr(self.cfg, "actionpred_gain", 1.0))
                delta_pred = int(round((pred_pod - float(current)) * gain))
                max_abs = int(getattr(self.cfg, "actionpred_max_abs_delta", 5))
                delta = int(np.clip(delta_pred, -max_abs, max_abs))
            except Exception:
                # fall back to agent action
                logger.exception("ActionPred override failed; falling back to agent action")

        # 应用冷却/约束：裁剪delta
        delta = int(np.clip(delta, -self.cfg.max_scale_per_step, self.cfg.max_scale_per_step))

        # 目标副本数
        current = self.current_replicas.get(service_id, 0)
        bounds = self._get_bounds(service_id)
        target = int(np.clip(current + delta, bounds.min_replicas, bounds.max_replicas))

        # 执行动作（若提供执行器），否则仅更新本地副本估计（dry-run）
        if self.executor:
            try:
                self.executor(service_id, target,self.cfg.namespace)
            except Exception as e:
                logger.warning("执行扩缩容失败：%s", e)
        self.current_replicas[service_id] = target

        # 可等待一段时间使监控指标更新（真实集群模式）
        if self.cfg.step_wait_seconds > 0:
            time.sleep(self.cfg.step_wait_seconds)

        # 刷新观测与计算奖励
        obs = self._build_observation()
        reward, reward_components = self._compute_reward()

        # 更新计数与累计reward
        self.step_count += 1
        self._episode_reward_accum += float(reward)

        # 采集被操作服务的即时指标用于记录
        try:
            s = self.monitor.get_service_state(service_id, namespace=self.cfg.namespace)
            pod_num = int(s.pod_num)
            pending = int(s.pending_pod_num)
            avg_startup = float(s.avg_startup_delay_ms)
            resp_ms = float(s.response_time)
            current_repl = int(self.current_replicas.get(service_id, 0))
        except Exception:
            pod_num = pending = 0
            avg_startup = resp_ms = 0.0
            current_repl = int(self.current_replicas.get(service_id, 0))

        # 缓存 CSV 行（若启用），最后在 episode 结束时一次性写入磁盘
        if self.cfg.log_csv_path is not None:
            row = [
                datetime.utcnow().isoformat() + "Z",
                self.episode_count,
                self.step_count,
                service_id,
                target,
                current_repl,
                pod_num,
                pending,
                avg_startup,
                resp_ms,
                float(reward),
                float(reward_components.get("r_c", 0.0)),
                float(reward_components.get("r_l", 0.0)),
                float(reward_components.get("r_r", 0.0)),
                float(reward_components.get("p_pending", 0.0)),
            ]
            try:
                self._csv_buffer.append(row)
            except Exception:
                logger.exception("缓存 CSV 行时出错")

        # 判断 episode 是否结束
        terminated = False
        if self.step_count >= self.cfg.max_episode_steps:
            terminated = True

        truncated = False
        # standard info to return every step
        info = {
            "target_replicas": {service_id: target},
            "reward_components": reward_components,
            "step": self.step_count,
        }

        # when episode ends, include episode summary in the info dict under 'episode'
        # as expected by common RL libraries (e.g., SB3 ep_info_buffer) — a dict with
        # total return 'r' and episode length 'l'
        if terminated or truncated:
            info["episode"] = {"r": float(self._episode_reward_accum), "l": int(self.step_count)}
            info["episode_done"] = True
            info["episode_summary"] = {
                "episode": self.episode_count,
                "steps": self.step_count,
                "total_reward": self._episode_reward_accum,
            }
            # flush CSV buffer was handled above; also flush TensorBoard for episode end
            if self._tb_writer is not None:
                try:
                    self._tb_writer.flush()
                except Exception:
                    logger.exception("Failed to flush tensorboard writer")
            # 在 episode 结束时一次性写入 CSV 并关闭文件句柄
            if self.cfg.log_csv_path is not None and self._csv_buffer:
                try:
                    # ensure directory exists
                    os.makedirs(os.path.dirname(self.cfg.log_csv_path), exist_ok=True)
                    need_header = True
                    if os.path.exists(self.cfg.log_csv_path) and os.path.getsize(self.cfg.log_csv_path) > 0:
                        need_header = False
                    with open(self.cfg.log_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if need_header:
                            header = [
                                "timestamp", "episode", "step", "service_id", "target_replicas", "current_replicas",
                                "pod_num", "pending_pod_num", "avg_startup_delay_ms", "response_time_ms",
                                "reward_total", "r_c", "r_l", "r_r", "p_pending"
                            ]
                            writer.writerow(header)
                        writer.writerows(self._csv_buffer)
                        f.flush()
                except Exception:
                    logger.exception("在 episode 结束时写入 CSV 失败")
                finally:
                    # 清空缓冲
                    self._csv_buffer = []

        return obs, float(reward), terminated, truncated, info

    # write per-service metrics to tensorboard (after constructing obs and reward)
    def _tb_log_step(self):
        if self._tb_writer is None or self._last_analyze_result is None:
            return
        try:
            # per-service scalars
            for sid in self.service_ids:
                s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
                prefix = f"service/{sid}"
                self._tb_writer.add_scalar(f"{prefix}/pod_num", float(s.pod_num), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/pending_pod_num", float(s.pending_pod_num), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/requested_replicas", float(s.requested_replicas), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/cpu_utilization", float(s.cpu_utilization), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/cpu_usage", float(s.cpu_usage), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/memory_usage", float(s.memory_usage), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/response_time_ms", float(s.response_time), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/avg_startup_delay_ms", float(s.avg_startup_delay_ms), self._tb_step)
                # embeddings as histogram if present
                emb = self._last_analyze_result.embedding.node_embeddings.get(sid)
                if emb is not None:
                    try:
                        self._tb_writer.add_histogram(f"{prefix}/embedding", np.asarray(emb, dtype=np.float32), self._tb_step)
                    except Exception:
                        # some tensorboard backends may not accept histograms of certain shapes
                        pass
                # forecast values
                fut_cpu = self._last_analyze_result.forecast.future_node_cpu_util[0].get(sid, 0.0)
                fut_pod = self._last_analyze_result.forecast.future_node_pod_num[0].get(sid, 0.0)
                self._tb_writer.add_scalar(f"{prefix}/forecast_cpu_next", float(fut_cpu), self._tb_step)
                self._tb_writer.add_scalar(f"{prefix}/forecast_pod_next", float(fut_pod), self._tb_step)

            # overall reward components
            # note: reward components were computed in step() and stored in local variable; we'll recompute here
            _, reward_components = self._compute_reward()
            for k, v in reward_components.items():
                self._tb_writer.add_scalar(f"reward/{k}", float(v), self._tb_step)

            self._tb_step += 1
        except Exception:
            logger.exception("Failed to write tensorboard step")

    def close(self):
        # Close tensorboard writer if present
        # placeholder for resources
        return

    def get_metrics(self) -> Dict[str, float]:
        """Return a flat dict of per-service metrics and global reward components.
        Structure:
          {
             'service/<sid>/pod_num': value,
             'service/<sid>/cpu_utilization': value,
             ...,
             'reward/r_c': value, ...
          }
        This is intended to be called from the main process (callback) to log to
        a single SummaryWriter.
        """
        metrics: Dict[str, float] = {}
        # ensure last analyze result is available
        if self._last_analyze_result is None:
            try:
                result: AnalyzeResult = self.analyzer.analyze_topology(
                    adjacency_json_path=self.cfg.adjacency_json_path,
                    monitor=self.monitor,
                    history_node_features=None,
                )
                self._last_analyze_result = result
            except Exception:
                self._last_analyze_result = None

        # per-service scalars
        for sid in self.service_ids:
            try:
                s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
                prefix = f"service/{sid}"
                metrics[f"{prefix}/pod_num"] = float(s.pod_num)
                metrics[f"{prefix}/pending_pod_num"] = float(s.pending_pod_num)
                metrics[f"{prefix}/requested_replicas"] = float(s.requested_replicas)
                metrics[f"{prefix}/cpu_utilization"] = float(s.cpu_utilization)
                metrics[f"{prefix}/cpu_usage"] = float(s.cpu_usage)
                metrics[f"{prefix}/memory_usage"] = float(s.memory_usage)
                metrics[f"{prefix}/response_time_ms"] = float(s.response_time)
                metrics[f"{prefix}/avg_startup_delay_ms"] = float(s.avg_startup_delay_ms)
                # forecasts
                if self._last_analyze_result is not None:
                    fut_cpu = self._last_analyze_result.forecast.future_node_cpu_util[0].get(sid, 0.0)
                    fut_pod = self._last_analyze_result.forecast.future_node_pod_num[0].get(sid, 0.0)
                    metrics[f"{prefix}/forecast_cpu_next"] = float(fut_cpu)
                    metrics[f"{prefix}/forecast_pod_next"] = float(fut_pod)
            except Exception:
                logger.exception("Failed to collect metrics for %s", sid)

        # reward components
        try:
            total_reward, reward_components = self._compute_reward()
            metrics["reward/total"] = float(total_reward)
            for k, v in reward_components.items():
                metrics[f"reward/{k}"] = float(v)
        except Exception:
            logger.exception("Failed to compute reward components for metrics")

        return metrics

    def _refresh_replicas(self):
        for sid in self.service_ids:
            s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            self.current_replicas[sid] = int(s.pod_num)

    def _get_bounds(self, service_id: str) -> ServiceReplicaBounds:
        if self.cfg.replica_bounds and service_id in self.cfg.replica_bounds:
            return self.cfg.replica_bounds[service_id]
        # 默认下限1，上限10
        return ServiceReplicaBounds(min_replicas=1, max_replicas=10)

    def _build_observation(self) -> np.ndarray:
        """
        观测拼接（必选项）：
        - monitor 数值特征
        - 拓扑 embedding
    - forecast (未来2步的3项预测值：cpu_util, pod_num, edge_latency)
        最终按 service 级拼接后展平。
        """
        # 1) run analyze 获取拓扑（embedding/forecast）
        result: AnalyzeResult = self.analyzer.analyze_topology(
            adjacency_json_path=self.cfg.adjacency_json_path,
            monitor=self.monitor,
            history_node_features=None,
        )

        # cache last analyze result for tensorboard logging
        try:
            self._last_analyze_result = result
        except Exception:
            self._last_analyze_result = None

        # 2) 固定各段特征长度（用于“0占位”保证 obs 维度恒定）
        #    重要：同一个 SB3/SAC 模型要求 observation_space shape 恒定。
        #    因此这里无论开关如何，都输出同样长度的向量；禁用段用 0 向量占位。
        # NOTE: observation shape must match the pre-trained SB3/SAC model.
        # Per-service dim: base(5) + embedding(32) + forecast(6) = 43
        base_dim = 5
        emb_dim = int(getattr(self.analyzer.cfg, "transformer_d_model", 32))
        forecast_steps = 2
        forecast_dim = 3 * forecast_steps  # [cpu_util, pod_num, edge_latency] * steps

        per_service_features: List[np.ndarray] = []

        for sid in self.service_ids:
            state = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            base_feat = np.array(
                [
                    float(state.cpu_utilization) / 100.0,
                    float(state.memory_utilization) / 100.0,
                    float(state.pod_num)
                    / max(1.0, float(self._get_bounds(sid).max_replicas)),
                    float(state.pending_pod_num)
                    / max(1.0, float(self._get_bounds(sid).max_replicas)),
                    float(state.response_time) / self.cfg.latency_threshold_ms,
                ],
                dtype=np.float32,
            )
            assert base_feat.size == base_dim

            add_list: List[np.ndarray] = [base_feat]

            # Topology embedding (fixed-size; optional via zero-mask)
            emb_vec = np.zeros((emb_dim,), dtype=np.float32)
            if getattr(self.cfg, "include_embedding", True):
                emb = result.embedding.node_embeddings.get(sid)
                if emb is not None:
                    emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if emb_arr.size >= emb_dim:
                        emb_vec[:] = emb_arr[:emb_dim]
                    else:
                        emb_vec[: emb_arr.size] = emb_arr
            add_list.append(emb_vec)

            # Forecast features (fixed-size; optional via zero-mask)
            forecast_features: List[float] = []
            for h in range(forecast_steps):
                cpu_util_pred = 0.0
                if h < len(result.forecast.future_node_cpu_util):
                    cpu_util_pred = float(
                        result.forecast.future_node_cpu_util[h].get(sid, 0.0)
                    )

                pod_num_pred = 0.0
                if h < len(result.forecast.future_node_pod_num):
                    pod_num_pred = float(result.forecast.future_node_pod_num[h].get(sid, 0.0))

                edge_latency_pred = 0.0
                if hasattr(result.forecast, "future_node_edge_latency") and h < len(
                    result.forecast.future_node_edge_latency
                ):
                    edge_latency_pred = float(
                        result.forecast.future_node_edge_latency[h].get(sid, 0.0)
                    )

                forecast_features.extend([cpu_util_pred, pod_num_pred, edge_latency_pred])

            forecast_vec = np.zeros((forecast_dim,), dtype=np.float32)
            if (
                getattr(self.cfg, "include_forecast", True)
                and getattr(self.cfg, "pred_integration", "state") == "state"
            ):
                f_arr = np.asarray(forecast_features, dtype=np.float32).reshape(-1)
                if f_arr.size >= forecast_dim:
                    forecast_vec[:] = f_arr[:forecast_dim]
                else:
                    forecast_vec[: f_arr.size] = f_arr
            add_list.append(forecast_vec)

            per_service_features.append(np.concatenate(add_list, axis=0))

        obs = np.concatenate(per_service_features, axis=0).astype(np.float32)
        return obs

    def _compute_reward(self) -> Tuple[float, dict]:
        """
        R = w_c * r_c + w_l * r_l + w_r * r_r - w_pending * p_pending
        - 成本：pod数越少越好
        - 延迟：小于阈值时按比例奖励，否则为0
        - 资源利用：平均内存利用率越低惩罚越小
        """
        weights = self.cfg.reward_weights

        # 成本项：使用当前副本数与上下限归一化
        pods = []
        for sid in self.service_ids:
            s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            pods.append(float(s.pod_num))
        pods = np.array(pods, dtype=np.float32)
        mins = np.array([self._get_bounds(sid).min_replicas for sid in self.service_ids], dtype=np.float32)
        maxs = np.array([self._get_bounds(sid).max_replicas for sid in self.service_ids], dtype=np.float32)
        denom = np.maximum(maxs - mins, 1.0)
        r_c = 1.0 - float(np.sum(pods - mins) / np.sum(denom))

        # 延迟项：聚合服务的响应时间为应用响应时间
        latencies = []
        for sid in self.service_ids:
            s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            latencies.append(float(s.response_time))
        s_a = float(np.nanmedian(latencies)) if len(latencies) > 0 else 0.0
        t_a = self.cfg.latency_threshold_ms
        alpha = self.cfg.alpha
        r_l = max(0, 1 - s_a / t_a)

        # 资源利用项：使用 CPU 与内存利用率的加权平均（每项权重0.5），值范围为0~1
        # 公式：r_r = 1 - (1/M) * sum(min(cpu/100,1) * w_cpu + min(mem/100,1) * w_mem)
        w_cpu = 0.5
        w_mem = 0.5
        utils = []
        for sid in self.service_ids:
            s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            # 将利用率从百分比转换为小数，并裁剪到最大1.0
            cpu_u = min(float(s.cpu_utilization) / 100.0, 1.0)
            mem_u = min(float(s.memory_utilization) / 100.0, 1.0)
            utils.append(w_cpu * cpu_u + w_mem * mem_u)
        utils = np.array(utils, dtype=np.float32) if utils else np.array([], dtype=np.float32)
        # 若没有服务，默认资源利用带来最大奖励（1.0）
        r_r = 1.0 - float(np.mean(utils)) if utils.size > 0 else 1.0

        # pending penalty: fraction of non-running pods
        pending_counts = []
        total_pods = []
        for sid in self.service_ids:
            s = self.monitor.get_service_state(sid, namespace=self.cfg.namespace)
            pending_counts.append(float(s.pending_pod_num))
            total_pods.append(float(s.pod_num) + float(s.pending_pod_num))
        pending_counts = np.array(pending_counts, dtype=np.float32)
        total_pods = np.array(total_pods, dtype=np.float32)
        # avoid div by zero
        frac_non_running = 1.0 - np.sum(np.where(total_pods > 0, (total_pods - pending_counts) / total_pods, 1.0)) / float(max(1, len(total_pods)))
        # 使用sigmoid函数平滑缩放pending惩罚
        alpha = self.cfg.alpha
        p_pending = float(1.0 / (1.0 + np.exp(-alpha * frac_non_running)))

        total = float(weights.w_c * r_c + weights.w_l * r_l + weights.w_r * r_r - weights.w_pending * p_pending)

        # RewardPred: add prediction-based shaping term.
        # We keep it simple and stable: penalize predicted next-step pod demand mismatch.
        pred_mode = getattr(self.cfg, "pred_integration", "state")
        r_pred = 0.0
        if pred_mode == "reward" and getattr(self.cfg, "include_forecast", True):
            try:
                # Ensure analyze result exists
                if self._last_analyze_result is None:
                    result: AnalyzeResult = self.analyzer.analyze_topology(
                        adjacency_json_path=self.cfg.adjacency_json_path,
                        monitor=self.monitor,
                        history_node_features=None,
                    )
                    self._last_analyze_result = result

                h = int(getattr(self.cfg, "actionpred_h_index", 0))
                h = max(0, h)
                mismatches = []
                if self._last_analyze_result is not None and h < len(self._last_analyze_result.forecast.future_node_pod_num):
                    for sid in self.service_ids:
                        pred = float(self._last_analyze_result.forecast.future_node_pod_num[h].get(sid, 0.0))
                        cur = float(self.current_replicas.get(sid, 0))
                        mismatches.append(abs(pred - cur))
                if mismatches:
                    # normalize by max replica bound mean to keep magnitude small
                    norm = float(np.mean([self._get_bounds(sid).max_replicas for sid in self.service_ids]))
                    norm = max(norm, 1.0)
                    r_pred = -float(np.mean(mismatches) / norm)
            except Exception:
                logger.exception("RewardPred shaping failed")

        if r_pred != 0.0:
            w_pred = float(getattr(self.cfg, "rewardpred_weight", 0.05))
            total = float(total + w_pred * r_pred)
        components = {
            "r_c": float(r_c),
            "r_l": float(r_l),
            "r_r": float(r_r),
            "p_pending": float(p_pending),
            "r_pred": float(r_pred),
            "w_c": float(weights.w_c),
            "w_l": float(weights.w_l),
            "w_r": float(weights.w_r),
            "w_pending": float(weights.w_pending),
            "w_pred": float(getattr(self.cfg, "rewardpred_weight", 0.05)),
        }
        return total, components

