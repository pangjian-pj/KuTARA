import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from monitor.clients import KubernetesClients
from planning.env import ServiceReplicaBounds

logger = logging.getLogger(__name__)


class K8sExecutor:
    """
    Kubernetes-based executor to scale Deployments (by name==service_id) in a namespace.

    Safety features:
    - enforces per-service and global bounds are checked by caller; this class will still clamp if out-of-range
    - simple cooldown: limit number of pods changed within sliding 1-second window (default max_delta_per_sec)
    - unavailable service handling: prevent infinite loop when service becomes unavailable
    """

    def __init__(self, k8s: KubernetesClients, max_delta_per_sec: int = 50):
        self.k8s = k8s
        self.max_delta_per_sec = max_delta_per_sec
        # track timestamps of recent pod adjustments (list of (ts, delta)) -- use thread-safe structure
        self.lock = threading.Lock()
        self.recent_adjustments = []  # list of timestamps (float)
        
        # Safe scaling features for unavailable services
        self.safe_min_replicas = 1  # Minimum replicas for safe operation
        self.unavailable_cooldown_seconds = 5  # Cooldown period after service becomes unavailable
        self.last_unavailable_timestamp = {}  # Track when services became unavailable

    def _prune_old(self):
        cutoff = time.time() - 1.0
        with self.lock:
            while self.recent_adjustments and self.recent_adjustments[0] < cutoff:
                self.recent_adjustments.pop(0)

    def _can_adjust(self, desired_delta: int) -> bool:
        """Return True if allowed under cooldown (total |delta| within last 1s + desired_delta <= max_delta_per_sec)"""
        self._prune_old()
        with self.lock:
            current = len(self.recent_adjustments)
            # current counts of adjustments (approx count of pod-change events in last 1s)
            if current + abs(desired_delta) > self.max_delta_per_sec:
                return False
            return True

    def _record_adjust(self, delta: int):
        with self.lock:
            for _ in range(abs(delta)):
                self.recent_adjustments.append(time.time())

    def _resolve_deployment_name(self, service_id: str, namespace: str) -> Optional[str]:
        """Resolve the actual Deployment name for a given service_id.

        In this repo we typically use deployment name == service_id, but we still keep this
        helper to make the executor more robust across different Online Boutique manifests.
        """
        # fast path
        try:
            self.k8s.apps_v1.read_namespaced_deployment(name=service_id, namespace=namespace)
            return service_id
        except Exception:
            pass

        # fallback: scan deployments once (only when needed)
        try:
            deps = self.k8s.apps_v1.list_namespaced_deployment(namespace=namespace)
            names = [d.metadata.name for d in (deps.items or []) if d and d.metadata and d.metadata.name]
            # simple heuristics
            candidates = [n for n in names if n == service_id or n.replace("-", "") == service_id.replace("-", "")]
            if candidates:
                return candidates[0]
        except Exception:
            logger.exception("Failed to list deployments in namespace=%s", namespace)

        return None

    def scale_service(self, service_id: str, target_replicas: int, namespace: str = "default", bounds: Optional[ServiceReplicaBounds] = None) -> bool:
        """
        Scale a deployment named `service_id` to `target_replicas` in `namespace`.
        Performs basic bounds clamping and cooldown enforcement.
        If deployment is in Unavailable state, scales directly to 1 replica.

        Returns True if request was issued to k8s API (not a guarantee of pod readiness).
        """
        if bounds:
            target_replicas = max(bounds.min_replicas, min(bounds.max_replicas, target_replicas))

        deploy_name = self._resolve_deployment_name(service_id, namespace) or service_id

        # get current replicas via apps API
        try:
            dep = self.k8s.apps_v1.read_namespaced_deployment(name=deploy_name, namespace=namespace)
            current = dep.spec.replicas or 0

            # --- 1. 检查 Deployment 是否不可用 ---
            is_unavailable = False
            if hasattr(dep, 'status') and hasattr(dep.status, 'conditions'):
                for condition in dep.status.conditions:
                    if condition.type == "Available" and condition.status == "False":
                        is_unavailable = True
                        break

            # --- 2. 如果服务当前不可用 ---
            if is_unavailable:
                logger.warning("Deployment %s 不可用，暂不执行扩缩容动作", service_id)
                return False
        
        except Exception as e:
            logger.warning("读取 Deployment %s/%s 失败: %s", namespace, deploy_name, e)
            current = None

        desired_delta = 0 if current is None else (target_replicas - current)
        if desired_delta == 0:
            logger.debug("目标副本数与当前一致，无需操作: %s -> %d", service_id, target_replicas)
            return True

        if not self._can_adjust(desired_delta):
            logger.warning("Cooldown 限制：1s 内允许的最大 pod 调整为 %d，当前请求 delta=%d 被拒绝", self.max_delta_per_sec, desired_delta)
            return False

        # issue scale (patch scale subresource)
        body = {"spec": {"replicas": int(target_replicas)}}
        try:
            # try the scale subresource first
            try:
                # NOTE: scale subresource belongs to AppsV1Api, not CoreV1.
                self.k8s.apps_v1.patch_namespaced_deployment_scale(
                    name=deploy_name,
                    namespace=namespace,
                    body=body,
                )
            except Exception:
                # fallback to patch deployment
                self.k8s.apps_v1.patch_namespaced_deployment(
                    name=deploy_name,
                    namespace=namespace,
                    body=body,
                )
            self._record_adjust(desired_delta)
            logger.info(
                "Issued scale for %s/%s: %s -> %s",
                namespace,
                deploy_name,
                current,
                target_replicas,
            )
            return True
        except Exception as e:
            logger.warning(
                "Scale %s/%s -> %s 失败: %s",
                namespace,
                deploy_name,
                target_replicas,
                e,
            )
            return False
