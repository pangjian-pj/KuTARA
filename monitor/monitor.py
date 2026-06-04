import logging
import math
from typing import List, Optional, Tuple

from .clients import KubernetesClients, PrometheusClient
from .types import MicroserviceState
from .utils import (
    parse_cpu_to_cores,
    parse_memory_to_mi,
    safe_div,
    build_label_selector_from_match_labels,
)

logger = logging.getLogger(__name__)


class MonitorConfig:
    """
    Monitor配置。
    - namespace: 默认命名空间
    - sla_quantile: SLA使用的分位点（0.999）P999延迟
    - prom_latency_metric: 延迟直方图bucket指标名
    - prom_latency_query_template: PromQL模板（可使用 {metric} {quantile} {service_id} {namespace} {window} {selector} 占位符）
    - prom_latency_selectors: PromQL标签选择器模板列表
    - prom_latency_window: PromQL中的时间窗口
    - prom_app_label_key: 过滤服务的label key
    - use_metrics_api: 优先使用 metrics.k8s.io（True），否则使用Prometheus作为使用量兜底
    """
    def __init__(
        self,
        namespace: str = "default",
        sla_quantile: float = 0.999,
        prom_latency_metric: str = "istio_request_duration_milliseconds_bucket",
        prom_latency_selectors: Optional[List[str]] = None,
        prom_latency_window: str = "1m",
        prom_app_label_key: str = "app",
        use_metrics_api: bool = True,
    ):
        self.namespace = namespace
        self.sla_quantile = sla_quantile
        self.prom_latency_metric = prom_latency_metric
        # 标准 Istio P999 延迟查询 (单位: 毫秒)
        # 使用 rate() 计算每秒增长率，再用 histogram_quantile 计算分位数
        self.prom_latency_query_template = (
            'histogram_quantile({quantile}, '
            'sum(rate({metric}{{{selector}}}[{window}])) by (le, destination_workload, destination_workload_namespace)'
            ')'
        )
        self.prom_latency_selectors = prom_latency_selectors or [
            # 先尝试 destination reporter（标准 Istio）
            'reporter="destination", destination_workload="{service_id}", destination_workload_namespace="{namespace}"',
            # 回退到 source reporter
            'reporter="source", destination_workload="{service_id}", destination_workload_namespace="{namespace}"',
            # 最后只按 destination_workload 过滤
            'destination_workload="{service_id}", destination_workload_namespace="{namespace}"'
        ]
        self.prom_latency_window = prom_latency_window
        self.prom_app_label_key = prom_app_label_key
        self.use_metrics_api = use_metrics_api


class Monitor:
    """
    生产级可用的Monitor实现：函数调用（非HTTP），聚合K8s/Prometheus，返回统一JSON/对象。
    """
    def __init__(
        self,
        k8s_clients: KubernetesClients,
        prom_client: Optional[PrometheusClient],
        config: Optional[MonitorConfig] = None,
    ):
        self.k8s = k8s_clients
        self.prom = prom_client
        self.config = config or MonitorConfig()

    def get_service_state(self, service_id: str, namespace: Optional[str] = None) -> MicroserviceState:
        """
        基于Deployment的selector匹配Pod，聚合requests/limits、实时使用量、SLA延迟。
        """
        ns = namespace or self.config.namespace

        req_cpu, req_mem, lim_cpu, lim_mem, label_selector, requested_replicas = self._get_deploy_resources_and_selector(service_id, ns)
        cpu_usage, mem_usage, pod_num, pending_num, avg_startup_ms = self._get_usage_for_pods(service_id, ns, label_selector)
        response_time_ms = self._get_latency_ms(service_id, namespace=ns)
        cpu_util = safe_div(cpu_usage, req_cpu)
        mem_util = safe_div(mem_usage, req_mem)

        return MicroserviceState(
            id=service_id,
            request_cpu=req_cpu,
            request_memory=req_mem,
            limit_cpu=lim_cpu,
            limit_memory=lim_mem,
            cpu_usage=cpu_usage,
            memory_usage=mem_usage,
            cpu_utilization=cpu_util,
            memory_utilization=mem_util,
            pod_num=pod_num,
            requested_replicas=requested_replicas,
            pending_pod_num=pending_num,
            avg_startup_delay_ms=avg_startup_ms,
            response_time=response_time_ms,
        )

    def _get_deploy_resources_and_selector(self, service_id: str, namespace: str) -> Tuple[float, float, float, float, str]:
        """
        读取Deployment，聚合容器requests/limits并返回标准化label_selector。
        """
        deploy = self.k8s.apps_v1.read_namespaced_deployment(service_id, namespace)
        req_cpu = req_mem = lim_cpu = lim_mem = 0.0

        for c in deploy.spec.template.spec.containers:
            req = c.resources.requests or {}
            lim = c.resources.limits or {}
            req_cpu += parse_cpu_to_cores(req.get("cpu"))
            req_mem += parse_memory_to_mi(req.get("memory"))
            lim_cpu += parse_cpu_to_cores(lim.get("cpu"))
            lim_mem += parse_memory_to_mi(lim.get("memory"))

        match_labels = {}
        if deploy.spec.selector and deploy.spec.selector.match_labels:
            match_labels = dict(deploy.spec.selector.match_labels)
        label_selector = build_label_selector_from_match_labels(match_labels)
        if not label_selector:
            # fallback 到 app=<service_id>
            label_selector = f'{self.config.prom_app_label_key}={service_id}'
            logger.warning("Deployment未提供matchLabels，使用兜底label_selector: %s", label_selector)

        # requested replicas from deployment spec
        requested_replicas = 0
        try:
            requested_replicas = int(deploy.spec.replicas or 0)
        except Exception:
            requested_replicas = 0

        return req_cpu, req_mem, lim_cpu, lim_mem, label_selector, requested_replicas

    def _get_usage_for_pods(self, service_id: str, namespace: str, label_selector: str) -> Tuple[float, float, int, int, float]:
        """
        优先从 metrics.k8s.io 拉取pod container使用量；若失败/不可用，尝试从Prometheus聚合。
        返回：cpu_total(cores)、mem_total(Mi)、pod_num
        """
        pods = self.k8s.core_v1.list_namespaced_pod(namespace, label_selector=label_selector)
        pod_num = len(pods.items)
        if pod_num == 0:
            return 0.0, 0.0, 0, 0, 0.0

        if self.config.use_metrics_api:
            cpu_total, mem_total, ok = self._try_metrics_api(pods, namespace)
            if ok:
                # also compute pending and avg startup delay from pod list
                pending_num, avg_startup_ms = self._compute_pending_and_startup(pods)
                return cpu_total, mem_total, pod_num, pending_num, avg_startup_ms
            logger.warning("metrics.k8s.io不可用或数据缺失，回退到Prometheus进行使用量估计")
        cpu_total, mem_total = self._prom_usage_fallback(service_id)
        pending_num, avg_startup_ms = self._compute_pending_and_startup(pods)
        return cpu_total, mem_total, pod_num, pending_num, avg_startup_ms

    def _compute_pending_and_startup(self, pods) -> Tuple[int, float]:
        """
        Compute number of pods in Pending state and estimate average startup delay in ms.
        We compute startup delay as the delta between the Pod's Ready condition lastTransitionTime and pod.metadata.creation_timestamp
        """
        pending = 0
        delays = []
        for pod in pods.items:
            # determine Ready condition
            try:
                conditions = getattr(pod.status, "conditions", None) or []
                ready_cond = None
                for c in conditions:
                    # condition may have 'type' or 'type'
                    if getattr(c, "type", None) == "Ready":
                        ready_cond = c
                        break
                creation_ts = getattr(pod.metadata, "creation_timestamp", None)
                if ready_cond is None:
                    # not Ready -> consider as pending
                    pending += 1
                else:
                    # if ready_cond has lastTransitionTime, compute delay
                    last_ts = getattr(ready_cond, "lastTransitionTime", None) or getattr(ready_cond, "last_transition_time", None) or getattr(ready_cond, "lastTransition", None)
                    if creation_ts is not None and last_ts is not None:
                        try:
                            delta = (last_ts - creation_ts).total_seconds() * 1000.0
                            if delta >= 0:
                                delays.append(delta)
                        except Exception:
                            # ignore malformed timestamps
                            pass
            except Exception:
                # on any unexpected structure, skip
                continue
        avg = float(sum(delays) / len(delays)) if delays else 0.0
        return pending, avg

    def _try_metrics_api(self, pods, namespace: str) -> Tuple[float, float, bool]:
        """
        通过 metrics.k8s.io 获取实时容器使用量。
        使用 CustomObjectsApi 访问 group='metrics.k8s.io', version='v1beta1', plural='pods'
        """
        cpu_total = 0.0
        mem_total = 0.0
        ok_any = False
        for pod in pods.items:
            pod_name = pod.metadata.name
            try:
                pod_metrics = self.k8s.custom_objects.get_namespaced_custom_object(
                    group="metrics.k8s.io", version="v1beta1", namespace=namespace, plural="pods", name=pod_name
                )
                containers = pod_metrics.get("containers", [])
                for c in containers:
                    usage = c.get("usage", {})
                    cpu_total += parse_cpu_to_cores(usage.get("cpu"))
                    mem_total += parse_memory_to_mi(usage.get("memory"))
                ok_any = True
            except Exception as e:
                logger.debug("读取Pod metrics失败（可能未就绪）%s: %s", pod_name, e)
        return cpu_total, mem_total, ok_any

    def _prom_usage_fallback(self, service_id: str) -> Tuple[float, float]:
        """
        通过Prometheus估计CPU/内存使用量（依赖集群的Exporter与Metric命名）。
        """
        if not self.prom:
            return 0.0, 0.0
        try:
            # 示例：以container_cpu_usage_seconds_total推导CPU使用（近1分钟平均），再汇总为“cores”
            # 注意：不同环境指标名可能不同，需要按需修改。
            cpu_q = f'sum(rate(container_cpu_usage_seconds_total{{image!="", {self.config.prom_app_label_key}="{service_id}"}}[1m]))'
            mem_q = f'sum(container_memory_working_set_bytes{{image!="", {self.config.prom_app_label_key}="{service_id}"}}) / 1024 / 1024'
            cpu_res = self.prom.query_instant(cpu_q)
            mem_res = self.prom.query_instant(mem_q)
            cpu_val = float(cpu_res[0]["value"][1]) if cpu_res else 0.0  # already in cores (seconds per second)
            mem_val = float(mem_res[0]["value"][1]) if mem_res else 0.0  # in Mi
            return cpu_val, mem_val
        except Exception as e:
            logger.warning("Prometheus使用量兜底失败: %s", e)
            return 0.0, 0.0

    def _get_latency_ms(self, service_id: str, namespace: str) -> float:
        """
        从 Prometheus 读取延迟（单位ms）。
        调用链路：service -> istio sidecar -> prometheus -> monitor
        默认使用Istio目的端指标：istio_request_duration_milliseconds_bucket
        """
        if not self.prom:
            return 0.0
        metric = self.config.prom_latency_metric
        selectors = self.config.prom_latency_selectors or []
        last_error: Optional[Exception] = None

        for selector_tpl in selectors:
            selector = selector_tpl.format(service_id=service_id, namespace=namespace)
            try:
                promql = self.config.prom_latency_query_template.format(
                    quantile=self.config.sla_quantile,
                    metric=metric,
                    service_id=service_id,
                    namespace=namespace,
                    window=self.config.prom_latency_window,
                    selector=selector,
                )
                logger.debug("Prometheus latency query: %s", promql)
                res = self.prom.query_instant(promql)
                logger.debug("Prometheus latency raw result: %s", res)
                if not res:
                    continue
                value_raw = res[0]["value"][1]
                if value_raw is None:
                    continue
                try:
                    value = float(value_raw)
                except (TypeError, ValueError):
                    continue
                if math.isnan(value):
                    continue
                # Istio 的 milliseconds_bucket 返回的是毫秒，直接返回
                return value
            except Exception as e:
                last_error = e
                continue

        if last_error:
            logger.warning("Prometheus延迟查询失败(service=%s): %s", service_id, last_error)
        else:
            logger.debug("未查询到延迟指标(service=%s, namespace=%s)", service_id, namespace)
        return 0.0

