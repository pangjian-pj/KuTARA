import logging
import os
from typing import Optional, Any, Dict, List

from kubernetes import client, config
from kubernetes.client import AppsV1Api, CoreV1Api, CustomObjectsApi
from prometheus_api_client import PrometheusConnect

logger = logging.getLogger(__name__)


class KubernetesClients:
    """
    聚合K8s相关客户端（Apps/Core/CustomObjects）。
    支持指定kubeconfig路径；若未指定则优先InCluster，否则回落本地默认kubeconfig。
    """
    def __init__(self, in_cluster_preferred: bool = True, kubeconfig_path: Optional[str] = None):
        self._load_config(in_cluster_preferred=in_cluster_preferred, kubeconfig_path=kubeconfig_path)
        self.apps_v1: AppsV1Api = client.AppsV1Api()
        self.core_v1: CoreV1Api = client.CoreV1Api()
        self.custom_objects: CustomObjectsApi = client.CustomObjectsApi()

    @staticmethod
    def _load_config(in_cluster_preferred: bool = True, kubeconfig_path: Optional[str] = None):
        # 若显式提供kubeconfig路径，则优先使用该路径
        if kubeconfig_path:
            try:
                config.load_kube_config(config_file=kubeconfig_path)
                logger.info("Loaded kube config from explicit path: %s", kubeconfig_path)
                return
            except Exception as e:
                logger.error("无法从指定路径加载kubeconfig(%s): %s", kubeconfig_path, e)
                raise

        if in_cluster_preferred:
            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster kube config")
                return
            except Exception as e:
                logger.warning("InCluster配置加载失败，回退到KUBECONFIG: %s", e)
        # fallback
        try:
            config.load_kube_config()
            logger.info("Loaded kube config from KUBECONFIG")
        except Exception as e:
            logger.error("无法加载Kubernetes配置: %s", e)
            raise


class PrometheusClient:
    """
    Prometheus API客户端适配器。
    """
    def __init__(self, url: str, disable_ssl: bool = True, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self._client = PrometheusConnect(url=url, disable_ssl=disable_ssl, headers=headers or {})

    def query_instant(self, promql: str) -> List[Dict[str, Any]]:
        return self._client.custom_query(query=promql)

