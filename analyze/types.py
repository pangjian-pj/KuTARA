from typing import Dict, List, Tuple
from pydantic import BaseModel, Field


class TopologyEmbedding(BaseModel):
    """
    节点/边的拓扑嵌入结果：
    - node_embeddings: 映射 service_id -> embedding(list[float])
    - edge_index: 按 (src_idx, dst_idx) 的有向边列表（与输入邻接矩阵一致）
    - node_order: 计算时使用的节点顺序（与邻接矩阵行列顺序一致）
    """
    node_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    edge_index: List[Tuple[int, int]] = Field(default_factory=list)
    node_order: List[str] = Field(default_factory=list)


class TopologyForecast(BaseModel):
    """
    拓扑负载预测：
    - future_node_cpu_util[h][service_id] -> 第h步的CPU利用率预测
    - future_node_pod_num[h][service_id] -> 第h步的Pod数预测
    - future_edge_latency[h][(src, dst)] -> 第h步的边延迟预测（ms）
    """
    horizon: int
    future_node_cpu_util: List[Dict[str, float]]
    future_node_pod_num: List[Dict[str, float]]
    future_edge_latency: List[Dict[Tuple[str, str], float]]


class AnalyzeResult(BaseModel):
    """
    Analyze模块输出：拓扑嵌入 + 预测
    """
    embedding: TopologyEmbedding
    forecast: TopologyForecast

