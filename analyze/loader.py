import json
from typing import List, Tuple
import numpy as np


def load_adjacency_with_nodes(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    JSON格式:
    {"nodes": ["frontend","checkout",...], "adjacency": [[0,1,...], ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    adj = np.array(obj.get("adjacency", []), dtype=np.float32)
    nodes = obj.get("nodes")
    if nodes is None or len(nodes) == 0:
        raise ValueError("邻接文件缺少节点列表 'nodes'，请提供 nodes 对应的 service_id 顺序")
    if adj.shape[0] != adj.shape[1] or adj.shape[0] != len(nodes):
        raise ValueError("邻接矩阵维度与节点数不一致")
    return adj, nodes


def adjacency_to_edge_index(adj: np.ndarray) -> List[Tuple[int, int]]:
    """
    将邻接矩阵转为有向边索引列表（src_idx, dst_idx）
    """
    src, dst = np.where(adj > 0)
    return list(zip(src.tolist(), dst.tolist()))

