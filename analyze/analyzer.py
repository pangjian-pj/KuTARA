import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from .types import TopologyEmbedding, TopologyForecast, AnalyzeResult
from .loader import load_adjacency_with_nodes, adjacency_to_edge_index
from .models import Informer, build_graph_encoder

if TYPE_CHECKING:
    from monitor import Monitor

logger = logging.getLogger(__name__)


class AnalyzeConfig:
    def __init__(
        self,
        horizon: int = 6,
        gcn_hidden_dim: int = 32,
        transformer_d_model: int = 32,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 128,
        dropout: float = 0.1,
        device: str = "cpu",
        history_length: int = 4,
        weights_path: Optional[str] = None,
        use_causal_mask: bool = True,
        encoder_type: str = "mta",
    ):
        encoder_type = (encoder_type or "mta").lower()
        if encoder_type not in {"mta", "gat", "gcn"}:
            raise ValueError(f"encoder_type must be one of mta/gat/gcn, got: {encoder_type}")
        self.horizon = horizon
        self.gcn_hidden_dim = gcn_hidden_dim
        self.transformer_d_model = transformer_d_model
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ffn_dim = transformer_ffn_dim
        self.dropout = dropout
        self.device = device
        self.history_length = history_length
        self.weights_path = weights_path
        self.use_causal_mask = use_causal_mask
        self.encoder_type = encoder_type


class Analyzer:
    """
    Analyze模块主入口：
    - 加载拓扑（邻接矩阵+节点列表）
    - 使用Monitor获取节点特征：包含 Monitor 返回的全部数值字段
    - 使用指定图编码器(MTA/GAT/GCN)进行图学习
    - 使用Informer进行未来H步预测（节点CPU利用率/Pod数、边延迟）
    """
    def __init__(self, config: Optional[AnalyzeConfig] = None):
        self.cfg = config or AnalyzeConfig()
        self.device = torch.device(self.cfg.device)

        # 将在 analyze_topology 内按输入维度初始化模型
        self.encoder = None
        self.gat = None
        self.informer = None
        # 持久化的线性投影层（将图编码器输出映射到 transformer d_model）
        self.proj_to_dmodel = None
        # 将时间序列动态特征映射到 transformer d_model 的持久层
        self.time_proj = None
        self.history_len = self.cfg.history_length
        self.history_buffer = []
        self.weights_loaded = False

    def _build_node_features(self, monitor: "Monitor", service_ids: List[str]) -> np.ndarray:
        """
        使用Monitor查询每个节点的全部数值特征，固定顺序为：
        [request_cpu, request_memory, limit_cpu, limit_memory,
         cpu_usage, memory_usage, cpu_utilization, pod_num, response_time, desired_replicas]
        返回 (N, F) 的numpy数组
        
        """
        feats = []
        for sid in service_ids:
            state = monitor.get_service_state(sid)
            feats.append([
                float(state.request_cpu),
                float(state.request_memory),
                float(state.limit_cpu),
                float(state.limit_memory),
                float(state.cpu_usage),
                float(state.memory_usage),
                float(state.cpu_utilization),
                float(state.pod_num),
                float(state.response_time),
                float(state.requested_replicas),  # 第10个特征，与训练时一致
            ])
        return np.array(feats, dtype=np.float32)

    def _ensure_models(self, in_dim: int):
        if self.encoder is None:
            self.encoder = build_graph_encoder(
                encoder_type=self.cfg.encoder_type,
                in_dim=in_dim,
                hidden_dim=self.cfg.gcn_hidden_dim,
                dropout=self.cfg.dropout,
            ).to(self.device)
            self.gat = self.encoder
        if self.informer is None:
            self.informer = Informer(
                d_model=self.cfg.transformer_d_model,
                nhead=self.cfg.transformer_heads,
                num_layers=self.cfg.transformer_layers,
                dim_feedforward=self.cfg.transformer_ffn_dim,
                dropout=self.cfg.dropout,
                horizon=self.cfg.horizon,
                use_causal_mask=self.cfg.use_causal_mask,
            ).to(self.device)

        # Ensure persistent projection exists and matches dimensions if possible
        # If gat is present and has attribute out_dim, create proj accordingly
        try:
            gat_out_dim = getattr(self.encoder, "out_dim", None)
            if gat_out_dim is not None and self.proj_to_dmodel is None:
                if gat_out_dim != self.cfg.transformer_d_model:
                    self.proj_to_dmodel = torch.nn.Linear(gat_out_dim, self.cfg.transformer_d_model).to(self.device)
        except Exception:
            # ignore if GAT does not expose out_dim
            pass
        # Ensure persistent time_proj maps monitor feature dim -> d_model
        try:
            if self.time_proj is None and in_dim is not None:
                if in_dim != self.cfg.transformer_d_model:
                    self.time_proj = torch.nn.Linear(in_dim, self.cfg.transformer_d_model).to(self.device)
        except Exception:
            pass

    def analyze_topology(
        self,
        adjacency_json_path: str,
        monitor: "Monitor",
        history_node_features: Optional[List[np.ndarray]] = None,
    ) -> AnalyzeResult:
        """
        主流程：
        - 读取 A 与 节点顺序 nodes
        - 通过 Monitor 获取当前节点特征 X_cur (N, F)
        - 用 GAT 得到 Z (N, d_model)
        - 构造 Informer 输入（时间维度）
        - 输出未来H步的节点与边预测
        """
        # For backward compatibility, analyze_topology will call predict (inference) path
        return self.predict(adjacency_json_path=adjacency_json_path, monitor=monitor, history_node_features=history_node_features)

    def forward(
        self,
        adjacency_json_path: str,
        monitor: "Monitor",
        history_node_features: Optional[List[np.ndarray]] = None,
        training: bool = True,
    ) -> AnalyzeResult:
        """
        General forward pipeline. If training=True, gradients are enabled and
        modules are expected to be in train() mode. If training=False, this
        function will perform inference (no_grad) but still return the same
        AnalyzeResult structure.
        """
        adj_np, nodes = load_adjacency_with_nodes(adjacency_json_path)
        N = len(nodes)
        edge_index_pairs = adjacency_to_edge_index(adj_np)

        X_cur_np = self._build_node_features(monitor, nodes).astype(np.float32)  # (N, F)
        if training:
            # update internal history buffer only during inference/normal operation
            self.history_buffer.append(X_cur_np)
            if len(self.history_buffer) > self.history_len:
                self.history_buffer = self.history_buffer[-self.history_len:]
        else:
            # for inference, also maintain buffer
            self.history_buffer.append(X_cur_np)
            if len(self.history_buffer) > self.history_len:
                self.history_buffer = self.history_buffer[-self.history_len:]

        X_cur = X_cur_np.copy()
        in_dim = X_cur.shape[1]
        self._ensure_models(in_dim=in_dim)
        self._maybe_load_weights()

        # tensors
        A = torch.from_numpy(adj_np).float().to(self.device)                    # (N,N)
        X_tensor = torch.from_numpy(X_cur).float().to(self.device)              # (N,F)

        # Graph encoder forward: conditionally compute with or without grad
        if training:
            Z = self.encoder(X_tensor, A)
        else:
            with torch.no_grad():
                Z = self.encoder(X_tensor, A)

        # Transformer输入：时间维度
        d_model = self.cfg.transformer_d_model
        # 将GCN隐向量Z映射/裁剪到d_model维（若不等）
        if Z.size(1) != d_model:
            # 使用持久化投影层，避免每次新建导致随机权重
            if self.proj_to_dmodel is None:
                self.proj_to_dmodel = torch.nn.Linear(Z.size(1), d_model).to(self.device)
            # 若投影的输入维变化（极少见），重新初始化投影以匹配新维度
            if self.proj_to_dmodel.in_features != Z.size(1):
                self.proj_to_dmodel = torch.nn.Linear(Z.size(1), d_model).to(self.device)
            Z = self.proj_to_dmodel(Z)

        # prepare history arrays
        if history_node_features and len(history_node_features) > 0:
            history_arrays = list(history_node_features)
        else:
            history_arrays = list(self.history_buffer)
        # pad: use zero-frames (same shape) instead of repeating earliest frame
        while len(history_arrays) < self.history_len:
            zero_frame = np.zeros_like(history_arrays[0])
            history_arrays.insert(0, zero_frame)

        seq_list = []
        for arr in history_arrays:
            arr_np = np.asarray(arr, dtype=np.float32)
            arr_t = torch.from_numpy(arr_np).float().to(self.device)  # (N, F_time)
            if arr_t.size(1) != in_dim:
                raise ValueError("history节点特征维度不一致")
            # ensure time_proj exists and matches
            if self.time_proj is None:
                self.time_proj = torch.nn.Linear(arr_t.size(1), d_model).to(self.device)
            if self.time_proj.in_features != arr_t.size(1):
                self.time_proj = torch.nn.Linear(arr_t.size(1), d_model).to(self.device)
            if training:
                t_proj = self.time_proj(arr_t)  # (N, d_model)
            else:
                with torch.no_grad():
                    t_proj = self.time_proj(arr_t)
            seq_list.append(t_proj.unsqueeze(0))

        # current timestep: project X_cur dynamic features and fuse with topology embedding Z
        X_cur_t = torch.from_numpy(X_cur).float().to(self.device)
        if self.time_proj is None:
            self.time_proj = torch.nn.Linear(X_cur_t.size(1), d_model).to(self.device)
        if self.time_proj.in_features != X_cur_t.size(1):
            self.time_proj = torch.nn.Linear(X_cur_t.size(1), d_model).to(self.device)
        if training:
            cur_t_proj = self.time_proj(X_cur_t)  # (N, d_model)
        else:
            with torch.no_grad():
                cur_t_proj = self.time_proj(X_cur_t)

        # 融合策略：简单相加，使 Transformer 同时看到动态与拓扑信息
        seq_list.append((cur_t_proj + Z).unsqueeze(0))
        seq_x = torch.cat(seq_list, dim=0)  # (T, N, d_model)

        # 构造边索引与边表示
        if len(edge_index_pairs) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            ei = torch.tensor(edge_index_pairs, dtype=torch.long, device=self.device).t().contiguous()  # (2,E)
            edge_index = ei

        # Informer forward
        if training:
            node_preds, edge_preds = self.informer(seq_x, Z, edge_index)
        else:
            with torch.no_grad():
                node_preds, edge_preds = self.informer(seq_x, Z, edge_index)

        # 整理输出字典
        H = self.cfg.horizon
        node_embeddings = {nodes[i]: Z[i].detach().cpu().tolist() for i in range(N)}
        forecast_node_cpu = []
        forecast_node_pod = []
        for h in range(H):
            cpu_h: Dict[str, float] = {}
            pod_h: Dict[str, float] = {}
            for i, sid in enumerate(nodes):
                cpu_val = float(torch.sigmoid(node_preds[h, i, 0]).detach().cpu().item())  # 限制0~1
                pod_val = float(torch.relu(node_preds[h, i, 1]).detach().cpu().item())     # 非负
                cpu_h[sid] = cpu_val
                pod_h[sid] = pod_val
            forecast_node_cpu.append(cpu_h)
            forecast_node_pod.append(pod_h)

        forecast_edge_latency: List[Dict[Tuple[str, str], float]] = []
        for h in range(H):
            e_dict: Dict[Tuple[str, str], float] = {}
            if edge_index.size(1) > 0:
                e_vals = torch.relu(edge_preds[h, :, 0]).detach().cpu().numpy().tolist()
                for e_idx, (src_i, dst_i) in enumerate(edge_index_pairs):
                    # 输出单位：ms（模型输出已是非负数，这里可直接视为ms；实际可乘尺度因子）
                    e_dict[(nodes[src_i], nodes[dst_i])] = float(e_vals[e_idx])
            forecast_edge_latency.append(e_dict)

        embedding = TopologyEmbedding(
            node_embeddings=node_embeddings,
            edge_index=edge_index_pairs,
            node_order=nodes,
        )
        forecast = TopologyForecast(
            horizon=H,
            future_node_cpu_util=forecast_node_cpu,
            future_node_pod_num=forecast_node_pod,
            future_edge_latency=forecast_edge_latency,
        )
        return AnalyzeResult(embedding=embedding, forecast=forecast)

    def predict(self, adjacency_json_path: str, monitor: "Monitor", history_node_features: Optional[List[np.ndarray]] = None) -> AnalyzeResult:
        """Inference wrapper: set eval mode and run forward under no_grad."""
        # set eval
        try:
            if self.encoder is not None:
                self.encoder.eval()
            self.informer.eval()
            if self.proj_to_dmodel is not None:
                self.proj_to_dmodel.eval()
            if self.time_proj is not None:
                self.time_proj.eval()
        except Exception:
            pass
        # run forward in inference mode
        return self.forward(adjacency_json_path=adjacency_json_path, monitor=monitor, history_node_features=history_node_features, training=False)

    def _maybe_load_weights(self):
        if self.weights_loaded:
            return
        if not self.cfg.weights_path:
            self.weights_loaded = True
            return
        try:
            checkpoint = torch.load(self.cfg.weights_path, map_location=self.device)
            ckpt_config = checkpoint.get("config", {})
            ckpt_analyze_cfg = ckpt_config.get("analyze_config", {}) if isinstance(ckpt_config, dict) else {}
            ckpt_encoder_type = checkpoint.get("encoder_type") or ckpt_analyze_cfg.get("encoder_type") or "mta"
            ckpt_encoder_type = str(ckpt_encoder_type).lower()
            if ckpt_encoder_type != self.cfg.encoder_type:
                raise ValueError(
                    "Analyzer checkpoint encoder_type mismatch: "
                    f"checkpoint={ckpt_encoder_type}, requested={self.cfg.encoder_type}, "
                    f"path={self.cfg.weights_path}"
                )
            # support both old names (gcn/transformer) and new (encoder/gat/informer)
            encoder_state = checkpoint.get("encoder") or checkpoint.get("gat")
            if encoder_state is None and ckpt_encoder_type == "mta":
                encoder_state = checkpoint.get("gcn")
            if encoder_state is not None and self.encoder is not None:
                self.encoder.load_state_dict(encoder_state)
            if "informer" in checkpoint:
                if self.informer is not None:
                    self.informer.load_state_dict(checkpoint["informer"])
            elif "transformer" in checkpoint:
                if self.informer is not None:
                    self.informer.load_state_dict(checkpoint["transformer"])
            # load persistent projection if present in checkpoint
            if "proj_to_dmodel" in checkpoint and checkpoint["proj_to_dmodel"] is not None:
                try:
                    # create layer if not exists
                    proj_state = checkpoint["proj_to_dmodel"]
                    in_f = proj_state["in_features"] if "in_features" in proj_state else None
                    out_f = proj_state["out_features"] if "out_features" in proj_state else None
                    if self.proj_to_dmodel is None and in_f is not None and out_f is not None:
                        self.proj_to_dmodel = torch.nn.Linear(in_f, out_f).to(self.device)
                    if self.proj_to_dmodel is not None:
                        # assume checkpoint stored under state_dict key 'proj_state_dict'
                        if "proj_state_dict" in proj_state:
                            self.proj_to_dmodel.load_state_dict(proj_state["proj_state_dict"])
                except Exception:
                    logger.warning("无法加载 proj_to_dmodel 权重")
            self.weights_loaded = True
            logger.info("Analyzer权重已加载: %s", self.cfg.weights_path)
        except FileNotFoundError:
            logger.warning("未找到Analyzer权重文件: %s", self.cfg.weights_path)
            self.weights_loaded = True
        except Exception as exc:
            logger.warning("加载Analyzer权重失败: %s", exc)
            self.weights_loaded = True
