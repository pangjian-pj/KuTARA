import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

from .analyzer import AnalyzeConfig
from .loader import load_adjacency_with_nodes, adjacency_to_edge_index
from .models import Informer, build_graph_encoder

logger = logging.getLogger(__name__)


BASE_CPU_REQUEST_PER_POD = 0.1  # cores
BASE_MEMORY_REQUEST_PER_POD = 128.0  # Mi


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        cpu_utils: np.ndarray,
        pod_counts: np.ndarray,
        history_len: int,
        horizon: int,
    ):
        """
        features: (T, N, F)
        cpu_utils: (T, N)
        pod_counts: (T, N)
        """
        assert features.shape[0] == cpu_utils.shape[0] == pod_counts.shape[0]
        self.features = features.astype(np.float32)
        self.cpu_utils = np.clip(cpu_utils.astype(np.float32), 0.0, 1.5)
        self.pod_counts = pod_counts.astype(np.float32)
        self.history_len = history_len
        self.horizon = horizon
        self.indices = [
            idx
            for idx in range(history_len - 1, features.shape[0] - horizon)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        hist = self.features[t - self.history_len + 1 : t + 1]  # (history, N, F)
        future_cpu = self.cpu_utils[t + 1 : t + 1 + self.horizon]  # (H, N)
        future_pod = self.pod_counts[t + 1 : t + 1 + self.horizon]  # (H, N)
        return (
            torch.from_numpy(hist),
            torch.from_numpy(future_cpu),
            torch.from_numpy(future_pod),
        )


@dataclass
class TrainerConfig:
    csv_path: str
    adjacency_path: str
    save_path: str = "models/analyzer_weights.pt"
    history_len: int = 8
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    # device selection: 'auto' | 'cpu' | 'cuda' | 'mps'
    device: str = "auto"
    # fraction of data to reserve for testing (time-based split on dataset samples)
    test_split: float = 0.2

    # model hyperparameter overrides (optional)
    horizon: Optional[int] = None
    gcn_hidden_dim: Optional[int] = None
    transformer_d_model: Optional[int] = None
    transformer_heads: Optional[int] = None
    transformer_layers: Optional[int] = None
    transformer_ffn_dim: Optional[int] = None
    dropout: Optional[float] = None
    use_causal_mask: Optional[bool] = None
    max_samples: Optional[int] = None


class AnalyzerTrainer:
    """
    使用历史指标数据对Analyzer中的图编码器+Informer进行离线训练。
    """

    def __init__(self, analyze_config: AnalyzeConfig, trainer_config: TrainerConfig):
        self.analyze_cfg = analyze_config
        self.trainer_cfg = trainer_config

        # device selection
        requested_device = (trainer_config.device or analyze_config.device)
        if requested_device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)
        logger.info(f"使用设备: {self.device}")

        self.analyze_cfg.device = str(self.device)
        self.analyze_cfg.history_length = trainer_config.history_len

        # propagate trainer-specified model overrides
        if trainer_config.horizon is not None:
            self.analyze_cfg.horizon = trainer_config.horizon
        if trainer_config.gcn_hidden_dim is not None:
            self.analyze_cfg.gcn_hidden_dim = trainer_config.gcn_hidden_dim
        if trainer_config.transformer_d_model is not None:
            self.analyze_cfg.transformer_d_model = trainer_config.transformer_d_model
        if trainer_config.transformer_heads is not None:
            self.analyze_cfg.transformer_heads = trainer_config.transformer_heads
        if trainer_config.transformer_layers is not None:
            self.analyze_cfg.transformer_layers = trainer_config.transformer_layers
        if trainer_config.transformer_ffn_dim is not None:
            self.analyze_cfg.transformer_ffn_dim = trainer_config.transformer_ffn_dim
        if trainer_config.dropout is not None:
            self.analyze_cfg.dropout = trainer_config.dropout
        if trainer_config.use_causal_mask is not None:
            self.analyze_cfg.use_causal_mask = trainer_config.use_causal_mask

        (
            self.adj_matrix,
            self.node_order,
        ) = load_adjacency_with_nodes(trainer_config.adjacency_path)
        self.edge_index_pairs = adjacency_to_edge_index(self.adj_matrix)
        self.edge_index_tensor = (
            torch.tensor(self.edge_index_pairs, dtype=torch.long).t().contiguous()
            if self.edge_index_pairs
            else torch.zeros((2, 0), dtype=torch.long)
        ).to(self.device)

        (
            self.features,
            self.cpu_utils,
            self.pod_counts,
        ) = self._load_dataset(trainer_config.csv_path, self.node_order)

        self.dataset = TimeSeriesDataset(
            self.features,
            self.cpu_utils,
            self.pod_counts,
            history_len=trainer_config.history_len,
            horizon=self.analyze_cfg.horizon,
        )
        if trainer_config.max_samples is not None:
            self.dataset.indices = self.dataset.indices[: max(0, trainer_config.max_samples)]
        # split dataset into train/test by sample indices (time-based split)
        dataset_len = len(self.dataset)
        split_frac = getattr(trainer_config, "test_split", 0.2)
        split_idx = int(dataset_len * (1.0 - split_frac))
        from torch.utils.data import Subset
        train_indices = list(range(0, max(0, split_idx)))
        test_indices = list(range(max(0, split_idx), dataset_len))
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim = self.features.shape[-1]
        self.encoder = build_graph_encoder(
            encoder_type=self.analyze_cfg.encoder_type,
            in_dim=input_dim,
            hidden_dim=self.analyze_cfg.gcn_hidden_dim,
            dropout=self.analyze_cfg.dropout,
        ).to(self.device)
        self.gat = self.encoder
        self.informer = Informer(
            d_model=self.analyze_cfg.transformer_d_model,
            nhead=self.analyze_cfg.transformer_heads,
            num_layers=self.analyze_cfg.transformer_layers,
            dim_feedforward=self.analyze_cfg.transformer_ffn_dim,
            dropout=self.analyze_cfg.dropout,
            horizon=self.analyze_cfg.horizon,
            use_causal_mask=self.analyze_cfg.use_causal_mask,
        ).to(self.device)

        # 我们使用Adam优化器同时优化图编码器和Informer的参数
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.informer.parameters()),
            lr=trainer_config.learning_rate,
        )
        # 计算真实值和预测值之间的均方误差
        self.mse_loss = nn.MSELoss()

    def _load_dataset(
        self, csv_path: str, node_order: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info("加载历史数据集: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        features = []
        cpu_utils = []
        pod_counts = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            service_features = []
            service_cpu_utils = []
            service_pods = []
            for svc in node_order:
                prefix = svc
                num_pods = float(row_dict.get(f"{prefix}_num_pods", 0.0))
                desired = float(row_dict.get(f"{prefix}_desired_replicas", max(num_pods, 1.0)))
                cpu_usage_raw = float(row_dict.get(f"{prefix}_cpu_usage", 0.0))  # assumed mCPU
                mem_usage = float(row_dict.get(f"{prefix}_mem_usage", 0.0))
                latency = float(row_dict.get(f"{prefix}_latency", 0.0))

                cpu_usage = cpu_usage_raw / 1000.0  # mCPU -> cores
                request_cpu = max(desired, 1.0) * BASE_CPU_REQUEST_PER_POD
                limit_cpu = request_cpu * 1.5
                request_memory = max(desired, 1.0) * BASE_MEMORY_REQUEST_PER_POD
                limit_memory = request_memory * 1.5
                cpu_utilization = cpu_usage / request_cpu if request_cpu > 0 else 0.0
                # 计算 CPU 利用率并限制在 0~1.5 之间
                cpu_utilization = float(np.clip(cpu_utilization, 0.0, 1.5))

                # 每个节点 10 维特征，这些特征会作为模型的输入
                feature_vec = np.array(
                    [
                        request_cpu,
                        request_memory,
                        limit_cpu,
                        limit_memory,
                        cpu_usage,
                        mem_usage,
                        cpu_utilization,
                        num_pods,
                        latency,
                        desired
                    ],
                    dtype=np.float32,
                )
                service_features.append(feature_vec)
                service_cpu_utils.append(cpu_utilization)
                service_pods.append(num_pods)

            features.append(np.stack(service_features, axis=0))
            cpu_utils.append(np.array(service_cpu_utils, dtype=np.float32))
            pod_counts.append(np.array(service_pods, dtype=np.float32))

        features_arr = np.stack(features, axis=0)
        cpu_utils_arr = np.stack(cpu_utils, axis=0)
        pod_counts_arr = np.stack(pod_counts, axis=0)
        logger.info(
            "数据集加载完成: 样本数=%d, 服务数=%d, 特征维度=%d",
            features_arr.shape[0],
            features_arr.shape[1],
            features_arr.shape[2],
        )
        return features_arr, cpu_utils_arr, pod_counts_arr

    def train(self):
        logger.info(
            "开始训练Analyzer模型 (encoder=%s, epochs=%d, batch_size=%d, history_len=%d, horizon=%d)",
            self.analyze_cfg.encoder_type,
            self.trainer_cfg.epochs,
            self.trainer_cfg.batch_size,
            self.trainer_cfg.history_len,
            self.analyze_cfg.horizon,
        )
        self.encoder.train()
        self.informer.train()
        # 将 numpy 格式的图邻接矩阵转换成 torch tensor，放到 GPU/CPU 上
        adjacency = torch.from_numpy(self.adj_matrix).float().to(self.device)
        # prepare loss logging
        loss_records = []  # list of (epoch, avg_loss)

        for epoch in range(1, self.trainer_cfg.epochs + 1):
            epoch_loss = 0.0
            for hist, fut_cpu, fut_pod in tqdm(
                self.dataloader, desc=f"Epoch {epoch}/{self.trainer_cfg.epochs}"
            ):
                hist = hist.to(self.device)  # (B, history, N, F)
                fut_cpu = fut_cpu.to(self.device)
                fut_pod = fut_pod.to(self.device)

                batch_loss = 0.0
                self.optimizer.zero_grad()
                for b in range(hist.size(0)):
                    hist_b = hist[b]  # (history, N, F)
                    fut_cpu_b = fut_cpu[b]  # (H, N)
                    fut_pod_b = fut_pod[b]  # (H, N)

                    z_seq = []
                    for t in range(hist_b.size(0)):
                        z_t = self.encoder(hist_b[t], adjacency)
                        z_seq.append(z_t.unsqueeze(0))
                    seq_embeddings = torch.cat(z_seq, dim=0)  # (history, N, d_model)
                    current_embedding = seq_embeddings[-1]

                    node_preds, _ = self.informer(
                        seq_embeddings, current_embedding, self.edge_index_tensor
                    )
                    cpu_pred = torch.sigmoid(node_preds[:, :, 0])  # (H, N)
                    pod_pred = torch.relu(node_preds[:, :, 1])  # (H, N)

                    cpu_loss = self.mse_loss(cpu_pred, fut_cpu_b)
                    pod_loss = self.mse_loss(pod_pred, fut_pod_b)
                    sample_loss = cpu_loss + pod_loss
                    sample_loss.backward()
                    batch_loss += sample_loss.item()
                # 完成一个batch的梯度更新
                self.optimizer.step()
                epoch_loss += batch_loss / hist.size(0)

            avg_loss = epoch_loss / len(self.dataloader)
            logger.info("Epoch %d 平均损失: %.6f", epoch, avg_loss)
            loss_records.append({"epoch": epoch, "avg_loss": float(avg_loss)})

        # 保存平均损失到 CSV，放在 models 目录下
        try:
            os.makedirs(os.path.dirname(self.trainer_cfg.save_path), exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_name = str(self.device).replace(":", "-")
            loss_csv = os.path.join(os.path.dirname(self.trainer_cfg.save_path), f"avg_loss_{device_name}_{ts}.csv")
            pd.DataFrame(loss_records).to_csv(loss_csv, index=False)
            logger.info("Avg loss saved to %s", loss_csv)
        except Exception:
            logger.exception("保存平均损失CSV失败")

        self._save_model(timestamp_suffix=True)
        self._save_model(timestamp_suffix=False)

        # After training, run evaluation on test split if available
        try:
            if len(self.test_dataset) > 0:
                eval_metrics = self.evaluate()
                logger.info("Test evaluation results: %s", eval_metrics)
        except Exception:
            logger.exception("测试评估失败")

    def _save_model(self, timestamp_suffix: bool = False):
        os.makedirs(os.path.dirname(self.trainer_cfg.save_path), exist_ok=True)
        target_path = self.trainer_cfg.save_path
        if timestamp_suffix:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(self.trainer_cfg.save_path)
            target_path = f"{base}_{ts}{ext}"

        torch.save(
            {
                # new names
                "encoder_type": self.analyze_cfg.encoder_type,
                "encoder": self.encoder.state_dict(),
                "gat": self.encoder.state_dict(),
                "informer": self.informer.state_dict(),
                # legacy names for compatibility
                "gcn": self.encoder.state_dict(),
                "transformer": self.informer.state_dict(),
                "config": {
                    "analyze_config": self.analyze_cfg.__dict__,
                    "trainer_config": self.trainer_cfg.__dict__,
                    "node_order": self.node_order,
                },
            },
            target_path,
        )
        logger.info("Analyzer模型权重已保存到 %s", target_path)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on the test dataset and return metrics (MSE for cpu and pod)."""
        self.encoder.eval()
        self.informer.eval()
        adjacency = torch.from_numpy(self.adj_matrix).float().to(self.device)
        total_cpu_loss = 0.0
        total_pod_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for hist, fut_cpu, fut_pod in tqdm(self.test_loader, desc="Evaluating"):
                hist = hist.to(self.device)  # (B, history, N, F)
                fut_cpu = fut_cpu.to(self.device)
                fut_pod = fut_pod.to(self.device)

                for b in range(hist.size(0)):
                    hist_b = hist[b]
                    fut_cpu_b = fut_cpu[b]
                    fut_pod_b = fut_pod[b]

                    z_seq = []
                    for t in range(hist_b.size(0)):
                        z_t = self.encoder(hist_b[t], adjacency)
                        z_seq.append(z_t.unsqueeze(0))
                    seq_embeddings = torch.cat(z_seq, dim=0)
                    current_embedding = seq_embeddings[-1]

                    node_preds, _ = self.informer(seq_embeddings, current_embedding, self.edge_index_tensor)
                    cpu_pred = torch.sigmoid(node_preds[:, :, 0])
                    pod_pred = torch.relu(node_preds[:, :, 1])

                    cpu_loss = self.mse_loss(cpu_pred, fut_cpu_b)
                    pod_loss = self.mse_loss(pod_pred, fut_pod_b)
                    total_cpu_loss += cpu_loss.item()
                    total_pod_loss += pod_loss.item()
                    total_samples += 1

        avg_cpu_mse = total_cpu_loss / max(1, total_samples)
        avg_pod_mse = total_pod_loss / max(1, total_samples)
        return {"cpu_mse": avg_cpu_mse, "pod_mse": avg_pod_mse, "samples": total_samples}
