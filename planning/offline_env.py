"""
离线数据环境包装器，用于从 CSV 数据中训练 RL 模型
"""
import logging
from typing import Optional, Tuple, Dict
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class OfflineReplayEnv(gym.Env):
    """从离线数据 CSV 回放的环境，用于预训练
    
    适用场景：
    - 从历史监控数据中学习策略
    - 在没有真实环境交互的情况下预训练模型
    
    CSV 格式要求（参考 datasets/online_boutique_gym_observation.csv）：
    - 观测特征列（如 cpu_usage, mem_usage, latency 等）
    - 动作列（可选，用于行为克隆）
    - 奖励会根据系统指标自动计算
    - done 标记会自动生成（每 max_episode_steps 划分一个 episode）
    """
    
    def __init__(self, data_path: str, obs_columns: list, action_dim: int,
                 service_ids: list = None,
                 reward_column: str = "reward", done_column: str = "done",
                 shuffle: bool = False, max_episode_steps: int = 100,
                 latency_threshold_ms: float = 3000.0,
                 reward_weights: dict = None):
        """
        Args:
            data_path: CSV 文件路径
            obs_columns: 观测列名列表（按顺序）
            action_dim: 动作空间维度
            service_ids: 服务 ID 列表（用于计算奖励）
            reward_column: 奖励列名（如果 CSV 中有，会优先使用；否则会计算）
            done_column: done 标志列名（如果 CSV 中没有会自动生成）
            shuffle: 是否打乱数据顺序
            max_episode_steps: 最大 episode 长度
            latency_threshold_ms: 延迟阈值（用于计算奖励）
            reward_weights: 奖励权重字典 {'w_c': 0.3, 'w_l': 0.4, 'w_r': 0.3}
        """
        super().__init__()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Offline data file not found: {data_path}")
        
        # 加载数据
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        self.obs_columns = obs_columns
        self.service_ids = service_ids or []
        self.reward_column = reward_column
        self.done_column = done_column
        self.max_episode_steps = max_episode_steps
        self.latency_threshold_ms = latency_threshold_ms
        
        # 奖励权重（默认值）
        if reward_weights is None:
            reward_weights = {'w_c': 0.3, 'w_l': 0.4, 'w_r': 0.3}
        self.reward_weights = reward_weights
        
        # 验证列是否存在
        missing_cols = [c for c in obs_columns if c not in self.data.columns]
        if missing_cols:
            logger.warning(f"Missing observation columns: {missing_cols}")
            # 用零填充缺失列
            for col in missing_cols:
                self.data[col] = 0.0
        
        # 计算奖励列（如果 CSV 中没有）
        if reward_column not in self.data.columns:
            logger.info(f"Reward column '{reward_column}' not found, computing from metrics...")
            self.data[reward_column] = self._compute_rewards()
        else:
            logger.info(f"Using existing reward column '{reward_column}' from CSV")
        
        # 生成 done 列（如果 CSV 中没有）
        # done 的作用：标记 episode 结束，用于：
        # 1. 环境 reset 的时机
        # 2. 折扣奖励计算的截断点
        # 3. 防止跨 episode 的错误关联
        if done_column not in self.data.columns:
            logger.info(f"Done column '{done_column}' not found, auto-generating (every {max_episode_steps} steps)...")
            self.data[done_column] = False
            # 每 max_episode_steps 标记一次 done
            for i in range(max_episode_steps - 1, len(self.data), max_episode_steps):
                self.data.at[i, done_column] = True
            # 最后一行也标记为 done
            if len(self.data) > 0:
                self.data.at[len(self.data) - 1, done_column] = True
            logger.info(f"Generated {self.data[done_column].sum()} episode boundaries")
        else:
            logger.info(f"Using existing done column '{done_column}' from CSV")
        
        if shuffle:
            self.data = self.data.sample(frac=1.0).reset_index(drop=True)
            logger.info("Shuffled offline data")
        
        self.current_idx = 0
        self.episode_step = 0
        
        # 定义观测和动作空间
        obs_dim = len(obs_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 假设离散动作空间（MultiDiscrete）
        # 如果是连续动作，可以改为 Box
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        logger.info(f"OfflineReplayEnv: obs_dim={obs_dim}, action_dim={action_dim}, samples={len(self.data)}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # 从数据开始处或随机位置开始新 episode
        if seed is not None:
            np.random.seed(seed)
        
        # 循环回放
        if self.current_idx >= len(self.data):
            self.current_idx = 0
        
        self.episode_step = 0
        obs = self._get_observation(self.current_idx)
        return obs, {}
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步（忽略 action，直接回放下一步）"""
        # 注意：这里的 action 被忽略，因为我们只是回放历史数据
        # 如果需要做行为克隆，可以记录 action 并计算 BC loss
        
        self.current_idx += 1
        self.episode_step += 1
        
        # 检查是否到达数据末尾
        if self.current_idx >= len(self.data):
            terminated = True
            truncated = False
            self.current_idx = 0  # 循环
            obs = self._get_observation(0)
            reward = 0.0
        else:
            row = self.data.iloc[self.current_idx]
            obs = self._get_observation(self.current_idx)
            reward = float(row[self.reward_column])
            terminated = bool(row[self.done_column])
            truncated = self.episode_step >= self.max_episode_steps
        
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """从数据中提取观测"""
        row = self.data.iloc[idx]
        obs = []
        for col in self.obs_columns:
            val = row[col]
            # 处理可能的 NaN
            if pd.isna(val):
                val = 0.0
            obs.append(float(val))
        return np.array(obs, dtype=np.float32)
    
    def _compute_rewards(self) -> pd.Series:
        """根据系统指标计算奖励
        
        奖励公式（参考 kutara/planning/env.py）：
        R = w_c * r_c + w_l * r_l + w_r * r_r
        
        其中：
        - r_c: 成本项（pod 数越少越好）
        - r_l: 延迟项（延迟低于阈值奖励高）
        - r_r: 资源利用项（CPU/内存利用率适中最好）
        
        注意：
        - CPU 单位：毫核（millicores），1000m = 1 core
        - 内存单位：Mi（兆字节），1024Mi = 1GB
        - 延迟单位：毫秒（ms）
        
        Returns:
            pd.Series: 每一行的奖励值
        """
        logger.info("Computing rewards from system metrics...")
        logger.info(f"Using latency threshold: {self.latency_threshold_ms} ms")
        
        rewards = []
        w_c = self.reward_weights.get('w_c', 0.3)
        w_l = self.reward_weights.get('w_l', 0.4)
        w_r = self.reward_weights.get('w_r', 0.3)
        
        for idx, row in self.data.iterrows():
            # 1. 成本项：pod 数越少越好（归一化到 [0, 1]）
            # r_c = 1 - (actual_pods / desired_pods)
            total_actual_pods = 0
            total_desired_pods = 0
            
            for sid in self.service_ids:
                actual_pod_col = f"{sid}_num_pods"
                desired_pod_col = f"{sid}_desired_replicas"
                
                if actual_pod_col in self.data.columns:
                    actual_pods = row.get(actual_pod_col, 0)
                    total_actual_pods += float(actual_pods) if not pd.isna(actual_pods) else 0
                
                if desired_pod_col in self.data.columns:
                    desired_pods = row.get(desired_pod_col, 0)
                    total_desired_pods += float(desired_pods) if not pd.isna(desired_pods) else 0
            
            # 避免除以零
            if total_desired_pods > 0:
                r_c = 1.0 - min(1.0, total_actual_pods / total_desired_pods)
            else:
                r_c = 0.0
            
            # 2. 延迟项：延迟越低越好
            # 使用所有服务的平均延迟
            latencies = []
            for sid in self.service_ids:
                lat_col = f"{sid}_latency"
                if lat_col in self.data.columns:
                    lat = row.get(lat_col, 0)
                    if not pd.isna(lat):
                        latencies.append(float(lat))
            
            avg_latency = np.mean(latencies) if latencies else 0.0
            # 延迟奖励：使用密集奖励 r_l = max(0, 1 - latency / threshold)
            r_l = max(0.0, 1.0 - avg_latency / self.latency_threshold_ms)
            
            # 3. 资源利用项：CPU 和内存利用率适中最好
            # CPU 单位：毫核（millicores），每 pod 分配 1 核 = 1000m
            # 内存单位：Mi（兆字节），每 pod 分配 2Gi = 2048Mi
            # 目标利用率：60-80%
            cpu_utils = []
            mem_utils = []
            
            for sid in self.service_ids:
                cpu_col = f"{sid}_cpu_usage"
                mem_col = f"{sid}_mem_usage"
                
                if cpu_col in self.data.columns:
                    cpu_millicores = row.get(cpu_col, 0)
                    if not pd.isna(cpu_millicores):
                        # 转换为百分比：每 pod 分配 1000m
                        # cpu_util% = (millicores / 1000) * 100 = millicores / 10
                        cpu_util = float(cpu_millicores) / 10.0  # 368m → 36.8%
                        cpu_util = min(100.0, max(0.0, cpu_util))  # 限制在 0-100
                        cpu_utils.append(cpu_util)
                
                if mem_col in self.data.columns:
                    mem_mi = row.get(mem_col, 0)
                    if not pd.isna(mem_mi):
                        # 转换为百分比：每 pod 分配 2048Mi (2Gi)
                        # mem_util% = (Mi / 2048) * 100
                        mem_util = (float(mem_mi) / 2048.0) * 100.0  # 1024Mi → 50%
                        mem_util = min(100.0, max(0.0, mem_util))
                        mem_utils.append(mem_util)
            
            avg_cpu = np.mean(cpu_utils) if cpu_utils else 50.0
            avg_mem = np.mean(mem_utils) if mem_utils else 50.0
            
            # 资源利用率奖励：目标是 60-80%，偏离惩罚
            def util_reward(util):
                """
                利用率奖励函数：
                - 60-80%: 最优（奖励 1.0）
                - < 60%: 资源浪费，线性惩罚
                - > 80%: 资源紧张，线性惩罚
                """
                if 60 <= util <= 80:
                    return 1.0
                elif util < 60:
                    return max(0.0, util / 60.0)  # 0% → 0.0, 60% → 1.0
                else:  # util > 80
                    return max(0.0, 1.0 - (util - 80) / 20.0)  # 80% → 1.0, 100% → 0.0
            
            r_r = 0.5 * util_reward(avg_cpu) + 0.5 * util_reward(avg_mem)
            
            # 总奖励
            total_reward = w_c * r_c + w_l * r_l + w_r * r_r
            rewards.append(total_reward)
        
        rewards_array = np.array(rewards)
        logger.info(f"Computed rewards: mean={rewards_array.mean():.4f}, std={rewards_array.std():.4f}, "
                   f"min={rewards_array.min():.4f}, max={rewards_array.max():.4f}")
        logger.info(f"Reward components contribution:")
        logger.info(f"  - Latency weight: {w_l:.2f}")
        logger.info(f"  - Resource weight: {w_r:.2f}")
        logger.info(f"  - Cost weight: {w_c:.2f}")
        
        # 统计各分量
        high_rewards = (rewards_array > 0.7).sum()
        low_rewards = (rewards_array < 0.3).sum()
        logger.info(f"Reward distribution: {high_rewards} high (>0.7), {low_rewards} low (<0.3) out of {len(rewards_array)}")
        
        return pd.Series(rewards, index=self.data.index)
    
    def load_to_replay_buffer(self, replay_buffer, max_samples: Optional[int] = None):
        """将离线数据加载到 SAC 的 replay buffer
        
        Args:
            replay_buffer: SAC model 的 replay_buffer
            max_samples: 最多加载多少样本（None 表示全部）
        """
        logger.info(f"Loading offline data into replay buffer...")
        
        num_samples = min(len(self.data) - 1, max_samples) if max_samples else len(self.data) - 1
        loaded = 0
        
        for idx in range(num_samples):
            obs = self._get_observation(idx)
            next_obs = self._get_observation(idx + 1)
            
            # 动作：这里用零向量占位（因为离线数据可能没有动作）
            # 如果 CSV 中有动作列，可以从中提取
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            
            reward = float(self.data.iloc[idx][self.reward_column])
            done = bool(self.data.iloc[idx][self.done_column])
            
            # 添加到 replay buffer
            # 注意：不同版本的 SB3 API 可能略有不同
            try:
                replay_buffer.add(obs, next_obs, action, reward, done, [{}])
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to add sample {idx} to buffer: {e}")
                break
        
        logger.info(f"Loaded {loaded} samples into replay buffer (size={replay_buffer.size()})")


def create_offline_env_from_csv(csv_path: str, service_ids: list, 
                                obs_features_per_service: list = None,
                                action_dim: int = 2,
                                max_episode_steps: int = 100,
                                latency_threshold_ms: float = 3000.0,
                                reward_weights: dict = None) -> OfflineReplayEnv:
    """从 CSV 创建离线环境的便捷函数
    
    Args:
        csv_path: CSV 文件路径
        service_ids: 服务 ID 列表
        obs_features_per_service: 每个服务的观测特征列表（如 ['num_pods', 'cpu_usage']）
        action_dim: 动作维度
        max_episode_steps: 最大 episode 长度
        latency_threshold_ms: 延迟阈值（用于奖励计算）
        reward_weights: 奖励权重 {'w_c': 0.3, 'w_l': 0.4, 'w_r': 0.3}
    
    Returns:
        OfflineReplayEnv 实例
    
    注意：
        CSV 列名格式应为 {service_id}_{feature}，例如：
        - frontend_num_pods
        - frontend_cpu_usage
        - cartservice_latency
        
        奖励会根据 num_pods, cpu_usage, mem_usage, latency 自动计算
        done 标记会每 max_episode_steps 自动生成
    """
    if obs_features_per_service is None:
        obs_features_per_service = [
            'num_pods', 'desired_replicas', 'cpu_usage', 
            'mem_usage', 'latency'
        ]
    
    # 构建观测列名（CSV 中列名格式为 service_id_feature，用下划线连接）
    obs_columns = []
    for sid in service_ids:
        for feat in obs_features_per_service:
            col = f"{sid}_{feat}"
            obs_columns.append(col)
    
    env = OfflineReplayEnv(
        data_path=csv_path,
        obs_columns=obs_columns,
        action_dim=action_dim,
        service_ids=service_ids,
        max_episode_steps=max_episode_steps,
        latency_threshold_ms=latency_threshold_ms,
        reward_weights=reward_weights
    )
    
    return env
