"""
改进的离线环境：基于查表的模拟器
借鉴开源项目的思路，让动作能影响状态转移
"""
import logging
from typing import Optional, Tuple, Dict
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class LookupBasedOfflineEnv(gym.Env):
    """
    基于查表的离线环境模拟器
    
    核心思想：
    1. Agent 执行动作 → 改变 num_pods
    2. 从CSV中查找匹配 num_pods 的历史记录
    3. 用查找到的记录作为 next_state
    
    这样动作就能"影响"状态转移（虽然是近似的）
    """
    
    def __init__(
        self, 
        data_path: str,
        service_ids: list,
        deltas: list = None,
        latency_threshold_ms: float = 3000.0,
        reward_weights: dict = None,
        max_episode_steps: int = 100,
    ):
        """
        Args:
            data_path: CSV 文件路径
            service_ids: 服务列表（按顺序）
            deltas: 允许的副本变化量
            latency_threshold_ms: 延迟阈值
            reward_weights: 奖励权重
            max_episode_steps: 最大步数
        """
        super().__init__()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # 加载数据
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        self.service_ids = service_ids
        self.N = len(service_ids)
        self.deltas = np.array(deltas or [-5, -3, -2, -1, 0, 1, 2, 3, 5], dtype=np.int32)
        self.latency_threshold_ms = latency_threshold_ms
        self.max_episode_steps = max_episode_steps
        
        # 奖励权重
        if reward_weights is None:
            reward_weights = {'w_c': 0.3, 'w_l': 0.4, 'w_r': 0.3}
        self.reward_weights = reward_weights
        
        # 当前状态（每个服务的副本数）
        self.current_pods = np.ones(self.N, dtype=np.int32)
        self.previous_pods = np.ones(self.N, dtype=np.int32)
        self.current_step = 0
        
        # 预计算 diff 列（加速查找）
        logger.info("Precomputing diff columns...")
        for svc_id in service_ids:
            col_name = f'{svc_id}_num_pods'
            if col_name in self.data.columns:
                self.data[f'diff_{svc_id}'] = self.data[col_name].diff().fillna(0)
        
        # 定义空间
        self.action_space = spaces.MultiDiscrete([self.N, len(self.deltas)])
        
        # 观测空间：每个服务5个基础特征
        obs_dim = self.N * 5  # num_pods, cpu, mem, traffic_in, traffic_out
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        logger.info(f"LookupBasedOfflineEnv: services={self.N}, obs_dim={obs_dim}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # 随机选择一个初始状态
        sample = self.data.sample(n=1).iloc[0]
        
        for i, svc_id in enumerate(self.service_ids):
            col_name = f'{svc_id}_num_pods'
            if col_name in self.data.columns:
                self.current_pods[i] = int(sample[col_name])
                self.previous_pods[i] = int(sample[col_name])
            else:
                self.current_pods[i] = 1
                self.previous_pods[i] = 1
        
        self.current_step = 0
        obs = self._build_observation(sample)
        
        return obs, {}
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作并查找下一个状态
        """
        self.current_step += 1
        
        # 解析动作
        service_idx = int(action[0])
        delta_idx = int(action[1])
        delta = int(self.deltas[delta_idx])
        
        # 保存之前的状态
        self.previous_pods = self.current_pods.copy()
        
        # 应用动作（改变副本数）
        new_pods = self.current_pods[service_idx] + delta
        new_pods = np.clip(new_pods, 1, 8)  # 限制在 [1, 8]
        self.current_pods[service_idx] = new_pods
        
        # 从CSV中查找匹配的状态
        next_sample = self._lookup_state()
        
        # 构建观测
        obs = self._build_observation(next_sample)
        
        # 计算奖励
        reward = self._compute_reward(next_sample)
        
        # 检查是否结束
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _lookup_state(self) -> pd.Series:
        """
        从CSV中查找匹配当前状态的记录
        
        查找策略：
        1. 尝试精确匹配：num_pods 和 diff 都匹配
        2. 如果没找到，放宽到只匹配 num_pods
        3. 如果还没找到，随机返回一条
        """
        # 计算变化量
        diff = self.current_pods - self.previous_pods
        
        # 构建查询条件
        mask = pd.Series([True] * len(self.data))
        
        for i, svc_id in enumerate(self.service_ids):
            col_pods = f'{svc_id}_num_pods'
            col_diff = f'diff_{svc_id}'
            
            if col_pods in self.data.columns:
                # 先尝试匹配 num_pods
                mask &= (self.data[col_pods] == self.current_pods[i])
        
        # 第一次尝试：匹配 num_pods
        candidates = self.data[mask].copy()
        
        if len(candidates) > 0:
            # 进一步尝试匹配 diff（如果有的话）
            # 重置索引以避免索引对齐问题
            candidates_reset = candidates.reset_index(drop=True)
            diff_mask = pd.Series([True] * len(candidates_reset))
            for i, svc_id in enumerate(self.service_ids):
                col_diff = f'diff_{svc_id}'
                if col_diff in candidates_reset.columns and diff[i] != 0:
                    diff_mask &= (candidates_reset[col_diff] == diff[i])
            
            diff_candidates = candidates_reset[diff_mask]
            if len(diff_candidates) > 0:
                return diff_candidates.sample(n=1).iloc[0]
            else:
                # 只匹配 num_pods
                return candidates_reset.sample(n=1).iloc[0]
        else:
            # 没有匹配的，随机返回一条（记录为debug级别）
            logger.debug(f"No matching state found for pods={self.current_pods}, using random sample")
            return self.data.sample(n=1).iloc[0]
    
    def _build_observation(self, sample: pd.Series) -> np.ndarray:
        """
        从CSV记录构建观测向量
        """
        obs = []
        for svc_id in self.service_ids:
            # 基础5个特征
            obs.append(float(sample.get(f'{svc_id}_num_pods', 1.0)))
            obs.append(float(sample.get(f'{svc_id}_cpu_usage', 0.0)))
            obs.append(float(sample.get(f'{svc_id}_mem_usage', 0.0)))
            obs.append(float(sample.get(f'{svc_id}_traffic_in', 0.0)))
            obs.append(float(sample.get(f'{svc_id}_traffic_out', 0.0)))
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, sample: pd.Series) -> float:
        """
        根据系统指标计算奖励
        """
        w_c = self.reward_weights['w_c']
        w_l = self.reward_weights['w_l']
        w_r = self.reward_weights['w_r']
        
        # 成本奖励（pod数越少越好）
        total_pods = sum([sample.get(f'{svc_id}_num_pods', 1) for svc_id in self.service_ids])
        max_pods = len(self.service_ids) * 8
        r_c = 1.0 - (total_pods / max_pods)
        
        # 延迟奖励（延迟低于阈值）
        latencies = [sample.get(f'{svc_id}_latency', 0.0) for svc_id in self.service_ids]
        avg_latency = np.mean([l for l in latencies if l > 0] or [0])
        r_l = max(0, 1.0 - avg_latency / self.latency_threshold_ms)
        
        # 资源利用奖励（CPU/内存在60-80%最优）
        cpus = [sample.get(f'{svc_id}_cpu_usage', 0.0) for svc_id in self.service_ids]
        avg_cpu = np.mean(cpus) if cpus else 0
        target_util = 0.7  # 目标70%
        r_r = 1.0 - abs(avg_cpu/1000.0 - target_util)
        
        reward = w_c * r_c + w_l * r_l + w_r * r_r
        
        return reward
