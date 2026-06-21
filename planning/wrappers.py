"""
Gym environment wrappers for RL training
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """
    将连续动作空间 Box(2,) 转换为离散动作空间 MultiDiscrete([n_services, n_deltas])
    
    用于 SAC 等连续动作算法与离散动作环境的适配。
    
    映射规则：
    - action[0] ∈ [0, 1] → service_idx ∈ [0, n_services-1]
    - action[1] ∈ [0, 1] → delta_idx ∈ [0, n_deltas-1]
    
    示例：
        假设有 11 个服务，9 种副本变化 [-4, -3, -2, -1, 0, +1, +2, +3, +4]
        SAC 输出: [0.73, 0.42]
        转换为: service_idx=8 (0.73*11≈8), delta_idx=3 (0.42*9≈3.78→3，对应-1)
    """
    
    def __init__(self, env: gym.Env):
        """
        Args:
            env: 原始环境（使用 MultiDiscrete 动作空间）
        """
        super().__init__(env)
        
        # 检查原始环境是否使用 MultiDiscrete
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError(
                f"ContinuousToDiscreteActionWrapper expects MultiDiscrete action space, "
                f"but got {type(env.action_space)}"
            )
        
        # 获取原始离散空间的维度
        self.discrete_nvec = env.action_space.nvec
        self.n_services = int(self.discrete_nvec[0])
        self.n_deltas = int(self.discrete_nvec[1])
        
        # 将动作空间改为连续空间 Box([0, 0], [1, 1])
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 存储原始动作空间用于验证
        self._original_action_space = env.action_space
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        将连续动作转换为离散动作
        
        Args:
            action: SAC 输出的连续动作 [a0, a1]，范围 [0, 1]
        
        Returns:
            离散动作 [service_idx, delta_idx]
        """
        # 确保 action 是 numpy 数组
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 限制到 [0, 1] 范围（防止数值误差）
        action = np.clip(action, 0.0, 1.0)
        
        # 映射到离散索引
        # service_idx: [0, 1) → [0, n_services)
        service_idx = int(action[0] * self.n_services)
        service_idx = np.clip(service_idx, 0, self.n_services - 1)
        
        # delta_idx: [0, 1) → [0, n_deltas)
        delta_idx = int(action[1] * self.n_deltas)
        delta_idx = np.clip(delta_idx, 0, self.n_deltas - 1)
        
        discrete_action = np.array([service_idx, delta_idx], dtype=np.int64)
        
        return discrete_action
    
    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        将离散动作转换回连续动作（用于调试/可视化）
        
        Args:
            action: 离散动作 [service_idx, delta_idx]
        
        Returns:
            连续动作 [a0, a1]
        """
        service_idx, delta_idx = action
        
        # 映射回 [0, 1] 中心值
        a0 = (service_idx + 0.5) / self.n_services
        a1 = (delta_idx + 0.5) / self.n_deltas
        
        return np.array([a0, a1], dtype=np.float32)


class DiscreteActionInfoWrapper(gym.Wrapper):
    """
    在 info 中添加离散动作的详细信息（用于调试和日志）
    
    配合 ContinuousToDiscreteActionWrapper 使用，记录转换后的离散动作。
    """
    
    def __init__(self, env: gym.Env, service_ids: list = None, delta_map: dict = None):
        """
        Args:
            env: 包装后的环境
            service_ids: 服务名称列表（用于可读性）
            delta_map: 索引到副本变化的映射，例如 {0: -4, 1: -3, ..., 8: +4}
        """
        super().__init__(env)
        self.service_ids = service_ids or []
        self.delta_map = delta_map or {}
        self._last_continuous_action = None
        self._last_discrete_action = None
    
    def step(self, action):
        """记录连续动作，然后执行环境步骤"""
        self._last_continuous_action = action.copy() if isinstance(action, np.ndarray) else action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 如果使用了 ContinuousToDiscreteActionWrapper，记录转换后的动作
        if hasattr(self.env, 'action') and hasattr(self.env, 'n_services'):
            discrete_action = self.env.action(self._last_continuous_action)
            self._last_discrete_action = discrete_action
            
            service_idx, delta_idx = discrete_action
            
            # 添加到 info
            info['continuous_action'] = self._last_continuous_action
            info['discrete_action'] = discrete_action
            info['service_idx'] = int(service_idx)
            info['delta_idx'] = int(delta_idx)
            
            # 添加可读信息
            if service_idx < len(self.service_ids):
                info['service_name'] = self.service_ids[service_idx]
            if delta_idx in self.delta_map:
                info['replica_delta'] = self.delta_map[delta_idx]
        
        return obs, reward, terminated, truncated, info


class FlattenMultiDiscreteActionWrapper(gym.ActionWrapper):
    """Flatten MultiDiscrete([n_services, n_deltas]) into Discrete(n_services*n_deltas).

    This is useful for algorithms such as SB3 DQN that only support a single
    discrete action dimension.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError(
                f"FlattenMultiDiscreteActionWrapper expects MultiDiscrete action space, "
                f"but got {type(env.action_space)}"
            )
        self.discrete_nvec = env.action_space.nvec
        self.n_services = int(self.discrete_nvec[0])
        self.n_deltas = int(self.discrete_nvec[1])
        self.action_space = spaces.Discrete(self.n_services * self.n_deltas)
        self._original_action_space = env.action_space

    def action(self, action) -> np.ndarray:
        flat = int(np.asarray(action).item())
        flat = int(np.clip(flat, 0, self.action_space.n - 1))
        service_idx = flat // self.n_deltas
        delta_idx = flat % self.n_deltas
        return np.array([service_idx, delta_idx], dtype=np.int64)

    def reverse_action(self, action: np.ndarray) -> int:
        service_idx, delta_idx = action
        return int(service_idx) * self.n_deltas + int(delta_idx)


class FlattenDiscreteActionInfoWrapper(gym.Wrapper):
    """Add decoded action details to info for flattened DQN actions."""

    def __init__(self, env: gym.Env, service_ids: list = None, delta_map: dict = None):
        super().__init__(env)
        self.service_ids = service_ids or []
        self.delta_map = delta_map or {}
        self._last_flat_action = None
        self._last_discrete_action = None

    def step(self, action):
        self._last_flat_action = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = self.env.step(action)
        if hasattr(self.env, "action") and hasattr(self.env, "n_services"):
            discrete_action = self.env.action(self._last_flat_action)
            self._last_discrete_action = discrete_action
            service_idx, delta_idx = discrete_action
            info["flat_action"] = self._last_flat_action
            info["discrete_action"] = discrete_action
            info["service_idx"] = int(service_idx)
            info["delta_idx"] = int(delta_idx)
            if service_idx < len(self.service_ids):
                info["service_name"] = self.service_ids[service_idx]
            if delta_idx in self.delta_map:
                info["replica_delta"] = self.delta_map[delta_idx]
        return obs, reward, terminated, truncated, info


def wrap_for_sac(env: gym.Env, service_ids: list = None, add_info: bool = True) -> gym.Env:
    """
    便捷函数：为 SAC 训练包装环境
    
    将 MultiDiscrete 动作空间转换为 Box 连续空间，并可选添加动作信息记录。
    
    Args:
        env: 原始环境（MultiDiscrete 动作空间）
        service_ids: 服务名称列表
        add_info: 是否添加动作信息到 info（用于调试）
    
    Returns:
        包装后的环境（Box 动作空间）
    
    示例:
        >>> online_env = planner.make_env(...)  # MultiDiscrete action space
        >>> wrapped_env = wrap_for_sac(online_env, service_ids=['svc1', 'svc2', ...])
        >>> # 现在可以用 SAC 训练了
        >>> model = SAC("MlpPolicy", wrapped_env, ...)
    """
    # 1. 转换动作空间
    env = ContinuousToDiscreteActionWrapper(env)
    
    # 2. 可选：添加动作信息记录
    if add_info and service_ids:
        # 构建 delta 映射（假设是 [-4, -3, -2, -1, 0, +1, +2, +3, +4]）
        if hasattr(env, 'n_deltas'):
            n_deltas = env.n_deltas
            mid = n_deltas // 2
            delta_map = {i: i - mid for i in range(n_deltas)}
        else:
            delta_map = {}
        
        env = DiscreteActionInfoWrapper(env, service_ids=service_ids, delta_map=delta_map)
    
    return env


def wrap_for_dqn(env: gym.Env, service_ids: list = None, add_info: bool = True) -> gym.Env:
    """Wrap a MultiDiscrete KuTARA environment for DQN.

    The flattened action index is decoded as:
    ``service_idx = action // n_deltas`` and ``delta_idx = action % n_deltas``.
    """
    env = FlattenMultiDiscreteActionWrapper(env)
    if add_info and service_ids:
        if hasattr(env, "n_deltas"):
            n_deltas = env.n_deltas
            mid = n_deltas // 2
            delta_map = {i: i - mid for i in range(n_deltas)}
        else:
            delta_map = {}
        env = FlattenDiscreteActionInfoWrapper(env, service_ids=service_ids, delta_map=delta_map)
    return env
