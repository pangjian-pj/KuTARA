import logging
from typing import Optional, Callable
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from .env import KuTARAEnv, PlanningConfig
from monitor.monitor import Monitor
from analysis.analyzer import Analyzer

logger = logging.getLogger(__name__)


class Planner:
    """
    封装 SB3 的 RL 训练/推理流程，支持 SAC 离线预训练 + 在线微调
    """
    def __init__(self, algo: str = "sac", seed: int = 42, 
                 learning_rate: float = 3e-4,
                 buffer_size: int = 100000,
                 batch_size: int = 256):
        self.algo = algo.lower()
        self.seed = seed
        self.model = None
        # SAC specific hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def make_env(self, monitor: Monitor, analyzer: Analyzer, config: PlanningConfig, executor=None) -> gym.Env:
        env = KuTARAEnv(monitor=monitor, analyzer=analyzer, config=config, executor=executor)
        return env

    def build_model(self, env: gym.Env, tensorboard_log: Optional[str] = None, 
                   policy_kwargs: Optional[dict] = None,
                   learning_rate: Optional[float] = None,
                   ent_coef: str = 'auto'):
        """构建 RL 模型
        
        Args:
            env: Gym 环境
            tensorboard_log: TensorBoard 日志目录
            policy_kwargs: 策略网络参数
            learning_rate: 学习率（如果为 None 则使用初始化时的值）
            ent_coef: 熵系数，'auto' 表示自动调整
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        if self.algo == "sac":
            self.model = SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr,
                buffer_size=self.buffer_size,
                learning_starts=1000,
                batch_size=self.batch_size,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef=ent_coef,
                target_update_interval=1,
                verbose=1,
                seed=self.seed,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
            )
        elif self.algo in ["recurrent_ppo", "rppo", "r-ppo"]:
            self.model = RecurrentPPO(
                "MlpLstmPolicy", env, 
                learning_rate=lr,
                verbose=1, 
                seed=self.seed, 
                tensorboard_log=tensorboard_log, 
                policy_kwargs=policy_kwargs
            )
        elif self.algo == "ppo":
            self.model = PPO(
                "MlpPolicy", env,
                learning_rate=lr,
                verbose=1, 
                seed=self.seed, 
                tensorboard_log=tensorboard_log, 
                policy_kwargs=policy_kwargs
            )
        else:
            raise ValueError(f"不支持的算法: {self.algo}")
        
        logger.info(f"Built {self.algo.upper()} model with lr={lr}, buffer_size={self.buffer_size if self.algo == 'sac' else 'N/A'}")
        return self.model

    def train(self, total_timesteps: int = 50_000, callback=None, reset_num_timesteps: bool = True):
        """常规训练接口"""
        assert self.model is not None, "请先调用 build_model"
        self.model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
        return self.model

    def train_offline(self, offline_env: gym.Env, total_timesteps: int = 50000,
                     callback=None, load_data_fn: Optional[Callable] = None):
        """离线预训练阶段
        
        Args:
            offline_env: 离线数据环境（可以是 OfflineReplayEnv）
            total_timesteps: 离线训练步数
            callback: 训练回调
            load_data_fn: 可选的数据加载函数，用于预填充 replay buffer
        """
        assert self.model is not None, "请先调用 build_model"
        logger.info(f"=== Starting Offline Pre-training ({total_timesteps} steps) ===")
        
        # 如果提供了数据加载函数且模型支持 replay buffer，预填充数据
        if load_data_fn is not None and hasattr(self.model, 'replay_buffer'):
            logger.info("Pre-filling replay buffer with offline data...")
            load_data_fn(self.model.replay_buffer)
            logger.info(f"Replay buffer size: {self.model.replay_buffer.size()}")
        
        # 离线训练时使用更多 epoch 充分利用数据
        if self.algo == "sac":
            # SAC 会自动从 replay buffer 采样，无需特殊处理
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        else:
            # PPO 需要与环境交互
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        logger.info("Offline pre-training completed")
        return self.model

    def train_online(self, online_env: gym.Env, total_timesteps: int = 10000,
                    callback=None, fine_tune_lr: Optional[float] = None):
        """在线微调阶段
        
        Args:
            online_env: 在线真实环境
            total_timesteps: 在线训练步数
            callback: 训练回调
            fine_tune_lr: 微调学习率（建议比离线低 3-10 倍）
        """
        assert self.model is not None, "请先完成离线预训练或加载模型"
        logger.info(f"=== Starting Online Fine-tuning ({total_timesteps} steps) ===")
        
        # 切换到在线环境
        self.model.set_env(online_env)
        logger.info("Switched to online environment")
        
        # 降低学习率做微调（更保守的更新）
        if fine_tune_lr is not None:
            original_lr = self.model.learning_rate
            self.model.learning_rate = fine_tune_lr
            logger.info(f"Reduced learning rate: {original_lr} -> {fine_tune_lr}")
        
        # 对于 SAC，降低探索程度
        if self.algo == "sac" and hasattr(self.model, 'ent_coef'):
            if isinstance(self.model.ent_coef, str) and self.model.ent_coef == 'auto':
                logger.info("SAC entropy coefficient is auto-tuned")
            else:
                # 降低熵系数以减少探索
                original_ent = self.model.ent_coef
                self.model.ent_coef = max(0.001, float(original_ent) * 0.1)
                logger.info(f"Reduced entropy coefficient: {original_ent} -> {self.model.ent_coef}")
        
        # 在线微调
        self.model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
        
        logger.info("Online fine-tuning completed")
        return self.model

    def save(self, path: str):
        assert self.model is not None, "无可保存的模型"
        self.model.save(path)
        logger.info(f"Model saved to {path}.zip")

    def load(self, path: str, env: Optional[gym.Env] = None):
        """加载训练好的模型"""
        if self.algo == "sac":
            self.model = SAC.load(path, env=env)
        elif self.algo in ["recurrent_ppo", "rppo", "r-ppo"]:
            self.model = RecurrentPPO.load(path, env=env)
        elif self.algo == "ppo":
            self.model = PPO.load(path, env=env)
        else:
            raise ValueError(f"不支持的算法: {self.algo}")
        
        logger.info(f"Loaded {self.algo.upper()} model from {path}")
        return self.model

