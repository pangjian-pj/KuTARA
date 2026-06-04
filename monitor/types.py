from pydantic import BaseModel, Field
from typing import Optional


class MicroserviceState(BaseModel):
    """
    统一的微服务状态对象。单位说明：
    - request_cpu/limit_cpu/cpu_usage: 以“核”为单位（cores），如 0.5 表示 500m
    - request_memory/limit_memory/memory_usage: 以 Mi 为单位
    - response_time: 以毫秒为单位
    """
    id: str = Field(..., description="微服务ID（通常与Deployment名称一致）")
    request_cpu: float = Field(0.0, description="所有容器CPU request总和（cores）")
    request_memory: float = Field(0.0, description="所有容器内存request总和（Mi）")
    limit_cpu: float = Field(0.0, description="所有容器CPU limit总和（cores）")
    limit_memory: float = Field(0.0, description="所有容器内存limit总和（Mi）")
    cpu_usage: float = Field(0.0, description="实时CPU使用量（cores）")
    memory_usage: float = Field(0.0, description="实时内存使用量（Mi）")
    cpu_utilization: float = Field(0.0, description="CPU利用率（cpu_usage/request_cpu），若request_cpu=0则为0")
    memory_utilization: float = Field(0.0, description="内存利用率（memory_usage/request_memory），若request_memory=0则为0")
    pod_num: int = Field(0, description="当前Pod数量")
    requested_replicas: int = Field(0, description="Deployment spec 中的期望副本数（spec.replicas）")
    pending_pod_num: int = Field(0, description="当前处于 Pending 状态的 Pod 数量")
    avg_startup_delay_ms: float = Field(0.0, description="平均 pod 启动延迟（从创建到Running 的平均毫秒数）")
    response_time: float = Field(0.0, description="P99/P999响应时间（ms）")

    class Config:
        extra = "ignore"

