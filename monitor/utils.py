import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def parse_cpu_to_cores(cpu_str: Optional[str]) -> float:
    """
    将CPU字符串解析为以“核（cores）”为单位的浮点数。
    支持以下单位：
        - "m" : millicore -> 核 = m / 1000
        - "u" : microcore -> 核 = u / 1_000_000
        - "n" : nanocore  -> 核 = n / 1_000_000_000
        - 无单位或纯数字，直接转为浮点数
    出错或未知单位返回 0.0 并记录 warning
    """
    if not cpu_str:
        return 0.0

    s = cpu_str.strip().lower()

    try:
        if s.endswith("m"):
            # 毫核
            return float(s[:-1]) / 1000.0
        elif s.endswith("u"):
            # 微核
            return float(s[:-1]) / 1_000_000.0
        elif s.endswith("n"):
            # 纳核
            return float(s[:-1]) / 1_000_000_000.0
        else:
            # 没有单位，直接转换为浮点数
            return float(s)
    except ValueError:
        logger.warning("无法解析CPU单位: %s", cpu_str)
        return 0.0


def parse_memory_to_mi(mem_str: Optional[str]) -> float:
    """
    将内存字符串解析为以 Mi 为单位的浮点数。
    支持："512Mi" -> 512, "1Gi" -> 1024, "1024Ki" -> 1。
    若不带单位的纯数字，按字节（bytes）处理并转换为 Mi（即 value / 1024^2），
    这能避免不同数据源返回不统一单位导致的异常（例如Kubernetes某些API返回字节数）。
    """
    if not mem_str:
        return 0.0
    s = mem_str.strip()
    # normalize common variants and use regex to extract number + unit
    import re

    # common units mapping to Mi multiplier
    unit_map = {
        'gi': 1024.0,
        'g': 1024.0,
        'mi': 1.0,
        'm': None,  
        'ki': 1.0 / 1024.0,
        'k': 1.0 / 1024.0,
        'b': 1.0 / (1024.0 * 1024.0),  # bytes -> Mi
    }

    # match patterns like '123Mi', '1Gi', '1024', '64424509440m'
    m = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?\s*$', s)
    if not m:
        logger.warning("无法解析内存: %s", mem_str)
        return 0.0

    val_str, unit = m.group(1), (m.group(2) or '')
    try:
        val = float(val_str)
    except Exception:
        logger.warning("无法解析内存数值: %s", mem_str)
        return 0.0

    unit = unit.lower()
    # If no unit given, treat as bytes (convert to Mi)
    if unit == '':
        return val / (1024.0 * 1024.0)

    if unit == 'm' and val > 1024 * 1024 *1024:
        # treat as bytes
        return val / (1024.0 * 1024.0 * 1024.0)

    # normalize unit forms like 'Gi', 'G', 'Mi', 'M', 'Ki', 'K', 'B'
    unit_norm = unit
    if unit_norm.endswith('i') is False and unit_norm in ('g', 'm', 'k'):
        unit_norm = unit_norm + ('i' if unit_norm in ('g', 'm', 'k') else '')

    # check known mappings
    if unit_norm in unit_map and unit_map[unit_norm] is not None:
        return val * unit_map[unit_norm]

    # handle bytes (B) or 'bytes'
    if unit_norm in ('b', 'bytes'):
        return val / (1024.0 * 1024.0)

    # fallback: try stripping trailing letters and convert remaining
    try:
        return float(re.sub(r'[^0-9.]', '', s))
    except Exception:
        logger.warning("无法解析内存: %s", mem_str)
        return 0.0


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_label_selector_from_match_labels(match_labels: Optional[Dict[str, str]]) -> str:
    """
    根据Deployment.spec.selector.matchLabels构建label_selector字符串，确保与调度一致。
    """
    if not match_labels:
        return ""
    parts = []
    for k, v in match_labels.items():
        parts.append(f"{k}={v}")
    return ",".join(parts)

