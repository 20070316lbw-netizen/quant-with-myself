"""
批量生成数据划分配置文件

用法：
    直接运行，会生成 DEFAULT_CONFIGS 里定义的所有配置文件。
    也可以在其他脚本里 import generate_config() 单独调用。

注意：这个脚本只负责生成/更新 data 部分（时间范围 + train/test 划分）。
     如果目标文件已存在，会保留其他 key（比如 model:），只覆盖 data 这一块。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from datetime import datetime, timedelta
from utils.config import PROJECT_ROOT


def generate_config(
    start: str,
    end: str,
    train_ratio: float = 0.7,
    output_path: Path = None,
) -> dict:
    """
    生成训练/测试划分配置，并写入 yaml 文件。

    参数:
        start       数据起始日期，格式 'YYYY-MM-DD'
        end         数据结束日期，格式 'YYYY-MM-DD'
        train_ratio 训练集占比，默认 0.7
        output_path 输出文件路径；None 则写到项目根目录的 config.yaml

    返回:
        生成的配置 dict
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days
    train_days = int(total_days * train_ratio)

    train_end = start_dt + timedelta(days=train_days)
    test_start = train_end + timedelta(days=1)

    # 只生成 data 部分，splits 嵌套在 data 下面（跟 README 和 config.yaml 保持一致）
    data_block = {
        "start": start,
        "end": end,
        "splits": {
            "train": {
                "start": start,
                "end": train_end.strftime("%Y-%m-%d"),
            },
            "test": {
                "start": test_start.strftime("%Y-%m-%d"),
                "end": end,
            },
        },
    }

    if output_path is None:
        output_path = PROJECT_ROOT / "config.yaml"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在，先读出来，只替换 data 部分。
    # 这样才不会干掉你手写在同一个 yaml 里的 model:、feature: 等其他配置。
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    else:
        existing = {}

    existing["data"] = data_block

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✅ 已写入: {output_path}")
    return existing


# ── 批量生成的配置列表 ──────────────────────────────────────────
# 每一项: (start, end, train_ratio, 输出文件名)
DEFAULT_CONFIGS = [
    ("2024-05-12", "2025-05-12", 0.7, "config.yaml"),
    ("2023-01-01", "2025-01-01", 0.8, "config_2y_80.yaml"),
]


if __name__ == "__main__":
    for start, end, ratio, filename in DEFAULT_CONFIGS:
        cfg = generate_config(
            start=start,
            end=end,
            train_ratio=ratio,
            output_path=PROJECT_ROOT / filename,
        )
        print(yaml.dump(cfg, default_flow_style=False))
