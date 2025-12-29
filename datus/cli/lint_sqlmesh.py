from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _collect_sql_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".sql":
        return [path]
    files: List[Path] = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if name.lower().endswith(".sql"):
                files.append(Path(root) / name)
    return files


def run_lint_sqlmesh(agent_config, db_manager, args) -> Dict[str, any]:
    """
    轻量 lint：仅扫描文件名命名是否 snake_case / 层级前缀。
    """
    path = Path(getattr(args, "path", ""))
    if not path.exists():
        return {"status": "error", "message": f"path not found: {path}"}
    files = _collect_sql_files(path)
    if not files:
        return {"status": "success", "message": "no sql files"}

    warnings = []
    for file in files:
        name = file.stem
        if not name.replace("__", "_").islower():
            warnings.append(f"{file}: 文件名建议使用小写 snake_case")
        if not (name.startswith("dim_") or name.startswith("dwd_") or name.startswith("dws_") or "__" in name):
            warnings.append(f"{file}: 未检测到 dim_/dwd_/dws_ 前缀，建议按层命名")

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("lint 通过，未发现命名问题")
    return {"status": "success", "warnings": warnings}


if __name__ == "__main__":  # pragma: no cover
    print("请通过 datus-agent lint-sqlmesh 调用本模块")
