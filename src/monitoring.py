"""统一训练监控：CSV/JSONL 原始日志 + TensorBoard + 运行元数据。"""
from __future__ import annotations

import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback


def _scalar(value: Any) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return value or "run"


class RunLogger:
    """适用于 Trainer 回调和自定义 PPO 循环的统一指标记录器。"""

    def __init__(self, stage: str, run_name: Optional[str] = None,
                 root: str = "results/runs", metadata: Optional[dict] = None):
        self.stage = _safe_name(stage)
        self.run_name = _safe_name(run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = Path(root) / self.stage / self.run_name
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.run_name}_{suffix}"
            self.run_dir = Path(root) / self.stage / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.rows = []
        self.started = time.perf_counter()
        self.last_log_time = self.started
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        except Exception as exc:  # CSV/JSONL 仍可用，TensorBoard 安装问题不会中断训练
            print(f"[monitor] TensorBoard 不可用，将继续记录 CSV/JSONL：{exc}", flush=True)

        self.metadata = {
            "stage": self.stage,
            "run_name": self.run_name,
            "started_at": datetime.now().astimezone().isoformat(),
            "metrics_csv": str(self.csv_path),
            "tensorboard_dir": str(self.run_dir / "tensorboard"),
        }
        if metadata:
            self.metadata.update(metadata)
        self._write_metadata()
        print(f"[monitor] 指标保存目录：{self.run_dir}", flush=True)

    def _system_metrics(self) -> Dict[str, float]:
        result = {}
        if torch.cuda.is_available():
            try:
                result["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1024**3
                result["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1024**3
                result["system/gpu_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(0) / 1024**3
                result["system/gpu_max_memory_reserved_gb"] = torch.cuda.max_memory_reserved(0) / 1024**3
            except Exception:
                pass
        try:
            import psutil
            process = psutil.Process()
            result["system/ram_used_gb"] = process.memory_info().rss / 1024**3
            result["system/cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        return result

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None,
            epoch: Optional[float] = None) -> None:
        now = time.perf_counter()
        row: Dict[str, Any] = {
            "step": int(step or 0),
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": round(now - self.started, 3),
            "interval_seconds": round(now - self.last_log_time, 3),
        }
        self.last_log_time = now
        if epoch is not None:
            v = _scalar(epoch)
            if v is not None:
                row["epoch"] = v
        for key, value in (metrics or {}).items():
            v = _scalar(value)
            if v is not None:
                row[str(key)] = v
        row.update(self._system_metrics())
        self.rows.append(row)

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rewrite_csv()

        if self.writer is not None:
            global_step = int(row.get("step", 0))
            for key, value in row.items():
                if key in {"step", "timestamp"}:
                    continue
                scalar = _scalar(value)
                if scalar is not None:
                    self.writer.add_scalar(key, scalar, global_step)
            self.writer.flush()

    def _rewrite_csv(self) -> None:
        if not self.rows:
            return
        # 列会随训练阶段动态变化（如 GRPO 的 reward/*），因此每次合并字段后重写。
        fields = []
        seen = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_metadata(self) -> None:
        path = self.run_dir / "run_metadata.json"
        path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def finish(self, final_metrics: Optional[dict] = None) -> Path:
        if final_metrics:
            self.log(final_metrics, step=self.rows[-1].get("step", 0) if self.rows else 0)
        self.metadata["finished_at"] = datetime.now().astimezone().isoformat()
        self.metadata["duration_seconds"] = round(time.perf_counter() - self.started, 3)
        self.metadata["log_rows"] = len(self.rows)
        self.metadata["last_metrics"] = self.rows[-1] if self.rows else {}
        self._write_metadata()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        try:
            from .plotting import plot_run
            plot_run(self.run_dir)
        except Exception as exc:
            print(f"[monitor] 自动绘图失败（原始日志已保存）：{exc}", flush=True)
        print(f"[monitor] 训练指标已保存：{self.csv_path}", flush=True)
        return self.run_dir


class TrainingMonitorCallback(TrainerCallback):
    """Transformers/TRL Trainer 通用回调，自动捕获 on_log 中的训练与验证指标。"""

    def __init__(self, stage: str, run_name: Optional[str] = None,
                 root: str = "results/runs", metadata: Optional[dict] = None):
        self.logger = RunLogger(stage, run_name, root, metadata)

    @property
    def run_dir(self) -> Path:
        return self.logger.run_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logger.log(logs or {}, step=state.global_step, epoch=state.epoch)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.metadata["trainer_global_step"] = int(state.global_step)
        self.logger._write_metadata()
        return control

    def finish(self, final_metrics: Optional[dict] = None) -> Path:
        return self.logger.finish(final_metrics)
