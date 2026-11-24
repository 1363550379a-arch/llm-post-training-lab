"""从统一 metrics.csv 生成训练曲线图。"""
import csv
import math
from pathlib import Path


def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _load_rows(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _group_metric(name):
    n = name.lower()
    if "loss" in n:
        return "loss"
    if any(k in n for k in ["reward", "accuracy", "margin", "advantage", "return"]):
        return "reward"
    if any(k in n for k in ["kl", "entropy", "grad_norm", "learning_rate", "clip", "ratio"]):
        return "optimization"
    if any(k in n for k in ["memory", "ram", "cpu", "tokens_per", "samples_per", "response_length", "completion_length", "rollout_tokens", "duration", "time"]):
        return "system"
    return "other"


def plot_run(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到训练日志：{csv_path}")
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"训练日志为空：{csv_path}")

    metrics = {}
    for key in rows[0].keys():
        if key in {"step", "timestamp"}:
            continue
        values = [_number(row.get(key)) for row in rows]
        points = [(i, v) for i, v in enumerate(values) if v is not None]
        if points:
            # 横轴优先使用 global step；重复 step（例如评估日志）仍保留其先后次序。
            xs = []
            for i, _ in points:
                s = _number(rows[i].get("step"))
                xs.append(s if s is not None else i)
            metrics[key] = (xs, [v for _, v in points])

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = {"loss": [], "reward": [], "optimization": [], "system": [], "other": []}
    for key, values in metrics.items():
        groups[_group_metric(key)].append((key, values))

    generated = []
    for group, items in groups.items():
        if not items:
            continue
        cols = 2
        nrows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(14, max(4, 3.6 * nrows)), squeeze=False)
        for ax, (name, (xs, ys)) in zip(axes.flat, items):
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.4)
            ax.set_title(name)
            ax.set_xlabel("Training step / log order")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
        for ax in list(axes.flat)[len(items):]:
            ax.remove()
        fig.suptitle(f"{run_dir.parent.name} / {run_dir.name} — {group} metrics", fontsize=13)
        fig.tight_layout()
        path = plot_dir / f"{group}_metrics.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))

    # 单独生成一张总览图，方便快速查看训练是否稳定。
    preferred = [k for k in metrics if any(x in k.lower() for x in ["loss", "reward", "accuracy", "kl", "grad_norm", "learning_rate"])]
    if preferred:
        fig, axes = plt.subplots(len(preferred), 1, figsize=(11, max(3, 2.5 * len(preferred))), squeeze=False)
        for ax, key in zip(axes.flat, preferred):
            xs, ys = metrics[key]
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.3)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.25)
        axes.flat[-1].set_xlabel("Training step / log order")
        fig.suptitle(f"Training dashboard — {run_dir.parent.name}/{run_dir.name}")
        fig.tight_layout()
        path = plot_dir / "training_dashboard.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))

    return generated
