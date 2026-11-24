"""Render the public figures from the numeric evidence in results/evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotting import _load_rows, _number

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "figures"
GROUPS = {
    "sft": ["loss eval_loss", "grad_norm", "learning_rate"],
    "math_sft": ["loss eval_loss", "grad_norm", "learning_rate"],
    "reward_model": ["loss eval_loss", "accuracy eval_accuracy", "margin eval_margin", "chosen_derived rejected_derived"],
    "dpo": ["loss eval_loss", "rewards/accuracies eval_rewards/accuracies",
            "rewards/chosen rewards/rejected rewards/margins"],
    "ppo": ["proxy_reward mean_reward", "sampled_kl approx_kl", "policy_loss", "value_loss", "response_length", "clip_fraction"],
    "grpo": ["reward reward_std", "rewards/answer_reward/mean rewards/format_reward/mean",
             "frac_reward_zero_std", "completions/mean_length"],
}


def finish(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    costs = {}
    for stage, groups in GROUPS.items():
        rows = _load_rows(ROOT / "evidence" / (stage + ".csv"))
        if stage == "reward_model":
            for row in rows:
                if row.get("mean_reward") and row.get("margin"):
                    mean, half = float(row["mean_reward"]), float(row["margin"]) / 2
                    row.update(chosen_derived=mean + half, rejected_derived=mean - half)
        fig, axes = plt.subplots(len(groups), figsize=(8, 1.9 * len(groups)), layout="constrained")
        fig.suptitle(stage.replace("_", " ").upper())
        for ax, group in zip(axes, groups):
            for metric in group.split():
                points = [(int(float(row["step"])), _number(row.get(metric))) for row in rows]
                points = [(s, v) for s, v in points if v is not None]
                if points:
                    ax.plot(*zip(*points), marker=".", label=metric)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Logged step")
        finish(fig, stage)
        costs[stage] = [max(float(row["elapsed_seconds"]) for row in rows),
                        max(float(row["system/gpu_max_memory_allocated_gb"]) for row in rows),
                        next((float(row["train_samples_per_second"]) for row in rows if row.get("train_samples_per_second")), None)]
    fig, axes = plt.subplots(3, figsize=(8, 7), layout="constrained")
    for i, (ax, title) in enumerate(zip(axes, ["Time to last log (s)", "Peak allocated (GiB)", "Trainer samples/s"])):
        vals = {k:v[i] for k,v in costs.items() if v[i] is not None}
        ax.bar_label(ax.bar(vals.keys(), vals.values()), fmt="%.2f", fontsize=8)
        ax.set_title(title)
        ax.margins(y=.17)
    finish(fig, "cost")
    data = json.loads((ROOT / "evidence/evaluations.json").read_text())
    for name, suffix in [("Preference ranking", "_pref_ranking"), ("GSM8K exact match", "_eval")]:
        fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
        models = ["base","sft","dpo","ppo"] if suffix.startswith("_pref") else ["base","math_sft","grpo"]
        vals = [data[m + suffix] for m in models]
        bars = ax.bar(models, [100*v["correct"]/v["n"] for v in vals], color=["#596780","#397c97","#188978","#a86532"][:len(models)])
        ax.bar_label(bars, labels=[f'{v["correct"]/v["n"]:.1%} ({v["correct"]}/{v["n"]})' for v in vals])
        ax.set(ylabel="Accuracy (%)", title=name, ylim=(0, 65))
        finish(fig, name.split()[0].lower())


if __name__ == "__main__":
    main()
