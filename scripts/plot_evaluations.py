"""绘制独立评测 JSON 的模型对比图。"""
import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[跳过] {path}: {exc}")
        return None

p = argparse.ArgumentParser()
p.add_argument("--results-dir", default="results")
p.add_argument("--out-dir", default="results/plots")
args = p.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

root = Path(args.results_dir)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

# GSM8K exact-match / 输出长度
labels, exact, lengths = [], [], []
for path in sorted(root.glob("*_eval.json")):
    data = read_json(path)
    if not data or "summary" not in data:
        continue
    summary = data["summary"]
    if "pass_at_1_greedy_exact_match" not in summary:
        continue
    labels.append(path.stem.replace("_eval", ""))
    exact.append(float(summary["pass_at_1_greedy_exact_match"]) * 100)
    lengths.append(float(summary.get("mean_generated_tokens", 0)))

if labels:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(labels, exact)
    ax.set_ylabel("GSM8K Exact Match (%)")
    ax.set_title("GSM8K 测试集准确率对比")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "gsm8k_exact_match_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(labels, lengths)
    ax.set_ylabel("平均生成 Token 数")
    ax.set_title("GSM8K 平均回答长度对比")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "gsm8k_output_length_comparison.png", dpi=160)
    plt.close(fig)

# 偏好验证集排序准确率
pref_labels, pref_acc = [], []
for path in sorted(root.glob("*_pref_ranking.json")):
    data = read_json(path)
    if not data or "summary" not in data:
        continue
    summary = data["summary"]
    if "accuracy" not in summary:
        continue
    pref_labels.append(path.stem.replace("_pref_ranking", ""))
    pref_acc.append(float(summary["accuracy"]) * 100)

if pref_labels:
    fig, ax = plt.subplots(figsize=(max(8, len(pref_labels) * 1.2), 5))
    ax.bar(pref_labels, pref_acc)
    ax.set_ylabel("Chosen 条件对数似然排序准确率 (%)")
    ax.set_title("偏好验证集排序准确率对比（非人工胜率）")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "preference_ranking_comparison.png", dpi=160)
    plt.close(fig)

if not labels and not pref_labels:
    raise SystemExit("没有找到 *_eval.json 或 *_pref_ranking.json 评测文件。")
print("评测对比图已保存至：", out)
