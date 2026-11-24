"""为单次训练或全部训练运行生成曲线图。"""
import argparse
from pathlib import Path
from src.plotting import plot_run

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default="", help="指定某次运行目录，例如 results/runs/sft/20261120_180000")
p.add_argument("--all", action="store_true", help="为 results/runs 下所有有 metrics.csv 的运行绘图")
p.add_argument("--root", default="results/runs")
args = p.parse_args()

if args.run_dir:
    targets = [Path(args.run_dir)]
else:
    targets = sorted(x.parent for x in Path(args.root).glob("**/metrics.csv"))
if not targets:
    raise SystemExit("没有找到 metrics.csv。请先运行训练脚本。")
for target in targets:
    try:
        files = plot_run(target)
        print(f"\n[{target}]")
        for file in files:
            print("  已生成：", file)
    except Exception as exc:
        print(f"[跳过] {target}: {exc}")
