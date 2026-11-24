"""Aggregate actual benchmark result files into a Markdown CV evidence table."""
import argparse,glob,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--results",default="results/*_eval.json");p.add_argument("--out",default="results/REPORT.md");args=p.parse_args()
files=sorted(glob.glob(args.results));rows=[]
for filename in files:
    d=json.loads(Path(filename).read_text(encoding="utf-8"))["summary"]
    rows.append((Path(filename).stem,d["evaluated"],d["pass_at_1_greedy_exact_match"],d["mean_generated_tokens"],d.get("reward_pair_accuracy_val")))
lines=["# Post-training comparison (measured, not fabricated)","","All numbers from independent local inference, no training/test split mixing.","","| variant | GSM8K N | greedy exact match | mean output tokens | reward-pair accuracy (if run) |","|---|---:|---:|---:|---:|"]
for r in rows:lines.append(f"| {r[0]} | {r[1]} | {r[2]*100:.2f}% | {r[3]:.1f} | {r[4]*100:.2f}%"+" |" if r[4] is not None else f"| {r[0]} | {r[1]} | {r[2]*100:.2f}% | {r[3]:.1f} | — |")
lines += ["","## Interpretation","","- GSM8K is used for mathematical reasoning; UltraFeedback preference validation tests only reward ranking.","- Inaccurate or truncated outputs count as wrong; inspect samples for formatting failures.","- Small benchmarks are exploratory; report N, decoding and seed. For a robust claim rerun with full 1319 examples and multiple seeds.","- PPO reward logits and length improvements do not imply human preference gains."]
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text("\n".join(lines)+"\n",encoding="utf-8")
print("Saved",args.out)
