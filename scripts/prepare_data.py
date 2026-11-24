"""Prepare reproducible train/validation sets. No test examples enter training."""
import argparse,json,random
from pathlib import Path
from datasets import load_dataset
from src.common import gsm_gold,normalize_math_answer

def write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")

def role_to_str(x):
    if isinstance(x,str):return x
    return "\n".join(str(t.get("content","")) for t in x if isinstance(t,dict))

p=argparse.ArgumentParser();p.add_argument("--n-pref",type=int,default=512);p.add_argument("--n-math",type=int,default=1024);p.add_argument("--seed",type=int,default=42);args=p.parse_args()
# HF preference dataset uses chosen/rejected message lists
pref=load_dataset("trl-lib/ultrafeedback_binarized",split="train")
pref=pref.shuffle(seed=args.seed).select(range(min(args.n_pref,len(pref))))
items=[]
for r in pref:
    ch=r.get("chosen"); rej=r.get("rejected")
    if not ch or not rej:continue
    items.append({"chosen":ch,"rejected":rej})
# Fixed shuffle/split: disjoint validation samples; no public test leakage.
n_val=max(16,min(128,len(items)//10))
if len(items)<=n_val:raise ValueError("Increase --n-pref")
write("data/pref_train.jsonl",items[n_val:]);write("data/pref_val.jsonl",items[:n_val])
math=load_dataset("openai/gsm8k","main",split="train")
math=math.shuffle(seed=args.seed).select(range(min(args.n_math,len(math))))
math_items=[]
for r in math:
    gold=gsm_gold(r["answer"])
    if gold is None:continue
    math_items.append({"question":r["question"],"solution":normalize_math_answer(r["answer"]),"gold":gold})
n_val=max(16,min(128,len(math_items)//10))
if len(math_items)<=n_val:raise ValueError("Increase --n-math")
write("data/math_train.jsonl",math_items[n_val:]);write("data/math_val.jsonl",math_items[:n_val])
print(f"Preference train={len(items)-max(16,min(128,len(items)//10))}, val={max(16,min(128,len(items)//10))}; Math train={len(math_items)-n_val}, val={n_val}")
print("GSM8K official test is not loaded here. Use it only from evaluate.py after training.")
