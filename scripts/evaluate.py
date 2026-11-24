"""Generate a proper heldout report: GSM8K exact-match and preference proxy RM pairwise accuracy.
All numbers are computed locally and saved; none are prefilled.
"""
import argparse,collections,json,os,time
from pathlib import Path
import torch
from datasets import load_dataset
from peft import PeftModel,PeftConfig,AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM
from src.common import tokenizer_for,dtype,device_info,pred_number,gsm_gold,num_equal
p=argparse.ArgumentParser();p.add_argument("--model",default="outputs/sft_merged");p.add_argument("--adapter",default="");p.add_argument("--reward-adapter",default="");p.add_argument("--limit",type=int,default=100);p.add_argument("--max-new",type=int,default=128);p.add_argument("--out",default="results/sft_eval.json");args=p.parse_args()
if not torch.cuda.is_available():raise SystemExit("No GPU detected")
tok=tokenizer_for(args.model)
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype(),low_cpu_mem_usage=True).to("cuda:0").eval()
if args.adapter:model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False).eval()
# Evaluate on the official UNSEEN GSM8K test split.
test=load_dataset("openai/gsm8k","main",split="test")
if args.limit>0:test=test.select(range(min(args.limit,len(test))))
records=[];start=time.perf_counter()
for idx,r in enumerate(test):
    prompt=[{"role":"user","content":r["question"]+"\nSolve and end with Final answer: <number>."}]
    ids=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_tensors="pt").to("cuda:0")
    with torch.inference_mode():out=model.generate(ids,max_new_tokens=args.max_new,do_sample=False,pad_token_id=tok.eos_token_id)
    result=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
    pred=pred_number(result);gold=gsm_gold(r["answer"])
    records.append({"id":idx,"correct":num_equal(pred,gold),"prediction":pred,"gold":gold,"generated_tokens":int(out.shape[1]-ids.shape[1]),"completion":result})
    if (idx+1)%20==0:print(f"evaluated {idx+1}/{len(test)}",flush=True)
summary={"model":args.model,"adapter":args.adapter,"dataset":"openai/gsm8k main test","evaluated":len(records),"pass_at_1_greedy_exact_match":sum(x["correct"] for x in records)/max(1,len(records)),"mean_generated_tokens":sum(x["generated_tokens"] for x in records)/max(1,len(records)),"wall_time_seconds":round(time.perf_counter()-start,1),"max_new_tokens":args.max_new,"seed":42,"decoding":"greedy","hardware":device_info()}
# Reward model is only assessed on independent preference validation examples;
# do not mistake agreement with these labels for evaluation of generation quality.
if args.reward_adapter:
    rm=AutoPeftModelForSequenceClassification.from_pretrained(args.reward_adapter,torch_dtype=dtype(),is_trainable=False).to("cuda:0").eval()
    rtok=tokenizer_for(args.reward_adapter);rm.config.pad_token_id=rtok.pad_token_id
    pref=load_dataset("json",data_files="data/pref_val.jsonl",split="train")
    pairs=min(args.limit if args.limit>0 else len(pref),len(pref));correct=0
    for i in range(pairs):
        row=pref[i]
        def score(msg):
            ids=rtok.apply_chat_template(msg,tokenize=True,add_generation_prompt=False,return_tensors="pt",truncation=True,max_length=384).to("cuda:0")
            with torch.inference_mode():return rm(input_ids=ids,attention_mask=torch.ones_like(ids)).logits.float().reshape(-1)[0].item()
        correct+=int(score(row["chosen"])>score(row["rejected"]))
    summary["reward_pair_accuracy_val"] = correct / max(1,pairs)
    summary["reward_pairs"] = pairs
path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({"summary":summary,"samples":records},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2,ensure_ascii=False))
