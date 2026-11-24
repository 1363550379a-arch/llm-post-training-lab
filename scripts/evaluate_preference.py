"""Held-out offline preference likelihood ranking; NOT an independent LLM judge win rate."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import PeftModel
from src.common import tokenizer_for,dtype,device_info
p=argparse.ArgumentParser();p.add_argument("--model",default="outputs/sft_merged");p.add_argument("--adapter",default="");p.add_argument("--limit",type=int,default=100);p.add_argument("--out",default="results/sft_pref_ranking.json");args=p.parse_args()
tok=tokenizer_for(args.model)
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype()).to("cuda:0").eval()
if args.adapter:model=PeftModel.from_pretrained(model,args.adapter).eval()
val=load_dataset("json",data_files="data/pref_val.jsonl",split="train")
if args.limit>0:val=val.select(range(min(args.limit,len(val))))
def avg_response_ll(msgs):
    pids=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True)
    ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)
    if len(ids)>512:
        ids=ids[-512:]
        prompt_len=max(1,len(pids)-(len(tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False))-512))
    else:prompt_len=len(pids)
    seq=torch.tensor([ids],device="cuda:0")
    with torch.inference_mode():
        logits=model(input_ids=seq).logits[:,:-1]
        logp=F.log_softmax(logits.float(),dim=-1).gather(-1,seq[:,1:].unsqueeze(-1)).squeeze(-1)
    return float(logp[:,max(0,prompt_len-1):].mean())
records=[]
for i,row in enumerate(val):
    ch=avg_response_ll(row["chosen"]);rej=avg_response_ll(row["rejected"])
    records.append({"id":i,"chosen_mean_loglik":ch,"rejected_mean_loglik":rej,"chosen_ranked_higher":ch>rej})
summary={"model":args.model,"adapter":args.adapter,"metric":"chosen_vs_rejected_mean_conditional_loglik_accuracy","n":len(records),"accuracy":sum(x["chosen_ranked_higher"] for x in records)/max(1,len(records)),"note":"Offline heldout preference-label ranking; NOT independent human/LLM judge win-rate","hardware":device_info()}
path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({"summary":summary,"samples":records},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
