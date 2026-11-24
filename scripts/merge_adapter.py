import argparse,torch
from peft import PeftConfig,PeftModel
from transformers import AutoModelForCausalLM
from src.common import tokenizer_for,save_meta
p=argparse.ArgumentParser();p.add_argument("--adapter",default="outputs/sft_adapter");p.add_argument("--out",default="outputs/sft_merged");args=p.parse_args()
cfg=PeftConfig.from_pretrained(args.adapter)
base=AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map="cpu")
model=PeftModel.from_pretrained(base,args.adapter).merge_and_unload()
model.save_pretrained(args.out,safe_serialization=True,max_shard_size="1GB")
tokenizer_for(cfg.base_model_name_or_path).save_pretrained(args.out)
save_meta(args.out,{"base":cfg.base_model_name_or_path,"merged_from":args.adapter})
print("Merged SFT checkpoint:",args.out)
