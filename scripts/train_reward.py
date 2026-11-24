import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig,RewardTrainer
from src.common import BASE,dtype,lora,tokenizer_for,seed_everything,device_info,save_meta
from src.monitoring import TrainingMonitorCallback
p=argparse.ArgumentParser();p.add_argument("--model",default="outputs/sft_merged");p.add_argument("--steps",type=int,default=70);p.add_argument("--out",default="outputs/reward_adapter");p.add_argument("--run-name",default="");args=p.parse_args()
seed_everything();print(device_info())
tok=tokenizer_for(args.model)
model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=1,torch_dtype=dtype(),ignore_mismatched_sizes=True)
model.config.pad_token_id=tok.pad_token_id
train=load_dataset("json",data_files="data/pref_train.jsonl",split="train")
val=load_dataset("json",data_files="data/pref_val.jsonl",split="train")
config=RewardConfig(output_dir=args.out,max_steps=args.steps,learning_rate=1e-4,per_device_train_batch_size=1,gradient_accumulation_steps=4,gradient_checkpointing=True,max_length=384,logging_steps=5,eval_strategy="steps",eval_steps=max(10,args.steps//4),save_strategy="no",report_to="none",bf16=device_info()["bf16"],fp16=not device_info()["bf16"],optim="adamw_torch",seed=42)
trainer=RewardTrainer(model=model,args=config,train_dataset=train,eval_dataset=val,processing_class=tok,peft_config=lora("SEQ_CLS"))
monitor=TrainingMonitorCallback("reward_model",args.run_name or None,metadata={"base_model":args.model,"output_dir":args.out})
trainer.add_callback(monitor)
trainer.train();metrics=trainer.evaluate();trainer.save_model(args.out);tok.save_pretrained(args.out)
monitor.finish()
save_meta(args.out,{"base":args.model,"stage":"reward_model","metrics":metrics,"device":device_info(),"monitor_dir":str(monitor.run_dir)});print(metrics)
