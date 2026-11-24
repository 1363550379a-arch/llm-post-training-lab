import argparse
from datasets import load_dataset
from trl import DPOTrainer,DPOConfig
from src.common import causal_model,tokenizer_for,lora,seed_everything,device_info,save_meta
from src.monitoring import TrainingMonitorCallback
p=argparse.ArgumentParser();p.add_argument("--model",default="outputs/sft_merged");p.add_argument("--steps",type=int,default=60);p.add_argument("--out",default="outputs/dpo_adapter");p.add_argument("--run-name",default="");args=p.parse_args()
seed_everything();print(device_info())
train=load_dataset("json",data_files="data/pref_train.jsonl",split="train")
val=load_dataset("json",data_files="data/pref_val.jsonl",split="train")
# Chosen/rejected implicit conversational data; TRL derives common prompt.
kwargs=dict(output_dir=args.out,max_steps=args.steps,learning_rate=5e-5,per_device_train_batch_size=1,gradient_accumulation_steps=4,gradient_checkpointing=True,logging_steps=5,eval_strategy="steps",eval_steps=max(10,args.steps//4),save_strategy="no",report_to="none",max_length=384,max_prompt_length=192,beta=0.1,bf16=device_info()["bf16"],fp16=not device_info()["bf16"],optim="adamw_torch",seed=42)
config=DPOConfig(**kwargs)
trainer=DPOTrainer(model=causal_model(args.model),args=config,train_dataset=train,eval_dataset=val,processing_class=tokenizer_for(args.model),peft_config=lora())
monitor=TrainingMonitorCallback("dpo",args.run_name or None,metadata={"base_model":args.model,"output_dir":args.out})
trainer.add_callback(monitor)
trainer.train();metrics=trainer.evaluate();trainer.save_model(args.out)
monitor.finish()
print(metrics); save_meta(args.out,{"reference":"SFT merged checkpoint; adapter-disabled reference in TRL","stage":"DPO","metrics":metrics,"device":device_info(),"monitor_dir":str(monitor.run_dir)})
