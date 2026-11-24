import argparse,json
from datasets import load_dataset
from trl import SFTConfig,SFTTrainer
from src.common import BASE,causal_model,lora,tokenizer_for,seed_everything,device_info,save_meta
from src.monitoring import TrainingMonitorCallback
p=argparse.ArgumentParser();p.add_argument("--model",default=BASE);p.add_argument("--steps",type=int,default=80);p.add_argument("--out",default="outputs/sft_adapter");p.add_argument("--length",type=int,default=384);p.add_argument("--run-name",default="");args=p.parse_args()
seed_everything(); print(device_info())
tok=tokenizer_for(args.model)
def convert(row):
    msgs=row["chosen"]
    if not isinstance(msgs,list) or not msgs or msgs[-1].get("role")!="assistant":raise ValueError("Unexpected chosen format")
    prompt=tok.apply_chat_template(msgs[:-1],tokenize=False,add_generation_prompt=True)
    response=str(msgs[-1]["content"])+tok.eos_token
    return {"prompt":prompt,"completion":response}
train=load_dataset("json",data_files="data/pref_train.jsonl",split="train").map(convert,remove_columns=["chosen","rejected"])
val=load_dataset("json",data_files="data/pref_val.jsonl",split="train").map(convert,remove_columns=["chosen","rejected"])
def has_supervised_tokens(row):
    """TRL truncates on the right. Drop rows whose prompt consumes the window."""
    prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    return len(prompt_ids) < args.length and bool(row["completion"])

train = train.filter(has_supervised_tokens)
val = val.filter(has_supervised_tokens)
if not len(train) or not len(val):
    raise ValueError("No usable examples after applying the SFT length limit")
config=SFTConfig(output_dir=args.out,max_steps=args.steps,learning_rate=1e-4,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,gradient_checkpointing=True,max_length=args.length,completion_only_loss=True,logging_steps=5,eval_strategy="steps",eval_steps=max(10,args.steps//4),save_strategy="no",report_to="none",bf16=device_info()["bf16"],fp16=not device_info()["bf16"],optim="adamw_torch",seed=42)
trainer=SFTTrainer(model=causal_model(args.model),args=config,train_dataset=train,eval_dataset=val,processing_class=tok,peft_config=lora())
monitor=TrainingMonitorCallback("sft",args.run_name or None,metadata={"base_model":args.model,"output_dir":args.out})
trainer.add_callback(monitor)
trainer.train();metrics=trainer.evaluate();trainer.save_model(args.out);tok.save_pretrained(args.out)
monitor.finish()
save_meta(args.out,{"base":args.model,"stage":"SFT","steps":args.steps,"eval":metrics,"device":device_info(),"monitor_dir":str(monitor.run_dir)});print(metrics)
