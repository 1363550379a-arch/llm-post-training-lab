"""Math SFT branch so GRPO starts with non-sparse GSM8K rewards."""
import argparse
from datasets import load_dataset
from trl import SFTConfig,SFTTrainer
from src.common import BASE,causal_model,lora,tokenizer_for,seed_everything,device_info,save_meta
from src.monitoring import TrainingMonitorCallback
p=argparse.ArgumentParser();p.add_argument("--model",default=BASE);p.add_argument("--steps",type=int,default=120);p.add_argument("--out",default="outputs/math_sft_adapter");p.add_argument("--run-name",default="");args=p.parse_args()
seed_everything();info=device_info();print(info)
tok=tokenizer_for(args.model)
def convert(r):
    prompt=tok.apply_chat_template([{"role":"user","content":r["question"]+"\nSolve and end with Final answer: <number>."}],tokenize=False,add_generation_prompt=True)
    completion=str(r["solution"])+"\nFinal answer: "+str(r["gold"])+tok.eos_token
    return {"prompt":prompt,"completion":completion}
train=load_dataset("json",data_files="data/math_train.jsonl",split="train").map(convert,remove_columns=["question","solution","gold"])
val=load_dataset("json",data_files="data/math_val.jsonl",split="train").map(convert,remove_columns=["question","solution","gold"])
config=SFTConfig(output_dir=args.out,max_steps=args.steps,learning_rate=1e-4,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,gradient_checkpointing=True,max_length=384,completion_only_loss=True,logging_steps=5,eval_strategy="steps",eval_steps=max(10,args.steps//4),save_strategy="no",report_to="none",bf16=info["bf16"],fp16=not info["bf16"],optim="adamw_torch",seed=42)
trainer=SFTTrainer(model=causal_model(args.model),args=config,train_dataset=train,eval_dataset=val,processing_class=tok,peft_config=lora())
monitor=TrainingMonitorCallback("math_sft",args.run_name or None,metadata={"base_model":args.model,"output_dir":args.out})
trainer.add_callback(monitor)
trainer.train();metrics=trainer.evaluate();trainer.save_model(args.out);tok.save_pretrained(args.out)
monitor.finish()
save_meta(args.out,{"base":args.model,"stage":"math_SFT","steps":args.steps,"metrics":metrics,"device":info,"monitor_dir":str(monitor.run_dir)})
