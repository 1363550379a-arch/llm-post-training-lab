import argparse
from datasets import load_dataset
from trl import GRPOConfig,GRPOTrainer
from src.common import causal_model,tokenizer_for,lora,pred_number,num_equal,seed_everything,device_info,save_meta
from src.monitoring import TrainingMonitorCallback
from pathlib import Path

p = argparse.ArgumentParser(description="Historical GRPO sampling-size comparison")
p.add_argument("--model", default="outputs/math_sft_merged")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--num-generations", type=int, choices=(2, 4), default=2)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-new", "--max-completion-length", type=int, default=96)
p.add_argument("--out", default="outputs/grpo_adapter")
p.add_argument("--run-name", default="")
args = p.parse_args()
if args.steps < 1 or args.max_new < 1:
    p.error("Steps and completion length must be positive")
if Path(args.out).exists():
    p.error("Choose a new --out directory for each run")
seed_everything(args.seed)
print(device_info())
def conv(r):
    return {"prompt":[{"role":"user","content":r["question"]+"\nSolve and end with Final answer: <number>."}],"gold":r["gold"]}
train=load_dataset("json",data_files="data/math_train.jsonl",split="train").map(conv,remove_columns=["question","solution"])
def answer_reward(completions,gold,**kwargs):
    result=[]
    for c,g in zip(completions,gold):
        text=c[0].get("content","") if isinstance(c,list) else str(c)
        result.append(1.0 if num_equal(pred_number(text),g) else 0.0)
    return result
def format_reward(completions,**kwargs):
    result=[]
    for c in completions:
        t=c[0].get("content","") if isinstance(c,list) else str(c)
        result.append(0.1 if pred_number(t) is not None else 0.0)
    return result
config=GRPOConfig(output_dir=args.out,max_steps=args.steps,learning_rate=2e-5,per_device_train_batch_size=1,gradient_accumulation_steps=4,generation_batch_size=4,num_generations=args.num_generations,max_completion_length=args.max_new,max_prompt_length=192,gradient_checkpointing=True,logging_steps=1,save_strategy="no",report_to="none",use_vllm=False,beta=0.0,bf16=device_info()["bf16"],fp16=not device_info()["bf16"],optim="adamw_torch",seed=args.seed)
trainer=GRPOTrainer(model=causal_model(args.model),args=config,reward_funcs=[answer_reward,format_reward],train_dataset=train,processing_class=tokenizer_for(args.model),peft_config=lora())
monitor=TrainingMonitorCallback("grpo",args.run_name or None,metadata={"base_model":args.model,"output_dir":args.out})
trainer.add_callback(monitor)
trainer.train();trainer.save_model(args.out)
monitor.finish()
save_meta(args.out,{"base":args.model,"stage":"GRPO","config":vars(args),"reward":"1.0 exact match + 0.1 format","device":device_info(),"monitor_dir":str(monitor.run_dir)})
