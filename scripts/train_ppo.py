"""Single-GPU minimal RLHF-PPO: trained pairwise reward model + frozen SFT ref + LoRA policy + trainable critic.
Teaching implementation, not TRL production PPO.
"""
import argparse, json, random
import torch
from datasets import load_dataset
from peft import get_peft_model,AutoPeftModelForSequenceClassification
from src.common import causal_model,tokenizer_for,lora,dtype,seed_everything,device_info,save_meta
from src.monitoring import RunLogger
from src.ppo import gae, only_logps, rollout_logps_and_values

p=argparse.ArgumentParser();p.add_argument("--model",default="outputs/sft_merged");p.add_argument("--reward",default="outputs/reward_adapter");p.add_argument("--steps",type=int,default=30);p.add_argument("--max-new",type=int,default=64);p.add_argument("--out",default="outputs/ppo_adapter");p.add_argument("--ppo-epochs",type=int,default=2);p.add_argument("--run-name",default="");args=p.parse_args()
seed_everything();info=device_info();print(info)
dev=torch.device("cuda:0")
tok=tokenizer_for(args.model)
ref=causal_model(args.model).to(dev).eval()
for x in ref.parameters():x.requires_grad_(False)
rm=AutoPeftModelForSequenceClassification.from_pretrained(args.reward,torch_dtype=dtype(),num_labels=1,is_trainable=False).to(dev).eval()
rm.config.pad_token_id=tok.pad_token_id
for x in rm.parameters():x.requires_grad_(False)
policy=get_peft_model(causal_model(args.model),lora()).to(dev)
policy.config.use_cache=False
policy.enable_input_require_grads()
policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
value_head=torch.nn.Linear(policy.config.hidden_size,1).to(dev,dtype=torch.float32)
torch.nn.init.zeros_(value_head.weight);torch.nn.init.zeros_(value_head.bias)
optimizer=torch.optim.AdamW(list(p for p in policy.parameters() if p.requires_grad)+list(value_head.parameters()),lr=1.5e-5)
rows=load_dataset("json",data_files="data/pref_train.jsonl",split="train")
logger=RunLogger("ppo",args.run_name or None,metadata={"base_model":args.model,"reward_model":args.reward,"output_dir":args.out})
logs=[]
verified_policy_gradient = False
for step in range(args.steps):
    policy.eval()
    row=rows[step%len(rows)]
    msgs=row["chosen"][:-1]
    ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(dev)
    if ids.shape[1]>200: ids=ids[:,-200:]
    with torch.no_grad():
        seq=policy.generate(ids,max_new_tokens=args.max_new,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,pad_token_id=tok.pad_token_id,use_cache=True)
        old_logp,old_values=rollout_logps_and_values(policy,value_head,seq,ids.shape[1],grad=False)
        ref_logp=only_logps(ref,seq,ids.shape[1])
        # Use the learned RM, not a fabricated human label. RM is a proxy for preferences.
        score=float(rm(input_ids=seq,attention_mask=torch.ones_like(seq)).logits.reshape(-1)[0].float().item())
        rm_reward=float(torch.tanh(torch.tensor(score/5.0)).item())
        rewards=-0.02*(old_logp-ref_logp)
        rewards[-1]+=rm_reward
        adv,returns=gae(rewards,old_values)
        # Keep the temporal ordering; centered/scaled for numerical stability.
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
    policy.train()
    # Retain checkpointing while scoring the same dropout-free rollout policy.
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    for _ in range(args.ppo_epochs):
        new_logp,new_values=rollout_logps_and_values(policy,value_head,seq,ids.shape[1],grad=True)
        ratio=(new_logp-old_logp.detach()).clamp(-10,10).exp()
        clipped=ratio.clamp(0.8,1.2)
        policy_loss=-torch.minimum(ratio*adv,clipped*adv).mean()
        value_loss=0.5*(new_values-returns).square().mean()
        loss=policy_loss+0.5*value_loss
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad]+list(value_head.parameters()),1.0)
        if not verified_policy_gradient and adv.abs().max().item() > 0:
            verified_policy_gradient = any(
                param.grad is not None and torch.count_nonzero(param.grad).item()
                for name, param in policy.named_parameters() if "lora_" in name
            )
            if not verified_policy_gradient:
                raise RuntimeError("No LoRA policy gradient on a nonconstant advantage")
        optimizer.step()
    log_ratio=(new_logp.detach()-old_logp.detach())
    ratio_det=log_ratio.clamp(-10,10).exp()
    item=dict(
        step=step,
        loss=round(float(loss.detach()),5),
        policy_loss=round(float(policy_loss.detach()),5),
        value_loss=round(float(value_loss.detach()),5),
        proxy_reward=round(rm_reward,5),
        mean_reward=round(float(rewards.sum()),5),
        episode_return=round(float(rewards.sum()),5),
        reward_model_logit=round(score,5),
        response_length=int(seq.shape[1]-ids.shape[1]),
        prompt_length=int(ids.shape[1]),
        rollout_tokens=int(seq.shape[1]-ids.shape[1]),
        sampled_kl=round(float((old_logp-ref_logp).mean()),5),
        approx_kl=round(float(((ratio_det-1.0)-log_ratio).mean()),5),
        clip_fraction=round(float(((ratio_det-1.0).abs()>0.2).float().mean()),5),
        advantage_mean=round(float(adv.mean()),5),
        advantage_std=round(float(adv.std(unbiased=False)),5),
        value_mean=round(float(old_values.mean()),5),
        return_mean=round(float(returns.mean()),5),
        grad_norm=round(float(grad_norm),5),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )
    print(item,flush=True);logs.append(item);logger.log(item,step=step)
policy.save_pretrained(args.out);tok.save_pretrained(args.out)
import pathlib
pathlib.Path(args.out).mkdir(parents=True,exist_ok=True)
torch.save(value_head.cpu().state_dict(),pathlib.Path(args.out)/"value_head.pt")
with open(pathlib.Path(args.out)/"ppo_log.jsonl","w",encoding="utf-8") as f:
    for r in logs:f.write(json.dumps(r)+"\n")
monitor_dir=logger.finish()
save_meta(args.out,{"base":args.model,"stage":"PPO","note":"small-batch on-policy PPO teaching implementation with reward model and KL to frozen SFT","device":info,"monitor_dir":str(monitor_dir)})
