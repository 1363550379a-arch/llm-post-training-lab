import os, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from .metrics import last_number, gsm_gold, pred_number, num_equal, normalize_math_answer

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_info():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch has no usable CUDA/HIP device. For Radeon install the matching ROCm PyTorch wheel on supported Linux; for NVIDIA install the matching CUDA wheel.")
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, vram_gb=round(p.total_memory/2**30,2), torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda, bf16=torch.cuda.is_bf16_supported())

def dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def tokenizer_for(model):
    tok=AutoTokenizer.from_pretrained(model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def causal_model(model):
    net=AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype(), low_cpu_mem_usage=True)
    net.config.use_cache=False
    return net

def lora(task="CAUSAL_LM", r=8):
    kw={"r":r,"lora_alpha":2*r,"lora_dropout":0.05, "bias":"none", "target_modules":["q_proj","k_proj","v_proj","o_proj"]}
    if task=="SEQ_CLS":
        kw.update(task_type="SEQ_CLS",modules_to_save=["score"])
    else: kw.update(task_type="CAUSAL_LM")
    return LoraConfig(**kw)

def save_meta(path, payload):
    import json
    Path(path).mkdir(parents=True,exist_ok=True)
    (Path(path)/"run_metadata.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")

