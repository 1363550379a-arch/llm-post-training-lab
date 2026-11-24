"""Pure-Python benchmark parsing, easy to test without GPU or Hugging Face dependencies."""
import re

def last_number(text):
    matches=re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?",str(text))
    return matches[-1].replace(",","") if matches else None

def gsm_gold(answer):
    return last_number(str(answer).split("####")[-1] if "####" in str(answer) else answer)

def pred_number(text):
    s=str(text)
    m=re.findall(r"(?:Final answer\s*[:：]|最终答案\s*[:：]|####)\s*([-+]?\d[\d,]*(?:\.\d+)?)",s,re.I)
    return m[-1].replace(",","") if m else None

def num_equal(a,b):
    try:return abs(float(a)-float(b))<1e-5*max(1,abs(float(b)))
    except (TypeError,ValueError):return False

def normalize_math_answer(text):
    return re.sub(r"<<[^>]*>>","",str(text)).strip()
