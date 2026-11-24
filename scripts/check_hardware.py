import json,platform,sys
import torch
from src.common import device_info
print(json.dumps({"python":sys.version.split()[0],"os":platform.platform(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"gpu":device_info() if torch.cuda.is_available() else None},indent=2,ensure_ascii=False))
