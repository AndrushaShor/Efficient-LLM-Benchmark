### Use this to make sure machine has GPU installed and GPU is recognized
import torch
print(torch.cuda.is_available())