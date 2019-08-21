# Check for CUDA in system

import torch

if torch.cuda.is_available():
    print ("YES")
else:
    print("No")

