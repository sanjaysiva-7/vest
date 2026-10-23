import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
