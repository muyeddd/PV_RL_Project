import torch

print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("GPU数量:", torch.cuda.device_count())