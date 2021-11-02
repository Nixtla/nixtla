import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1


print(f'Device name: {torch.cuda.get_device_name(0)} \n')
print(f'Current device: {torch.cuda.current_device(0)} \n')
