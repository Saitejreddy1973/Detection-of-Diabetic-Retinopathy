import torch

weights_path = "models\weights.pt"
state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
