import torch

# Load the state_dicts
state_dict1 = torch.load("logs/exman-train.py/runs/output_gauss100-200-denoised/checkpoint-100000.pth.tar")["state_dict"]
state_dict2 = torch.load("logs/exman-train.py/runs/output_gauss100-200-denoised/checkpoint.pth.tar")["state_dict"]

# Check if they have the same keys
if state_dict1.keys() != state_dict2.keys():
    print("The state_dicts have different keys.")
else:
    all_equal = True  # Initialize a flag for overall equality
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            all_equal = False
            print(f"Mismatch found in key: {key}")
    if all_equal:
        print("The state_dicts are identical.")
    else:
        print("The state_dicts are not identical.")