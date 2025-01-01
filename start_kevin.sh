# shot1
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot1 python train.py --config configs/mini_imagenet.yaml --out output_shot1-200
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot1-denoised python train.py --config configs/mini_imagenet.yaml --out output_shot1-200-denoised

# shot3
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot3 python train.py --config configs/mini_imagenet.yaml --out output_shot3-200
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot3-denoised python train.py --config configs/mini_imagenet.yaml --out output_shot3-200-denoised

# shot10
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot10 python train.py --config configs/mini_imagenet.yaml --out output_shot10-200
IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-shot10-denoised python train.py --config configs/mini_imagenet.yaml --out output_shot10-200-denoised