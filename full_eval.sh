for step in 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000 32500 35000 37500 40000 42500 45000 47500 50000 52500 55000 57500 60000 62500 65000 67500 70000 72500 75000 77500 80000 82500 85000 87500 90000 92500 95000 97500 100000
do
    IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-gauss100 python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-train.py/runs/output_gauss100-200/checkpoint-$step.pth.tar --out output_gauss100-200-$step
done

for step in 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000
do
    IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-gauss100 python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-train.py/runs/output_gauss100-resume-0-140-200-0-60-60/checkpoint-$step.pth.tar --out output_gauss100-resume-0-140-200-0-60-60-$step
done

for step in 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000 32500 35000 37500 40000 42500 45000 47500 50000 52500 55000 57500 60000 62500 65000 67500 70000 72500 75000 77500 80000 82500 85000 87500 90000 92500 95000 97500 100000
do
    IMAGENET_PATH=/home/wenquan-lu/Workspace/noisy_ssl/simclr-pytorch/noisy_mini-imagenet-gauss100-denoised python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-train.py/runs/output_gauss100-200-denoised/checkpoint-$step.pth.tar --out output_gauss100-200-denoised-$step
done

