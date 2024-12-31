for step in 150000 145000 140000
do
    python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-train.py/runs/000025/checkpoint-100000.pth.tar
done


python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-train.py/runs/000025/checkpoint-100000.pth.tar --out clean-100000