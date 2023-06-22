CUDA_VISIBLE_DEVICES=1 python train.py \
    --config configs/vocoder.json \
    --training_steps 800000 \
    --use_lmdb