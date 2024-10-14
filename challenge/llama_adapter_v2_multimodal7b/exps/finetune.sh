#!/usr/bin/bash

LLAMA_PATH=ckpts/Llama
# path to pre-trained checkpoint
PRETRAINED_PATH=ckpts/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth
CONFIG=finetune_data_config.yaml
OUTPUT_DIR=ft_output/

mkdir -p $OUTPUT_DIR

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 4 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 &>> "$OUTPUT_DIR"/output.log &