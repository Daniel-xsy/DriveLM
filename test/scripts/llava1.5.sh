# BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow
# Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  ZoomBlur

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_norm.json \
    --output ../res/llava-1.5-13b/baseline/clean \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 4 \
    --max_model_len 4096 \
    --corruption ''