# BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow
# Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  ZoomBlur


python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_norm.json \
    --output ../res/phi3.5/baseline/clean \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 4 \
    --corruption ''

# python phi3.5_dist.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ../res/phi3.5/clean \
#     --system_prompt ./prompts/1025_rc2.txt \
#     --num_processes 4 \
#     --corruption ''

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption ''

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_biterror.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'BitError'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_cameracrash.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'CameraCrash'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_fog.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'Fog'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_h256.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'H256ABRCompression'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_lowlight.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'LowLight'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_rain.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'Rain'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_snow.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'Snow'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_bright.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'Brightness'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_colorquant.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'ColorQuant'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_framelost.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'FrameLost'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_motion.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'MotionBlur'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_saturate.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'Saturate'

# python vlm.py \
#     --model 'microsoft/Phi-3.5-vision-instruct' \
#     --data ../data/test/test_gpt_norm.json \
#     --output ./results/phi3.5/phi3.5_output_zoom.json \
#     --system_prompt ./prompts/1017_fix.txt \
#     --num_processes 8 \
#     --corruption 'ZoomBlur'