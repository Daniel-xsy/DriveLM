# BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow
# Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  ZoomBlur


python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/clean \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption ''

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/biterror \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'BitError'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/cameracrash \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'CameraCrash'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/fog \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'Fog'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/h256 \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'H256ABRCompression'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/lowlight \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'LowLight'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/rain \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'Rain'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/snow \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'Snow'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/bright \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'Brightness'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/colorquant \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'ColorQuant'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/framelost \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'FrameLost'

python vlm.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_norm.json \
    --output ./results/phi3.5/phi3.5_output_motion.json \
    --system_prompt ./prompts/1017_fix.txt \
    --num_processes 8 \
    --corruption 'MotionBlur'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/saturate \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'Saturate'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/zoom \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'ZoomBlur'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/len \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'LensObstacleCorruption'

python phi3.5_dist.py \
    --model 'microsoft/Phi-3.5-vision-instruct' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/phi3.5/baseline/water \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --corruption 'WaterSplashCorruption'