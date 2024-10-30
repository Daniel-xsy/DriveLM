# BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow
# Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  ZoomBlur

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/clean \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption ''

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/biterror \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'BitError'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/cameracrash \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'CameraCrash'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/fog \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'Fog'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/h256 \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'H256ABRCompression'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/lowlight \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'LowLight'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/rain \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Rain'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/snow \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Snow'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/bright \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Brightness'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/colorquant \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'ColorQuant'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/framelost \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'FrameLost'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/lens \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'LensObstacleCorruption'
   
python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/motion \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'MotionBlur'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/saturate \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Saturate'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/zoom \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'ZoomBlur'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-13b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-13b/baseline_filter/water \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'WaterSplashCorruption'
