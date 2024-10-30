# BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow
# Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  ZoomBlur

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/clean \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption ''

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/biterror \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'BitError'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/cameracrash \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'CameraCrash'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/fog \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'Fog'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/h256 \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
    --corruption 'H256ABRCompression'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/lowlight \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'LowLight'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/rain \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Rain'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/snow \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Snow'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/bright \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Brightness'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/colorquant \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'ColorQuant'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/framelost \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'FrameLost'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/lens \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'LensObstacleCorruption'
   
python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/motion \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'MotionBlur'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/saturate \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'Saturate'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/zoom \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'ZoomBlur'

python llava1.5_dist.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data ../data/test/test_gpt_drivelm_train_300_final_v2_norm.json \
    --output ../res/llava-1.5-7b/baseline/water \
    --system_prompt ./prompts/baseline.txt \
    --num_processes 8 \
    --max_model_len 4096 \
   --corruption 'WaterSplashCorruption'
