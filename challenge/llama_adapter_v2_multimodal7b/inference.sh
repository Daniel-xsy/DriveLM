# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output.json \
#     --batch_size 8 \
#     --num_processes 8

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_biterror.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption BitError

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_camcrash.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption CameraCrash

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_fog.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption Fog

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_h256.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption H256ABRCompression

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_motion.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption MotionBlur

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_saturate.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption Saturate

# Brightness  ColorQuant   FrameLost  LowLight            Rain        Snow

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_bright.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption Brightness

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_colorquant.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption ColorQuant

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_framelost.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption FrameLost

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_lowlight.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption LowLight

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_rain.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption Rain

# python demo.py \
#     --llama_dir ckpts/Llama \
#     --checkpoint ft_output/checkpoint-3.pth \
#     --data test_llama.json  \
#     --output output_snow.json \
#     --batch_size 8 \
#     --num_processes 8 \
#     --corruption Snow


python demo.py \
    --llama_dir ckpts/Llama \
    --checkpoint ft_output/checkpoint-3.pth \
    --data test_llama.json  \
    --output output_noimage_wo_visual_token.json \
    --batch_size 8 \
    --num_processes 8 \
    --corruption NoImage