python demo.py \
    --llama_dir ckpts/Llama \
    --checkpoint ft_output/checkpoint-3.pth \
    --data test_llama.json  \
    --output output.json \
    --batch_size 4 \
    --num_processes 8