import argparse
import json
import math
import os
from multiprocessing import Process, Manager, set_start_method
from vllm.sampling_params import SamplingParams

def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True, help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, default=8192, help='Maximum model length')
    parser.add_argument('--num_images_per_prompt', type=int, default=6, help='Maximum number of images per prompt')
    parser.add_argument('--corruption', type=str, default='', help='Corruption type')
    return parser.parse_args()

def worker(rank, gpu_id, args, data_queue):
    # Set the environment variable to limit CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import libraries after setting CUDA_VISIBLE_DEVICES
    import torch
    from vllm import LLM
    from PIL import Image
    from tqdm import tqdm

    llm = LLM(
        model=args.model,
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.num_images_per_prompt},
    )

    # Load data for this worker
    with open(args.data, 'r') as f:
        data_all = json.load(f)

    num_processes = args.num_processes
    data_per_process = math.ceil(len(data_all) / num_processes)
    start_idx = rank * data_per_process
    end_idx = min((rank + 1) * data_per_process, len(data_all))
    data_to_process = data_all[start_idx:end_idx]

    corruption = args.corruption
    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    # Process each entry
    for entry in tqdm(data_to_process, desc=f"GPU {gpu_id} processing"):
        entry_id = entry['id']
        # Load images and build image placeholders
        images = []
        image_placeholders = ''
        image_index = 1
        filenames = entry.get('images', [])
        assert isinstance(filenames, list)
        for img_path in filenames:
            # Handle corruption path if needed
            if corruption and len(corruption) > 1 and corruption != 'NoImage':
                img_path = img_path.replace('nuscenes/samples', f'val_data_corruption/{corruption}')
            if corruption == 'NoImage':
                # Generate a blank image
                img = np.zeros(224, 224, 3)
            else:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                except Exception as e:
                    print(f"Error loading image: {img_path}, error: {e}")
                    continue
            images.append(img)
            image_placeholders += f"<|image_{image_index}|>\n"
            image_index += 1

        if len(images) == 0:
            print(f"[Error] No images loaded for ID {entry_id}. Skipping entry.")
            exit()

        # Start the user prompt
        prompt = "<|user|>\n"
        prompt += image_placeholders

        # Add system prompt if provided
        if system_prompt:
            prompt += f"{system_prompt}\n"

        # Add question
        question = entry.get('question', '')
        prompt += str(question) + "\n<|end|>\n<|assistant|>\n"

        # Prepare the multi_modal_data
        multi_modal_data = {"image": images}

        # Call the LLM
        # TODO: hard code this steps here
        outputs = llm.generate({
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
        },
        sampling_params= SamplingParams(temperature=0.2, top_p=0.2, max_tokens=1000),
        use_tqdm=False)

        generated_text = outputs[0].outputs[0].text

        # Append the result to the data queue
        data_queue.append({
            'id': entry_id,
            'question': question,
            'answer': generated_text
        })

    print(f"Process {rank} (GPU {gpu_id}) finished processing.")

def main():
    args = parse_arguments()

    # Check for the availability of GPUs
    try:
        import torch
        available_gpus = torch.cuda.device_count()
    except ImportError:
        raise ImportError("PyTorch is required to determine the number of available GPUs.")

    num_gpus = args.num_processes
    if num_gpus > available_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs, but only {available_gpus} are available.")
    print(f"Using {num_gpus} GPUs.")

    manager = Manager()
    data_queue = manager.list()
    processes = []

    for rank in range(num_gpus):
        gpu_id = rank  # Assuming GPU IDs are 0 to num_gpus-1
        p = Process(target=worker, args=(rank, gpu_id, args, data_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save results to JSON file
    with open(args.output, "w") as f:
        json.dump(list(data_queue), f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    # Set the start method to 'spawn' to avoid CUDA initialization issues
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()