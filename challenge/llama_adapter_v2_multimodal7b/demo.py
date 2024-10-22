import cv2
import llama
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Process, Manager, set_start_method
import math
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class LLamaDataset(Dataset):
    def __init__(self, data, transform=None, corruption=None):
        self.data = data
        self.transform = transform
        self.corruption = corruption

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        filename = data_item['image']
        ids = data_item['id']
        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']
        
        prompt = llama.format_prompt(question)

        if isinstance(filename, list):
            image_all = []
            for img_path in filename:
                if self.corruption is not None and len(self.corruption) > 1 and self.corruption != 'NoImage':
                    img_path = img_path.replace('nuscenes/samples', f'val_data_corruption/{self.corruption}')
                image = cv2.imread(img_path)
                if self.corruption == 'NoImage':
                    image = np.zeros_like(image)
                if image is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                image = Image.fromarray(image)
                if self.transform:
                    image = self.transform(image)
                image_all.append(image)
            image = torch.stack(image_all, dim=0)
        else:
            image = cv2.imread(filename)
            if image is None:
                raise FileNotFoundError(f"Image not found: {filename}")
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

        return image, prompt, ids, question, answer

def worker(rank, gpu_id, args, data_queue):
    # try:
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda")
    llama_dir = args.llama_dir

    # Load the model and preprocess within the subprocess
    model, preprocess = llama.load(args.checkpoint, llama_dir, llama_type="7B", device=device)
    model.eval()

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711])
    ])

    with open(args.data, 'r') as f:
        data_all = json.load(f)

    num_processes = args.num_processes
    data_per_process = math.ceil(len(data_all) / num_processes)
    start_idx = rank * data_per_process
    end_idx = min((rank + 1) * data_per_process, len(data_all))
    data_to_process = data_all[start_idx:end_idx]

    dataset = LLamaDataset(data_to_process, transform=transform_train, corruption=args.corruption)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for batch in tqdm(dataloader, desc=f"GPU {gpu_id}"):
        images, prompts, ids, questions, gt_answers = batch
        images = images.to(device)
        results = model.generate(images, prompts, temperature=0.2, top_p=0.1)
        
        for i, result in enumerate(results):
            data_queue.append({'id': ids[i], 'question': questions[i], 'answer': result})
    
    print(f"Process {rank} (GPU {gpu_id}) finished processing.")
    # except Exception as e:
    #     print(f"Error in process {rank} (GPU {gpu_id}): {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='LLAMA Adapter')
    parser.add_argument('--llama_dir', type=str, default="/path/to/llama_model_weights", help='path to llama model weights')
    parser.add_argument('--checkpoint', type=str, default="/path/to/pre-trained/checkpoint.pth", help='path to pre-trained checkpoint')
    parser.add_argument('--data', type=str, default="../test_llama.json", help='path to test data')
    parser.add_argument('--output', type=str, default="../output.json", help='path to output file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for parallel processing')
    parser.add_argument('--num_processes', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--corruption', type=str, default='', help='corruption type')
    return parser.parse_args()

def main():
    args = parse_arguments()

    num_gpus = args.num_processes
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs, but only {available_gpus} are available.")
    print(f"Using {num_gpus} GPUs")

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

    # Convert managed list to regular list before dumping to JSON
    with open(args.output, "w") as f:
        json.dump(list(data_queue), f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    # Set the start method to 'spawn' to avoid CUDA initialization issues
    try:
        set_start_method('spawn')
    except RuntimeError:
        # The start method has already been set
        pass
    main()