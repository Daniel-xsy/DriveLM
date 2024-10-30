import json
import cv2
import base64
import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import glob
import time
from datetime import datetime

from utils.request import NPImageEncode, VLMAgent

def process_chunk(args):
    chunk_file, output_file, model, api_key, system_prompt_file, corruption = args
    results = []

    # Check if output_file exists and load existing results
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_results = {entry['id']: entry for entry in json.load(f)}
            except json.JSONDecodeError:
                existing_results = {}

    # Load the chunk data
    with open(chunk_file, 'r') as f:
        data = json.load(f)

    # Open system prompt
    with open(system_prompt_file, 'r') as f:
        system_prompt = f.read()

    # Process each entry
    for entry in tqdm(data, desc=f"Processing {chunk_file}"):
        entry_id = entry['id']
        if entry_id in existing_results:
            continue  # Skip already processed entries

        if not "perception" in entry_id and not "behavior" in entry_id:
            continue
        print(f"Processing ID: {entry_id}")
        
        ## TODO: hard code here
        ## only evaluate the perception and behavior
        if 'perception' not in entry_id and 'behavior' not in entry_id:
            continue
        ans = None

        while ans is None:
            try:
                # Initialize GPT4V instance
                gpt4v = VLMAgent(api_key=api_key, model=model, max_tokens=8192)
                # Add system prompt
                gpt4v.addTextPrompt(system_prompt)

                # Add images
                for img_path in entry['images']:
                    if len(corruption) > 1 and corruption != 'NoImage':
                        img_path = img_path.replace('nuscenes/samples', f'val_data_corruption/{corruption}')
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error loading image: {img_path}")
                        continue
                    img_base64 = NPImageEncode(img)
                    gpt4v.addImageBase64(img_base64)

                # Add text prompt
                question = entry['question']
                gpt4v.addTextPrompt(str(question))

                # Perform inference
                ans, prompt_tokens, completion_tokens, total_tokens, timecost = gpt4v.convert_image_to_language()

                if ans is None:
                    print(f"Received None answer for ID {entry_id}. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    # Store the result
                    result_entry = {'id': entry_id, 'question': entry['question'], 'answer': ans}
                    results.append(result_entry)

                    # Save intermediate results
                    existing_results[entry_id] = result_entry
                    with open(output_file, 'w') as f_out:
                        json.dump(list(existing_results.values()), f_out, indent=4)

            except Exception as e_inner:
                print(f"Error during inference for ID {entry_id}: {e_inner}")
                time.sleep(1)  # Wait before retrying

    # No need to handle exceptions here since we're continuously retrying until success

def main(args):
    # Create hidden temporary folder
    temp_dir = f'.temp_{args.model}'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Split input file into chunks
    with open(args.input, 'r') as f:
        data = json.load(f)

    num_processes = args.num_processes
    chunk_size = (len(data) + num_processes - 1) // num_processes

    chunk_files = []
    output_files = []
    for i in range(num_processes):
        chunk_data = data[i*chunk_size : (i+1)*chunk_size]
        if not chunk_data:
            continue
        chunk_file = os.path.join(temp_dir, f'chunk_{i}.json')
        output_file = os.path.join(temp_dir, f'output_{i}.json')
        chunk_files.append(chunk_file)
        output_files.append(output_file)

        # If chunk file doesn't exist, create it
        if not os.path.exists(chunk_file):
            with open(chunk_file, 'w') as f_chunk:
                json.dump(chunk_data, f_chunk, indent=4)

    # Prepare arguments for processing chunks
    pool_args = []
    for chunk_file, output_file in zip(chunk_files, output_files):
        pool_args.append((chunk_file, output_file, args.model, args.key, args.system_prompt, args.corruption))

    # Use multiprocessing Pool to process chunks
    with Pool(processes=num_processes) as pool:
        pool.map(process_chunk, pool_args)

    # Merge all output files
    results = []
    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f_out:
                try:
                    chunk_results = json.load(f_out)
                    results.extend(chunk_results)
                except json.JSONDecodeError:
                    print(f"Error reading {output_file}")
                    exit(1)
        else:
            print(f"Output file {output_file} does not exist.")

    # Save final results
    with open(args.output, 'w') as f_final:
        json.dump(results, f_final, indent=4)

    # Clean up temporary files and directory
    if args.clean_temp:
        for file_path in glob.glob(os.path.join(temp_dir, '*')):
            os.remove(file_path)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for GPT with multiprocessing')
    parser.add_argument('--key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--model', type=str, required=True, help='GPT version')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True, help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use')
    parser.add_argument('--corruption', type=str, default='', help='Corruption type')
    parser.add_argument('--clean_temp', action='store_true', help='Delete temporary files after processing')
    args = parser.parse_args()
    main(args)