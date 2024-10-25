"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-node cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

import argparse
from typing import Any, Dict

import numpy as np
import ray
from PIL import Image
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"


def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='Maximum model length')
    parser.add_argument('--num_images_per_prompt', type=int, default=6,
                        help='Maximum number of images per prompt')
    parser.add_argument('--corruption', type=str, default='',
                        help='Corruption type')
    # Add hyperparameters
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.2,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_name, system_prompt, sampling_params,
                 num_images_per_prompt, max_model_len, tensor_parallel_size, corruption):
        # Create an LLM.
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": num_images_per_prompt},
            tensor_parallel_size=tensor_parallel_size
        )
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.corruption = corruption

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Generate texts from the prompts.
        sample_id = batch['id']
        questions = batch['question']
        filenames = batch['images']
        batch_size = len(filenames)

        # Load images and build image placeholders and multi_modal_data
        image_placeholders = [''] * batch_size
        multi_modal_datas = [dict(image=[]) for _ in range(batch_size)]

        for idx, sample_filenames in enumerate(filenames):
            # Handle corruption if needed
            image_index = 1
            for filename in sample_filenames:
                img_path = filename
                if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
                    img_path = img_path.replace('nuscenes/samples', f'val_data_corruption/{self.corruption}')
                if self.corruption == 'NoImage':
                    # Generate a blank image
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((224, 224))
                    except Exception as e:
                        print(f"Error loading image: {img_path}, error: {e}")
                        exit(1)
                placeholder = f"<|image_{image_index}|>"
                image_placeholders[idx] += placeholder + "\n"
                # Add to multi_modal_data
                multi_modal_datas[idx]["image"].append(img)
                image_index += 1

        # Build the prompt
        prompts = ["<|user|>\n"] * batch_size
        prompts = [prompt + image_placeholder for prompt, image_placeholder in zip(prompts, image_placeholders)]

        # Add system prompt if provided
        if self.system_prompt:
            prompts = [prompt + self.system_prompt for prompt in prompts]

        # Add question
        prompts = [prompt + question + "\n<|end|>\n<|assistant|>\n" for prompt, question in zip(prompts, questions)]

        # batch input list
        batch_inputs = [{"prompt": prompt, "multi_modal_data": multi_modal_data} for prompt, multi_modal_data in zip(prompts, multi_modal_datas)]

        # Generate outputs
        # outputs = self.llm.generate(
        #     {"prompt": prompt, "multi_modal_data": multi_modal_data},
        #     self.sampling_params
        # )

        outputs = self.llm.generate(
            batch_inputs,
            self.sampling_params,
            use_tqdm=False
        )

        generated_text = []
        for output in outputs:
            generated_text.append(output.outputs[0].text)
        return {
            "id": sample_id,
            "question": questions,
            "generated_text": generated_text,
        }


def main():
    args = parse_arguments()

    # Read system prompt
    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Set tensor_parallel_size per instance
    tensor_parallel_size = 1  # or args.tensor_parallel_size if needed

    # Set number of instances
    num_instances = args.num_processes

    # Read input data
    ds = ray.data.read_json(args.data)

    # For tensor_parallel_size > 1, create placement groups
    def scheduling_strategy_fn():
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        resources_kwarg["num_gpus"] = 1
    else:
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    # Apply batch inference for all input data
    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            sampling_params,
            args.num_images_per_prompt,
            args.max_model_len,
            tensor_parallel_size,
            args.corruption
        ),
        # Set the concurrency to the number of LLM instances
        concurrency=num_instances,
        # Specify the batch size for inference
        batch_size=32,
        **resources_kwarg,
    )

    # Peek first 10 results
    # outputs = ds.take(10)
    # for output in outputs:
    #     sample_id = output["id"]
    #     question = output["question"]
    #     generated_text = output["generated_text"]
    #     print(f"ID: {sample_id}, Question: {question}, Generated text: {generated_text}")

    # Write inference output data to output file
    ds.write_json(args.output)


if __name__ == '__main__':
    main()