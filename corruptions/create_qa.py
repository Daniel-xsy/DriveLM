import json
import os
import random
import time

import cv2
from rich import print

from utils.request import VLMAgent, NPImageEncode
from imagecorruptions import corrupt


def sample_images_uniformly(image_files, truncate_ratio, sample_number, include_last=False):
    """
    Uniformly samples a subset of images from a specified range of a list, considering temporal distribution.

    Parameters:
    - image_files (list): A list of image file names.
    - truncate_ratio (float): The upper limit of the range to sample from as a ratio (0 to 1).
    - sample_number (int): The number of images to sample.

    Returns:
    - list: A list of sampled image file names.
    """
    # Calculate the index to truncate the list of images
    truncate_index = int(len(image_files) * truncate_ratio)

    # Truncate the image list to the specified range
    truncated_images = image_files[:truncate_index]

    # Calculate the step to evenly distribute the samples
    step = len(truncated_images) // sample_number

    # Ensure that we are able to sample the specified number of images
    if sample_number > len(truncated_images):
        print("Warning: Requested sample number is greater than the available images in the specified range.")
        sample_number = len(truncated_images)
        sampled_images = truncated_images
    else:
        # Sample images uniformly from the truncated list
        sampled_images = [truncated_images[i * step] for i in range(sample_number)]

    if include_last:
        sampled_images.append(image_files[-1])

    return sampled_images


SYSTEM_PROMPT = """You are an autonomous vehicle agent. You are provided with ground truth information about the scene and the key objects in the scene. You are also given a series of perception, prediction, planning, and behavior questions related to the scene. The input images are corrupted with various types of noise and distortions. Your task is to create new, diverse, corruption-related question-answer pairs (QA pairs) based on the given information and the corrupted images. Please generate the QA pairs following the format below:
```json
"perception": [
                {"Q": "questions here",
                 "A": "answers here"},
                ...],
"prediction": [
                {"Q": "questions here",
                 "A": "answers here"},
                ...],
"planning": [
                {"Q": "questions here",
                 "A": "answers here"},
                ...],
"behavior": [
                {"Q": "Predict the behavior of the ego vehicle.", # Please fix the question for behavior
                 "A": "answers here"}]
```
Please generate 3 questions for each category (perception, prediction, planning) and 1 questions for behavior based on the given information and the corrupted images.

Here are more instructions:
- Focus on the meaningful spatial information and describe the object with a single word or phrase.
- Even if you're not sure of the answer, directly answer it.
- Do not change the visual content described in the original response.

Now, begin:
"""

if __name__ == '__main__':

    json_path = '/mnt/workspace/models/DriveLM/data/QA_dataset_nus/v1_1_train_nus.json'

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for scene_token, data_item in data.items():
        scene_desc = data_item['scene_description']
        key_frames = data_item['key_frames']

        for sample_token, data_ in key_frames.items():
            key_object = data_['key_object_infos']
            perception_qas = data_['QA']['perception']
            prediction_qas = data_['QA']['prediction']
            planning_qas = data_['QA']['planning']
            behavior_qas = data_['QA']['behavior']
            image_paths = data_.pop('image_paths')

            # init GPT-4V-based driver agent
            gpt4v = VLMAgent(api_key='')

            # close loop simulation
            total_start_time = time.time()
            success = False
            while not success:
                gpt4v.addTextPrompt(SYSTEM_PROMPT)
                for cam, img_path in image_paths.items():
                    img = cv2.imread(img_path.replace('../nuscenes', '/mnt/workspace/models/DriveLM/data/nuscenes'))
                    if img is None:
                        raise ValueError(f"Image not found: {image_path_corruption}")
                    corrupt_img = corrupt(corruption_name='snow', severity=3, image=img)
                    gpt4v.addImageBase64(NPImageEncode(img))
                gpt4v.addTextPrompt(str(data_))
                # get the decision made by the driver agent
                (
                    ans,
                    prompt_tokens, completion_tokens,
                    total_tokens, timecost
                ) = gpt4v.convert_image_to_language()

                if ans is None:
                    continue

            if not success:
                continue
            ans['img_name'] = os.path.join(instance_folder, img_path)

            with open(save_path, 'w') as f:
                json.dump(ans, f)
