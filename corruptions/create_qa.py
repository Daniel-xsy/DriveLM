import json
import os
import random
import time
import argparse
import numpy as np
from tqdm import tqdm

import cv2
from rich import print

from utils.request import VLMAgent, NPImageEncode
from utils.utils import convert_text_to_json


def preprocess_ans(ans_json):
    new_ans_json = []
    for i, ans in enumerate(ans_json):
        ans = ans.strip('\n')
        ans = ans.strip('```json')
        ans = ans.strip('```')
        ans = ans.strip('\\')
        new_ans_json.append(ans)
    return new_ans_json


def main(args):

    corruption = os.path.basename(args.root).lower()
    print("The current corruption is: ", corruption)

    with open(args.sys_prompt, 'r') as f:
        SYSTEM_PROMPT = f.read()

    json_path = args.json_path

    # load original json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # resume the saved data
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            save_dict = json.load(f)
    else:
        save_dict = dict()
        
    for scene_token, data_item in tqdm(data.items()):
        key_frames = data_item['key_frames']

        save_dict[scene_token] = dict()
        for sample_token, data_ in key_frames.items():

            if scene_token in save_dict and sample_token in save_dict[scene_token]:
                continue
            save_dict[scene_token][sample_token] = dict()

            perception_qas = data_['QA']['perception']
            prediction_qas = data_['QA']['prediction']
            planning_qas = data_['QA']['planning']
            behavior_qas = data_['QA']['behavior']
            image_paths = data_.pop('image_paths')

            save_dict[scene_token][sample_token]['image_paths'] = image_paths
            save_dict[scene_token][sample_token]['robust_qas'] = []

            # init GPT-4V-based driver agent
            gpt4v = VLMAgent(api_key=args.api_key)

            # close loop simulation
            total_start_time = time.time()
            success = False
            while not success:
                gpt4v.addTextPrompt(SYSTEM_PROMPT)
                gpt4v.addTextPrompt(f"The current corruption is: {corruption}\n")
                for cam, img_path in image_paths.items():
                    # TODO: modify the hardcode here
                    img_path_ = img_path.replace('../nuscenes/samples', args.root)
                    img = cv2.imread(img_path_)
                    if img is None:
                        raise ValueError(f"Image not found: {img_path_}")

                    gpt4v.addImageBase64(NPImageEncode(img))
                gpt4v.addTextPrompt(str(data_))
                # get the decision made by the driver agent
                (
                    ans,
                    prompt_tokens, completion_tokens,
                    total_tokens, timecost
                ) = gpt4v.convert_image_to_language()

                if ans is not None:
                    ans_json = ans.split('\n')
                    ans_json = preprocess_ans(ans_json)
                    for ans_json_ in ans_json:
                        ans_json_ = convert_text_to_json(ans_json_)
                        if ans_json_ is not None:
                            save_dict[scene_token][sample_token]['robust_qas'].append(ans_json_)
                    if len(ans_json) != 0:
                        success = True

            with open(args.save_path, 'w') as f:
                json.dump(save_dict, f, indent=4)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Generate QA pairs for corrupted images')
    argparser.add_argument('api_key', type=str, help='API key for OpenAI')
    argparser.add_argument('--root', type=str, help='Path to the corruption set')
    argparser.add_argument('--json_path', type=str, help='Path to the folder containing the raw json file')
    argparser.add_argument('--save_path', type=str, help='Path to the folder to save the generated corruption QAs.')
    argparser.add_argument('--sys-prompt', type=str, help='Path to the default system prompts.')
    argparser.add_argument('--vis', action='store_true', help='Visualize the corrupted images.')
    args = argparser.parse_args()

    main(args)
