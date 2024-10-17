import json
import os
import time
import argparse
import cv2
from rich import print
from utils.request import VLMAgent, NPImageEncode
from utils.utils import convert_text_to_json
from multiprocessing import Pool
import yaml
import mmcv

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

    # resume from the last saved state
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            save_dict = json.load(f)
    # new start
    else:
        save_dict = dict()

    prog_bar = mmcv.ProgressBar(len(data))
    for scene_token, data_item in data.items():
        key_frames = data_item['key_frames']

        if scene_token not in save_dict:
            save_dict[scene_token] = dict()
        else:
            pass

        for sample_token, data_ in key_frames.items():

            if sample_token in save_dict[scene_token]:
                continue
            
            save_dict[scene_token][sample_token] = dict()
            image_paths = data_.pop('image_paths')

            save_dict[scene_token][sample_token]['image_paths'] = image_paths
            save_dict[scene_token][sample_token]['robust_qas'] = []

            # init GPT-4V-based driver agent
            gpt4v = VLMAgent(api_key=args.api_key)

            success = False
            while not success:
                gpt4v.addTextPrompt(SYSTEM_PROMPT)
                gpt4v.addTextPrompt(f"The current corruption is: {corruption}\n")
                for cam, img_path in image_paths.items():
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
                    ans_json = preprocess_ans(ans.split('\n'))
                    for ans_json_ in ans_json:
                        ans_json_ = convert_text_to_json(ans_json_)
                        if ans_json_ is not None:
                            save_dict[scene_token][sample_token]['robust_qas'].append(ans_json_)
                    if len(save_dict[scene_token][sample_token]['robust_qas']) != 0:
                        success = True


            with open(args.save_path, 'w') as f:
                json.dump(save_dict, f)
        
        # update progress bar using desc
        prog_bar.update(i, desc=f"{corruption}: ")

def run_main(config_item):
    # Extract the arguments from the config item
    args_copy = argparse.Namespace(
        api_key=global_args.api_key,
        root=config_item['root'],
        json_path=global_args.json_path,
        save_path=config_item['save_path'],
        sys_prompt=config_item['sys_prompt'],
        vis=global_args.vis
    )
    main(args_copy)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Generate QA pairs for corrupted images')
    argparser.add_argument('api_key', type=str, help='API key for OpenAI')
    argparser.add_argument('--json_path', type=str, help='Path to the folder containing the raw json file')
    argparser.add_argument('--config', type=str, help='Path to the YAML config file')
    argparser.add_argument('--vis', action='store_true', help='Visualize the corrupted images.')
    global_args = argparser.parse_args()

    # Read the YAML config file
    with open(global_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the list of corruption sets
    corruption_sets = config.get('corruption_sets', [])

    # Run main() in parallel for each corruption set
    with Pool(processes=len(corruption_sets)) as pool:
        pool.map(run_main, corruption_sets)