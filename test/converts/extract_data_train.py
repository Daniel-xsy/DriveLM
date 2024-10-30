## Filter data to have up to 100 samples each for "Going ahead", "Turn left", "Turn right"
## Filter data to have 100 samples each for "Going ahead", "Turn left", "Turn right"
## Ensure steering is balanced within each category
## Ensure samples come from different scene_tokens

import json
import re
import random
from collections import defaultdict

def parse_options(question_text):
    """
    Extract options from the question text.
    """
    options = {}
    # Regex to find options like "A. Some text."
    matches = re.findall(r'([A-D])\.\s*(.*?)\s*(?=[A-D]\.|$)', question_text, re.DOTALL)
    for option, text in matches:
        options[option.strip()] = text.strip()
    return options

def get_selected_option_text(options, answer):
    """
    Retrieve the text corresponding to the selected option.
    """
    return options.get(answer.strip(), "")

def categorize_perception(option_text):
    """
    Categorize perception into "Going ahead", "Turn left", or "Turn right".
    """
    option_text_lower = option_text.lower()
    if "going ahead" in option_text_lower or "going straight" in option_text_lower:
        return "Going ahead"
    elif "turn left" in option_text_lower:
        return "Turn left"
    elif "turn right" in option_text_lower:
        return "Turn right"
    else:
        return None

def categorize_behavior(option_text):
    """
    Categorize behavior into steering directions.
    """
    steering_options = ["Left", "Slightly Left", "Going Straight", "Slightly Right", "Right"]
    for steering in steering_options:
        if steering.lower() in option_text.lower():
            return steering
    return None

def collect_candidates(input_file):
    """
    Collect candidates categorized by perception and behavior.
    """
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Structure: category -> steering -> list of samples
    candidates = defaultdict(lambda: defaultdict(list))
    
    for scene_token, scene_data in input_data.items():
        key_frames = scene_data.get('key_frames', {})
        for key_frame_token, key_frame_data in key_frames.items():
            QA = key_frame_data.get('QA', {})
            
            # Process Perception Questions
            perception_QA = QA.get('perception', [])
            perception_category = None
            for qa in perception_QA:
                question_text = qa['Q']
                ground_truth_answer = qa['A']
                options = parse_options(question_text)
                selected_option_text = get_selected_option_text(options, ground_truth_answer)
                category = categorize_perception(selected_option_text)
                if category:
                    perception_category = category
                    break  # Assuming one relevant perception question per key frame
            
            if not perception_category:
                continue  # Skip if perception category not found
            
            # Process Behavior Questions
            behavior_QA = QA.get('behavior', [])
            steering_direction = None
            for qa in behavior_QA:
                question_text = qa['Q']
                ground_truth_answer = qa['A']
                options = parse_options(question_text)
                selected_option_text = get_selected_option_text(options, ground_truth_answer)
                steering = categorize_behavior(selected_option_text)
                if steering:
                    steering_direction = steering
                    break  # Assuming one relevant behavior question per key frame
            
            if not steering_direction:
                continue  # Skip if steering direction not found
            
            # Add to candidates
            candidates[perception_category][steering_direction].append(
                (scene_token, key_frame_token, key_frame_data)
            )
    
    return candidates

def sample_data(candidates, target_per_category=100):
    """
    Sample data ensuring balance in steering directions and unique scene_tokens.
    """
    sampled_data = {}
    steering_options = ["Left", "Slightly Left", "Going Straight", "Slightly Right", "Right"]
    samples_per_steering = target_per_category // len(steering_options)  # 20 each
    
    used_scene_tokens = set()
    
    for category in ["Going ahead", "Turn left", "Turn right"]:
        if category not in candidates:
            print(f"Category '{category}' not found in the data.")
            continue
        sampled_category = {}
        used_key_frame_tokens = set()
        for steering in steering_options:
            options_list = candidates[category].get(steering, []).copy()
            random.shuffle(options_list)
            count = 0
            for scene_token, key_frame_token, key_frame_data in options_list:
                # Add to sampled data
                if scene_token not in sampled_category:
                    sampled_category[scene_token] = {'key_frames': {}}
                sampled_category[scene_token]['key_frames'][key_frame_token] = key_frame_data
                used_key_frame_tokens.add(key_frame_token)
                count += 1
                if count >= samples_per_steering:
                    break
            if count < samples_per_steering:
                print(f"Not enough samples for category '{category}', steering '{steering}'. Needed {samples_per_steering}, got {count}.")
        
        # After balancing steering, check if we need more samples to reach target
        total_sampled = sum(len(kf['key_frames']) for kf in sampled_category.values())
        if total_sampled < target_per_category:
            remaining = target_per_category - total_sampled
            # Collect remaining samples from any steering within the category
            all_remaining = []
            for steering in steering_options:
                all_remaining.extend([
                    (st, kf_token, kf_data) 
                    for st, kf_token, kf_data in candidates[category][steering]
                    if kf_token not in used_key_frame_tokens
                ])
            random.shuffle(all_remaining)
            for scene_token, key_frame_token, key_frame_data in all_remaining:
                # Add to sampled data
                if scene_token not in sampled_category:
                    sampled_category[scene_token] = {'key_frames': {}}
                sampled_category[scene_token]['key_frames'][key_frame_token] = key_frame_data
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining > 0:
                print(f"Could not fully meet the target for category '{category}'. Remaining: {remaining}")
        
        # Add sampled category to the overall sampled data
        for scene_token, key_frame_data in sampled_category.items():
            key_frame_tokens = key_frame_data['key_frames'].keys()
            for key_frame_token in key_frame_tokens:
                if scene_token not in sampled_data.keys():
                    sampled_data[scene_token] = {'key_frames': {}}
                sampled_data[scene_token]['key_frames'][key_frame_token] = key_frame_data['key_frames'][key_frame_token]
    
    print(f"Total samples: {sum(len(kf['key_frames']) for kf in sampled_data.values())}")
    return sampled_data


def main():
    input_file = '/home/shaoyux/models/DriveLM/challenge/test_eval.json'  # Replace with your input file path
    output_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_train_final_v2.json'  # Output file to save the filtered data
    
    random.seed(42)  # For reproducibility; remove or change seed as needed
    
    candidates = collect_candidates(input_file)
    filtered_data = sample_data(candidates, target_per_category=100)
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered data has been saved to {output_file}")

if __name__ == "__main__":
    main()