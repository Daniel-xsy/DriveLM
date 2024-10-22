## This script is used to create fixed answer question-answer pairs


import os
import json
import argparse
import random
import re
import cv2
from tqdm import tqdm


def count_cams(**kwargs):
    """
    Count corrupt cameras
    """
    image_paths = kwargs.get('image_paths', None)
    if image_paths is None:
        raise ValueError("The 'img_paths' argument is required for the 'count_cams' function.")
    count = 0
    for img_path in image_paths.values():
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path} in current working directory: {os.getcwd()}")
            continue
        if img.sum() < 1:
            count += 1
    return count    


def parse_qustions(file, **kwargs):
    """
    Parses a text file containing Q&A pairs and converts them into a list of dictionaries.
    
    Args:
        file (str): Path to the .txt file containing Q&A pairs.
        **kwargs: Current context information.
        
    Returns:
        list: A list of dictionaries with "Q" and "A" keys.
        
    Raises:
        NotImplementedError: If a referenced function in the answer is not implemented.
    """
    qa_pairs = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    
    # Regular expressions to match Q and A lines
    q_pattern = re.compile(r'^Q:\s*(.*?)\s*(<choices:\s*([^>]+)>)?$')
    func_pattern = re.compile(r'^A:\s*<func:\s*(\w+)\s*>$')
    answer_pattern = re.compile(r'^A:\s*(.+?)\.\s*$')
    
    # Iterate through lines two at a time (Q followed by A)
    i = 0
    while i < len(lines):
        q_line = lines[i].strip()
        a_line = lines[i+1].strip() if i+1 < len(lines) else ''
        
        # Parse the question line
        q_match = q_pattern.match(q_line)
        if not q_match:
            raise ValueError(f"Invalid question format at line {i+1}: {q_line}")
        
        question_text = q_match.group(1).strip()
        choices_str = q_match.group(3)
        choices = [choice.strip() for choice in choices_str.split(',')] if choices_str else None
        
        # Parse the answer line
        func_match = func_pattern.match(a_line)
        if func_match:
            func_name = func_match.group(1)
            # Attempt to retrieve the function from the global namespace
            func = globals().get(func_name)
            if not func:
                raise NotImplementedError(f"The function '{func_name}' is not implemented.")
            answer = func(**kwargs)
            # Convert the answer to string to match with choices
            answer = str(answer)
        else:
            answer_match = answer_pattern.match(a_line)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # Fallback: Remove 'A:' prefix and any trailing periods
                answer = a_line[2:].strip().rstrip('.')
        
        # If choices are present, handle multiple-choice formatting
        if choices:
            # Ensure the correct answer is among the choices
            if answer not in choices:
                raise ValueError(f"The answer '{answer}' is not among the provided choices for question: '{question_text}'")
            
            # Exclude the correct answer to sample incorrect choices
            incorrect_choices = [choice for choice in choices if choice != answer]
            if len(incorrect_choices) < 3:
                sampled_incorrect = incorrect_choices
            else:
                sampled_incorrect = random.sample(incorrect_choices, 3)
            
            # Combine correct answer with sampled incorrect choices
            options = sampled_incorrect + [answer]
            random.shuffle(options)
            
            # Assign labels A, B, C, D
            labels = ['A', 'B', 'C', 'D']
            labeled_options = {label: option for label, option in zip(labels, options)}
            
            # Find the label corresponding to the correct answer
            correct_label = next(label for label, option in labeled_options.items() if option == answer)
            
            # Format the question with labeled choices
            q_formatted = f"{question_text}: " + '. '.join([f"{label}. {option}" for label, option in labeled_options.items()])
            
            # Format the answer with the correct label and choice
            a_formatted = f"{correct_label}. {answer}"
        else:
            # If no choices, use the question and answer as is
            q_formatted = question_text
            a_formatted = answer
        
        # Append the formatted Q&A pair to the list
        qa_pairs.append({"Q": q_formatted, "A": a_formatted})
        
        # Move to the next Q&A pair
        i += 2
    
    return qa_pairs


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--question', type=str, required=True, help='Path to the fixed type question file')
    parser.add_argument('--corruption', type=str, required=True, help='Corruption type')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    save_dict = {}
    for scene_token, key_frames in tqdm(data.items()):
        save_dict[scene_token] = dict(key_frames=dict())
        for key_frame, data_info in key_frames['key_frames'].items():
            save_dict[scene_token]['key_frames'][key_frame] = dict()
            save_dict[scene_token]['key_frames'][key_frame]['QA'] = dict(robustness=[])

            image_paths = data_info.get('image_paths', None)
            if image_paths is None:
                raise ValueError("The 'img_paths' argument is required.")
            corruption_images_paths = dict()
            for cam, img_path in image_paths.items():
                corruption_images_paths[cam] = img_path.replace('../nuscenes/samples', f'data/val_data_corruption/{args.corruption}')
            data_info['image_paths'] = corruption_images_paths

            qa_pairs = parse_qustions(args.question, **data_info)
            save_dict[scene_token]['key_frames'][key_frame]['QA']['robustness'].extend(qa_pairs)
    
    with open(args.output_file, 'w') as f:
        json.dump(save_dict, f, indent=4)

    
