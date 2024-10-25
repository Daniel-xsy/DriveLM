# This script convert the answer of GPT towards the format for evaluation

import json
import re
from collections import defaultdict

def load_json(input_file):
    """
    Load JSON data from a file.
    """
    with open(input_file, 'r') as f:
        return json.load(f)

def save_json(data, output_file):
    """
    Save JSON data to a file.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def parse_id(qa_id):
    """
    Parse the QA pair ID into scene_id, frame_id, task, and task_idx.
    
    Expected ID format: <scene_id>_<frame_id>_<task>_<task_idx>
    
    Returns:
        tuple: (scene_id, frame_id, task, task_idx)
    """
    parts = qa_id.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid ID format: {qa_id}")
    scene_id = parts[0]
    frame_id = parts[1]
    task = parts[2]
    task_idx = parts[3]
    return scene_id, frame_id, task, task_idx

def group_qa_pairs(data):
    """
    Group QA pairs by scene_id and frame_id.
    
    Returns:
        defaultdict: Nested dictionary {scene_id: {frame_id: [QA_pairs]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for qa in data:
        try:
            scene_id, frame_id, task, task_idx = parse_id(qa['id'])
            grouped[scene_id][frame_id].append({
                'task': task,
                'task_idx': int(task_idx),
                'question': qa['question'],
                'answer': qa['answer']
            })
        except ValueError as e:
            print(f"Skipping QA pair due to error: {e}")
    return grouped

def sort_tasks(qa_pairs):
    """
    Sort QA pairs based on task priority and task index.
    
    Task Priority: perception < prediction < planning < behavior
    """
    task_priority = {'perception': 1, 'prediction': 2, 'planning': 3, 'behavior': 4}
    
    def sort_key(qa):
        # Assign a high number if task is not recognized
        priority = task_priority.get(qa['task'], 99)
        return (priority, qa['task_idx'])
    
    return sorted(qa_pairs, key=sort_key)

def simplify_answer(question, answer):
    """
    Simplify the answer if the question contains multiple-choice options.
    
    If the question asks to select from options, extract only the option letter.
    Otherwise, return the original answer.
    
    Args:
        question (str): The question text.
        answer (str): The original answer.
        
    Returns:
        str: Simplified or original answer.
    """
    # Check if the question contains "select the correct answer from the following options"
    if re.search(r'select the correct answer from the following options', question, re.IGNORECASE):
        # Extract the option letter (e.g., "C.")
        match = re.match(r'^([A-D])\b', answer.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return answer

def transform_data(grouped_data):
    """
    Transform the grouped QA pairs into the desired output format.
    
    Returns:
        list: Transformed list of QA dictionaries.
    """
    transformed = []
    for scene_id, frames in grouped_data.items():
        for frame_id, qa_pairs in frames.items():
            # Sort the QA pairs based on task priority and task index
            sorted_qas = sort_tasks(qa_pairs)
            
            # Assign a new sequential index starting from 0
            for new_idx, qa in enumerate(sorted_qas):
                # Modify the ID
                new_id = f"{scene_id}_{frame_id}_{new_idx}"
                
                # Prepend "<image>\n" to the question
                new_question = f"<image>\n{qa['question']}"
                
                # Simplify the answer if necessary
                new_answer = simplify_answer(qa['question'], qa['answer'])
                
                transformed.append({
                    "id": new_id,
                    "question": new_question,
                    "answer": new_answer
                })
    return transformed

def main(input_file, output_file):
    # Load the original JSON data
    data = load_json(input_file)
    
    # Group QA pairs by scene and frame
    grouped = group_qa_pairs(data)
    
    # Transform the grouped data
    transformed = transform_data(grouped)
    
    # Save the transformed data to a new JSON file
    save_json(transformed, output_file)
    
    print(f"Transformation complete. Output saved to '{output_file}'.")

if __name__ == "__main__":
    # Example usage:
    # Replace 'input.json' with your input file path
    # Replace 'output.json' with your desired output file path
    main('/mnt/workspace/models/DriveLM/test/results/phi3.5/phi3.5_output.json', 
        '/mnt/workspace/models/DriveLM/test/results/phi3.5/phi3.5_output_convert.json')