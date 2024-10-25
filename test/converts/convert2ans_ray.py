import json
import re
import os
from collections import defaultdict

def load_json_files(input_folder):
    """
    Load JSON data from all files in a directory, reading line-by-line.
    Each line in a file represents a separate JSON object.
    """
    data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            with open(os.path.join(input_folder, file_name), 'r') as f:
                for line in f:
                    try:
                        line_data = json.loads(line.strip())
                        data.append(line_data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line in {file_name} due to error: {e}")
    return data

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
                'answer': qa['generated_text']
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
        priority = task_priority.get(qa['task'], 99)
        return (priority, qa['task_idx'])
    
    return sorted(qa_pairs, key=sort_key)

def simplify_answer(question, answer):
    """
    Simplify the answer if the question contains multiple-choice options.
    
    If the question asks to select from options, extract only the option letter.
    Otherwise, return the original answer.
    """
    if re.search(r'select the correct answer from the following options', question, re.IGNORECASE):
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
            sorted_qas = sort_tasks(qa_pairs)
            
            for new_idx, qa in enumerate(sorted_qas):
                new_id = f"{scene_id}_{frame_id}_{new_idx}"
                new_question = f"<image>\n{qa['question']}"
                new_answer = simplify_answer(qa['question'], qa['answer'])
                
                transformed.append({
                    "id": new_id,
                    "question": new_question,
                    "answer": new_answer
                })
    return transformed

def main(input_folder, output_file):
    # Load all JSON data from files in the input directory
    data = load_json_files(input_folder)
    
    # Group QA pairs by scene and frame
    grouped = group_qa_pairs(data)
    
    # Transform the grouped data
    transformed = transform_data(grouped)
    
    # Save the transformed data to a new JSON file
    save_json(transformed, output_file)
    
    print(f"Transformation complete. Output saved to '{output_file}'.")

if __name__ == "__main__":
    # Example usage:
    # Replace 'input_folder' with your folder path containing JSON files
    # Replace 'output.json' with your desired output file path
    input_folder = '/home/shaoyux/models/DriveLM/res/phi3.5/prompt_1025_rc2/clean'
    output_file = '/home/shaoyux/models/DriveLM/res/phi3.5/prompt_1025_rc2/clean.json'
    main(input_folder, output_file)