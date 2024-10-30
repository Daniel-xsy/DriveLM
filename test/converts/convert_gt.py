## Used to convert the groud truth

## 1. Convert the object identifiers to their descriptions
##      e.g., <c1,CAM_FRONT,486.7,486.7> --> a moving white sedan
##      Only convert the first question in "prediction" task

## 2. Convert perception chocie to three
##      e.g., delete 'stopped' choice that don't exist in the ground truth


import json
import re


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def create_id_to_description_map(original_data, camera_map):
    """
    Create a mapping from object IDs (c1, c2, c3, etc.) to their descriptions.
    Assumes the structure of the original JSON as provided.
    """
    id_map = {}
    for scene_token, scene_content in original_data.items():
        id_map[scene_token] = {}
        key_frames = scene_content.get('key_frames', {})
        for key_frame_token, key_frame_content in key_frames.items():
            id_map[scene_token][key_frame_token] = {}
            key_object_infos = key_frame_content.get('key_object_infos', {})
            for obj_id_full, obj_info in key_object_infos.items():
                # Extract the object identifier (e.g., c1, c2, c3)
                match = re.match(r'<(c\d+),([^,]+),[^>]+>', obj_id_full)
                if match:
                    obj_id = match.group(1)
                    camera_code = match.group(2)
                    description = obj_info.get('Visual_description', '')
                    id_map[scene_token][key_frame_token][obj_id] = {
                        'description': description.lower().strip('.'),
                        'camera_code': camera_map[camera_code]
                    }
    return id_map


def replace_object_identifiers(processed_data, id_map):
    """
    Replace object identifiers with descriptions in the processed data.
    Specifically targets the "prediction" tasks with tag 3.
    """
    for scene_token, scene_content in processed_data.items():
        key_frames = scene_content.get('key_frames', {})
        for key_frame_token, key_frame_content in key_frames.items():
            predictions = key_frame_content.get('QA', {}).get('prediction', [])
            for prediction in predictions:
                tags = prediction.get('tag', [])
                if 3 in tags:
                    original_answer = prediction.get('A', '')
                    # Find all object identifiers in the answer
                    # Pattern matches strings like <c3,CAM_FRONT,0.6058,0.5769>
                    pattern = r'<(c\d+),[^>]+>'
                    matches = re.findall(pattern, original_answer)
                    for obj_id in matches:
                        infos = id_map[scene_token][key_frame_token].get(obj_id, None)  # Fallback to obj_id if not found
                        assert infos is not None, f"Object ID {obj_id} not found in the mapping"
                        description = infos.get('description', '')
                        camera_desc = infos.get('camera_code', '')
                        replacement_text = f"{description} at {camera_desc}"
                        # Replace the full identifier with the description
                        full_pattern = f'<{obj_id},[^>]+>'
                        original_answer = re.sub(f'<{obj_id},[^>]+>', replacement_text, original_answer)
                    # Update the answer
                    prediction['A'] = original_answer
    return processed_data

def shorten_multiple_choice_options(processed_data):
    """
    Shortens multiple-choice options in the 'perception' section where tag=0.
    Keeps only the choices: 'Going ahead', 'Turn left', 'Turn right'.
    Renumbers the remaining choices to A, B, C and updates the answer accordingly.
    Ensures that exactly three options are present by adding missing desired choices.
    Raises ValueError in unexpected cases to aid debugging.
    
    Args:
        processed_data (dict): The processed JSON data to be modified.
    
    Returns:
        dict: The updated processed data with modified multiple-choice questions.
    
    Raises:
        ValueError: If the original correct option is not found in the desired choices.
    """
    # Define the desired choices to retain
    desired_choices = ["Going ahead", "Turn left", "Turn right"]
    
    for scene_token, scene_content in processed_data.items():
        key_frames = scene_content.get('key_frames', {})
        for key_frame_token, key_frame_content in key_frames.items():
            perception_qas = key_frame_content.get('QA', {}).get('perception', [])
            for qa in perception_qas:
                tags = qa.get('tag', [])
                if 0 in tags:
                    question = qa.get('Q', '')
                    answer = qa.get('A', '').strip()
                    
                    # Extract all multiple-choice options using regex
                    # Matches patterns like "A. Option1.", "B. Option2.", etc.
                    options = re.findall(r'([A-D])\.\s*([^\.]+)\.', question)
                    
                    # Filter options to keep only the desired choices
                    filtered_options = [(label, option.strip()) for label, option in options if option.strip() in desired_choices]
                    
                    # Identify which desired choices are missing
                    existing_options = [option for _, option in filtered_options]
                    missing_choices = [choice for choice in desired_choices if choice not in existing_options]
                    
                    # Refill to ensure exactly three options
                    for choice in missing_choices:
                        if len(filtered_options) >= 3:
                            break  # Only need three options
                        filtered_options.append((None, choice))  # Label will be assigned later
                    
                    # After refilling, check if we have exactly three options
                    if len(filtered_options) < 3:
                        raise ValueError(f"Not enough desired choices found to refill to three options for question: '{question}'")
                    
                    # Assign new labels A, B, C to the filtered choices
                    new_labels = ['A', 'B', 'C']
                    new_options = []
                    for idx, (label, option) in enumerate(filtered_options[:3]):
                        new_label = new_labels[idx]
                        new_options.append((new_label, option))
                    
                    # Reconstruct the question without the old options
                    # Assume the question ends before "Please select the correct answer..."
                    split_phrase = "Please select the correct answer from the following options:"
                    if split_phrase in question:
                        base_question = question.split(split_phrase)[0].strip() + f" {split_phrase}"
                    else:
                        # If the split phrase is not found, use the entire question up to the first option
                        base_question_parts = re.split(r'[A-D]\.\s*[^\.]+\.', question)
                        if len(base_question_parts) > 1:
                            base_question = base_question_parts[0].strip() + " Please select the correct answer from the following options:"
                        else:
                            # If no split phrase and no options found, append the split phrase
                            base_question = question.strip() + " Please select the correct answer from the following options:"
                    
                    # Append the new filtered and renumbered options
                    for label, option in new_options:
                        base_question += f" {label}. {option}."
                    
                    # Update the 'Q' field with the new question
                    qa['Q'] = base_question
                    
                    # Determine the text of the original correct answer
                    original_correct_option = None
                    for label, option in options:
                        if label == answer:
                            original_correct_option = option.strip()
                            break
                    
                    if original_correct_option in desired_choices:
                        # Find the new label corresponding to the original correct option
                        new_answer_label = None
                        for new_label, new_option in new_options:
                            if new_option == original_correct_option:
                                new_answer_label = new_label
                                break
                        if new_answer_label:
                            # Update the 'A' field with the new answer label
                            qa['A'] = new_answer_label
                        else:
                            # If the correct option was not retained, raise an error
                            qa['A'] = None
                            raise ValueError(f"Original correct option '{original_correct_option}' not found in desired choices for question: '{question}'")
                    else:
                        # If the original correct option is not in the desired choices, raise an error
                        qa['A'] = None
                        raise ValueError(f"Original correct option '{original_correct_option}' not found in desired choices for question: '{question}'")
                                
    return processed_data

def main():
    # File paths (modify these paths as needed)
    original_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/v1_1_train_nus.json'
    processed_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_train_300_final_v2_norm.json'
    output_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_train_300_final_v3_norm_abc.json'

    # Define camera code to description mapping
    camera_map = {
        "CAM_FRONT": "front camera",
        "CAM_BACK": "back camera",
        "CAM_FRONT_LEFT": "left camera",
        "CAM_FRONT_RIGHT": "right camera",
        "CAM_BACK_LEFT": "left rear camera",
        "CAM_BACK_RIGHT": "right rear camera"
    }

    # Load JSON data
    original_data = load_json(original_file)
    processed_data = load_json(processed_file)

    # Create mapping from object IDs to descriptions
    id_map = create_id_to_description_map(original_data, camera_map)

    # Replace object identifiers with descriptions in processed data
    updated_processed_data = replace_object_identifiers(processed_data, id_map)
    
    # Shorten multiple-choice options in the 'perception' section
    updated_processed_data = shorten_multiple_choice_options(updated_processed_data)

    # Save the updated processed data to a new file
    save_json(updated_processed_data, output_file)
    print(f"Updated processed data saved to {output_file}")

if __name__ == "__main__":
    main()