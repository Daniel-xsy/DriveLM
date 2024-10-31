import json
import re
from typing import List

def convert_text_to_json(input_text, required_keys=["Q", "A"]):
    # Define the structure of the JSON object we are looking for
    try:
        # Attempt to load the text as JSON
        json_data = json.loads(input_text)
        # Check if all required keys exist in the JSON object
        if all(key in json_data for key in required_keys):
            # If yes, return the JSON object as is
            return json_data
        else:
            # If not all required keys are present, return None
            raise
    except json.JSONDecodeError:
        # If the text is not valid JSON, return None
        print('Fail to convert text to json', input_text)
        return None
    

def preprocess_answer(answer: str) -> str:
    """
    Preprocesses the answer from a Vision-Language Model (VLM) output
    to extract and clean either Yes/No or ABCD answers.
    
    Args:
    - answer (str): The raw output from the VLM model.

    Returns:
    - str: The cleaned answer (Yes, No, A, B, C, or D).
    """
    # Convert the answer to lowercase for easier matching
    answer = answer.strip().lower()

    # Pattern to identify yes/no responses
    yes_no_patterns = {
        "yes": r"\byes\b",
        "no": r"\bno\b"
    }
    
    # Check for yes/no answers
    if re.search(yes_no_patterns["yes"], answer):
        return "Yes"
    elif re.search(yes_no_patterns["no"], answer):
        return "No"
    
    # Pattern to identify multiple choice answers (A, B, C, D)
    choice_patterns = {
        "A": r"\b(a|option a)\b",
        "B": r"\b(b|option b)\b",
        "C": r"\b(c|option c)\b",
        "D": r"\b(d|option d)\b"
    }
    
    # Check for ABCD answers
    for choice, pattern in choice_patterns.items():
        if re.search(pattern, answer):
            return choice
    
    # If nothing matches, return an empty string (or raise an error depending on your use case)
    return ""


def replace_system_prompt(prompt: str, image_paths: List[str]) -> str:
    """
    Replaces the specific sentence in the system prompt to reflect only the provided camera images.

    Args:
        prompt (str): The original system prompt containing the sentence to be replaced.
        image_paths (List[str]): A list of image file paths corresponding to different cameras.

    Returns:
        str: The updated system prompt with the specified sentence adjusted to include only the available cameras.
    """
    # Define the order of cameras
    camera_order = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]

    # Regular expression to extract camera names from image paths
    # Assumes camera name is the directory name after 'samples/'
    camera_pattern = r'samples/([^/]+)/'

    # Extract camera names from image paths
    extracted_cameras = []
    for path in image_paths:
        match = re.search(camera_pattern, path)
        if match:
            camera_name = match.group(1)
            if camera_name in camera_order:
                extracted_cameras.append(camera_name)
            else:
                raise ValueError(f"Unrecognized camera name '{camera_name}' in path '{path}'.")
        else:
            raise ValueError(f"Unable to extract camera name from path '{path}'.")

    # Remove duplicates while preserving order
    unique_cameras = []
    seen = set()
    for cam in extracted_cameras:
        if cam not in seen:
            unique_cameras.append(cam)
            seen.add(cam)

    # Order the cameras based on the predefined camera_order
    ordered_cameras = [cam for cam in camera_order if cam in unique_cameras]

    # Construct the new sentence
    if not ordered_cameras:
        raise ValueError("No valid camera images provided.")
    else:
        cameras_str = ", ".join(ordered_cameras)
        if len(ordered_cameras) == 1:
            new_sentence = f"You are provided with a single camera image: [{cameras_str}]."
        else:
            new_sentence = f"You are provided with {len(ordered_cameras)} camera images in the sequence [{cameras_str}]."

    # Define the original sentence to be replaced
    original_sentence_pattern = r"You are provided with up to six camera images in the sequence \[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT\]\."

    # Use regex to replace the original sentence with the new sentence
    updated_prompt, num_subs = re.subn(original_sentence_pattern, new_sentence, prompt)

    if num_subs == 0:
        print("Warning: Original sentence not found in the prompt. No replacement made.")

    return updated_prompt