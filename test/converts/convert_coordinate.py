import re
import json
import numpy as np

# Function to normalize the coordinates from [1600, 900] to [0, 1]
def normalize_coordinates(coordinate, width=1600, height=900):
    normalized_x = coordinate[0] / width
    normalized_y = coordinate[1] / height
    return normalized_x, normalized_y

# Function to process the 'QA' part of the JSON
def normalize_qa_coordinates(qa):
    for tasks, items in qa.items():
        # Extract all coordinates in the answer
        for item in items:
            if 'A' in item:
                answer = item['A']
                coordinates = re.findall(r'\d+\.\d+', answer)

                # If we find coordinate pairs, we proceed
                if len(coordinates) % 2 == 0:
                    coordinates = np.array([float(x) for x in coordinates]).reshape(-1, 2)
                    
                    # Normalize each coordinate
                    for i in range(len(coordinates)):
                        coordinates[i] = normalize_coordinates(coordinates[i])

                    # Replace the original coordinates with the normalized values in the answer string
                    updated_answer = re.sub(r'(\d+\.\d+,\d+\.\d+)', lambda m: f"{coordinates[0][0]:.4f},{coordinates[0][1]:.4f}", answer)
                    item['A'] = updated_answer
            if 'Q' in item:
                question = item['Q']
                coordinates = re.findall(r'\d+\.\d+', question)

                # If we find coordinate pairs, we proceed
                if len(coordinates) % 2 == 0:
                    coordinates = np.array([float(x) for x in coordinates]).reshape(-1, 2)
                    
                    # Normalize each coordinate
                    for i in range(len(coordinates)):
                        coordinates[i] = normalize_coordinates(coordinates[i])

                    # Replace the original coordinates with the normalized values in the answer string
                    updated_question = re.sub(r'(\d+\.\d+,\d+\.\d+)', lambda m: f"{coordinates[0][0]:.4f},{coordinates[0][1]:.4f}", question)
                    item['Q'] = updated_question
    return qa

# Main function to normalize coordinates in the entire JSON structure
def normalize_json_coordinates(data):
    for scene_token, key_frames in data.items():
        for key_frame_token, details in key_frames['key_frames'].items():
            # Process 'QA' part if available
            if 'QA' in details:
                details['QA'] = normalize_qa_coordinates(details['QA'])
    return data


if __name__ == '__main__':
    # Load the JSON data
    json_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_val.json'
    output_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_val_norm.json'
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Convert the JSON structure
    normalized_json = normalize_json_coordinates(json_data)
    
    with open(output_file, 'w') as f:
        json.dump(normalized_json, f, indent=4)

