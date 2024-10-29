## Filter out the original validation data, make the data more balanced
## Use the prediction of GPT-4o-mini as criteria for filtering

import json
import re


def load_predictions(prediction_file):
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    prediction_dict = {}
    for item in predictions:
        id_parts = item['id'].split('_')
        scene_token = id_parts[0]
        key_frame_token = id_parts[1]
        idx = int(id_parts[2])
        prediction_dict[(scene_token, key_frame_token, idx)] = item['answer']
    return prediction_dict

def parse_options(question_text):
    # Extract options from the question text
    options = {}
    # Regex to find options like "A. Some text."
    matches = re.findall(r'([A-D])\.\s*(.*?)\s*(?=[A-D]\.|$)', question_text)
    for option, text in matches:
        options[option.strip()] = text.strip()
    return options

def get_selected_option_text(options, answer):
    return options.get(answer.strip(), "")

def is_going_ahead(option_text):
    return "going ahead" in option_text.lower()

def is_going_straight(option_text):
    return "going straight" in option_text.lower()

def process_input_file(input_file, predictions, quota):
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    filtered_data = {}
    for scene_token, scene_data in input_data.items():
        key_frames = scene_data.get('key_frames', {})
        filtered_key_frames = {}
        for key_frame_token, key_frame_data in key_frames.items():
            QA = key_frame_data.get('QA', {})
            perception_condition_met = False
            perception_wrong = False
            behavior_condition_met = False
            behavior_wrong = False
            # Process perception questions
            perception_QA = QA.get('perception', [])
            for idx, qa in enumerate(perception_QA):
                question_text = qa['Q']
                ground_truth_answer = qa['A']
                options = parse_options(question_text)
                selected_option_text = get_selected_option_text(options, ground_truth_answer)
                if len(options) == 0:
                    continue
                elif not is_going_ahead(selected_option_text):
                    perception_condition_met = True
                    break  # Condition met, no need to check further
                else:
                    # Compare with prediction
                    prediction_answer = predictions.get((scene_token, key_frame_token, idx), "")
                    predicted_option_text = get_selected_option_text(options, prediction_answer)
                    if predicted_option_text != selected_option_text:
                        perception_wrong = True
                        break  # Condition met, no need to check further
            # Process behavior questions
            behavior_QA = QA.get('behavior', [])
            for idx, qa in enumerate(behavior_QA):
                question_text = qa['Q']
                ground_truth_answer = qa['A']
                options = parse_options(question_text)
                selected_option_text = get_selected_option_text(options, ground_truth_answer)
                if not is_going_straight(selected_option_text):
                    behavior_condition_met = True
                    break  # Condition met, no need to check further
                else:
                    # Compare with prediction
                    idx_in_predictions = len(QA['perception'] + QA['prediction'] + QA['planning'])
                    prediction_answer = predictions.get((scene_token, key_frame_token, idx_in_predictions), "")
                    predicted_option_text = get_selected_option_text(options, prediction_answer)
                    if predicted_option_text != selected_option_text:
                        behavior_wrong = True
                        break  # Condition met, no need to check further
            # Decide whether to keep the key frame
            if perception_condition_met or behavior_condition_met or (perception_wrong and behavior_wrong):
                # Keep the key frame
                if 'key_frames' not in filtered_data.get(scene_token, {}):
                    filtered_data[scene_token] = {'key_frames': {}}
                filtered_data[scene_token]['key_frames'][key_frame_token] = key_frame_data
            elif (perception_wrong or behavior_wrong) and quota > 0:
                # Keep the key frame
                if 'key_frames' not in filtered_data.get(scene_token, {}):
                    filtered_data[scene_token] = {'key_frames': {}}
                filtered_data[scene_token]['key_frames'][key_frame_token] = key_frame_data
                quota -= 1
        if filtered_data.get(scene_token, {}).get('key_frames', {}):
            filtered_data[scene_token]['key_frames'] = filtered_data[scene_token]['key_frames']
    return filtered_data

def main():
    input_file = 'data/QA_dataset_nus/drivelm_val_norm.json'  # Replace with your input file path
    prediction_file = '/home/shaoyux/models/DriveLM/res/gpt4o_mini/1017/gpt4o_output_convert.json'  # Replace with your prediction output file path
    output_file = 'data/QA_dataset_nus/drivelm_val_norm_300.json'  # Output file to save the filtered data
    
    # after the original filtering, we have 274 key frames left
    # add it up to 300
    quota = 26  
    
    predictions = load_predictions(prediction_file)
    filtered_data = process_input_file(input_file, predictions, quota)

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered data has been saved to {output_file}")

if __name__ == "__main__":
    main()