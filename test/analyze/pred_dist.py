import re
import json
from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the specific choices for general driving actions
direction_choices = ["Going ahead", "Turn left", "Turn right", "Drive backward"]

# Regex patterns for extracting steering and speed
steering_pattern = r"(steering to the|slightly steering to the) (\w+)"
speed_pattern = r"driving (\w+)"

# Function to determine if a question is ABCD multiple-choice or Yes/No
def determine_question_type(question):
    pattern = r'\bA\.\s|B\.\s|C\.\s|D\.\s'
    if re.search(pattern, question):
        return 'abcd'
    else:
        return 'yesno'

# Function to extract ABCD content
def extract_abcd_content(answer, question):
    abcd_pattern = r'(A\.\s.*?)(?:B\.\s|$)(.*?)(?:C\.\s|$)(.*?)(?:D\.\s|$)(.*)'
    match = re.search(abcd_pattern, question)
    if match:
        answer_options = {
            'A': match.group(1).strip(),
            'B': match.group(2).strip(),
            'C': match.group(3).strip(),
            'D': match.group(4).strip()
        }
        return answer_options.get(answer, '').strip('A.').strip()
    return ''

# Function to shorten detailed answers for steering and speed
def shorten_abcd_content(content):
    if content in direction_choices:
        return content
    else:
        # Extract steering and speed
        steering_match = re.search(steering_pattern, content)
        speed_match = re.search(speed_pattern, content)
        
        # Generate shortened description
        steering_direction = ""
        if steering_match:
            steering_direction = f"slightly {steering_match.group(2)}" if "slightly" in steering_match.group(1) else steering_match.group(2)
        
        speed = speed_match.group(1) if speed_match else "not moving"
        return f"{steering_direction} {speed}".strip()

def extract_yesno_content(answer):
    if 'yes.' in answer.lower():
        return 'yes'
    elif 'no.' in answer.lower():
        return 'no'
    return None

# Function to analyze ABCD and Yes/No questions in the JSON file
def analyze_data_distribution(data, pred_data):
    abcd_confusion = defaultdict(Counter)
    yesno_confusion = defaultdict(Counter)

    for scene_token, key_frames in data.items():
        for key_frame_token, details in key_frames['key_frames'].items():
            assert 'QA' in details
            idx = 0
            for task, items in details['QA'].items():
                for item in items:
                    assert 'Q' in item and 'A' in item
                    question = item['Q']
                    answer = item['A']

                    question_type = determine_question_type(question)
                    
                    pred_token = f'{scene_token}_{key_frame_token}_{idx}'
                    pred_answer = pred_data[pred_token]['answer']

                    if question_type == 'abcd':
                        gt_content = extract_abcd_content(answer.strip(), question)
                        pred_content = extract_abcd_content(pred_answer.strip(), question)
                        if gt_content and pred_content:
                            if gt_content in direction_choices:
                                abcd_confusion[gt_content][pred_content] += 1

                    elif question_type == 'yesno':
                        gt_answer = extract_yesno_content(answer)
                        pred_answer = extract_yesno_content(pred_answer)
                        if gt_answer and pred_answer:
                            yesno_confusion[gt_answer][pred_answer] += 1

                    idx += 1

    return abcd_confusion, yesno_confusion

def plot_heatmap(confusion_data, title):
    # Convert the confusion data to a DataFrame and fill NaN with 0
    df = pd.DataFrame(confusion_data, index=['Turn left', 'Going ahead', 'Drive backward', 'Turn right', 'Stopped', 'Back up']).fillna(0)
    custom_order = ["Going ahead", "Turn left", "Drive backward", "Turn right", "Stopped", "Back up", "Reverse parking"]
    df = df.reindex(custom_order, fill_value=0)
    
    # Plot the heatmap with annotations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, annot=True, fmt='g', cmap="YlGnBu", cbar=True)
    
    # Adjust labels and rotation
    plt.xlabel('Prediction')
    plt.xticks(rotation=45)
    plt.ylabel('Ground Truth')
    plt.yticks(rotation=0)
    
    # Set title and layout
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot as a PDF
    plt.savefig(f'{title}.pdf')

if __name__ == '__main__':
    json_file = '/home/shaoyux/models/DriveLM/data/QA_dataset_nus/drivelm_val_norm_final.json'
    pred_file = '/home/shaoyux/models/DriveLM/res/gpt4o_mini/1025_rc2/clean_p1p4_convert.json'
    
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    pred_data = {pred_data[i]["id"]: pred_data[i] for i in range(len(pred_data))}
        
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Analyze the data
    abcd_confusion, yesno_confusion = analyze_data_distribution(json_data, pred_data)

    # Plot heatmaps
    print("ABCD Question Confusion Matrix:")
    plot_heatmap(abcd_confusion, "ABCD Question Confusion Matrix")

    # print("\nYes/No Question Confusion Matrix:")
    # plot_heatmap(yesno_confusion, "Yes/No Question Confusion Matrix")