## Data distribution in the GT dataset
import re
import json
from collections import Counter

# Function to determine if a question is ABCD multiple-choice or Yes/No
def determine_question_type(question):
    """
    Determines if a question is ABCD multiple-choice or Yes/No based on its content.
    """
    # Check for presence of options A., B., C., D.
    pattern = r'\bA\.\s|B\.\s|C\.\s|D\.\s'
    if re.search(pattern, question):
        return 'abcd'
    else:
        return 'yesno'

# Function to extract ABCD content
def extract_abcd_content(answer, question):
    """
    Extract the content behind the question options A., B., C., D.
    """
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

def extract_yesno_content(answer):
    """
    Extract the content behind the question options Yes, No.
    """
    if 'yes.' in answer.lower():
        return 'yes'
    elif 'no.' in answer.lower():
        return 'no'
    else:
        return None


# Function to analyze ABCD and Yes/No questions in the JSON file
def analyze_data_distribution(data):
    abcd_count = Counter()
    yesno_count = Counter()

    # Traverse the JSON structure
    for scene_token, key_frames in data.items():
        for key_frame_token, details in key_frames['key_frames'].items():
            # Process 'QA' part if available
            if 'QA' in details:
                for task, items in details['QA'].items():
                    for item in items:
                        if 'Q' in item and 'A' in item:
                            question = item['Q']
                            answer = item['A']

                            # Determine question type
                            question_type = determine_question_type(question)
                            
                            if question_type == 'abcd':
                                # Extract the content behind the selected ABCD answer
                                selected_answer = answer.strip()
                                content = extract_abcd_content(selected_answer, question)
                                if content:
                                    abcd_count[content] += 1

                            elif question_type == 'yesno':
                                # Count Yes/No answers
                                answer = extract_yesno_content(answer)
                                if answer is not None:
                                    yesno_count[answer.strip().lower()] += 1

    return abcd_count, yesno_count


if __name__ == '__main__':

    json_file = 'data/QA_dataset_nus/drivelm_val_norm_final.json'
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Analyze the data
    abcd_distribution, yesno_distribution = analyze_data_distribution(json_data)

    # Display results
    print("ABCD Question Content Distribution:")
    for content, count in abcd_distribution.items():
        print(f"{content}: {count}")

    print("\nYes/No Answer Distribution:")
    for answer, count in yesno_distribution.items():
        print(f"{answer}: {count}")