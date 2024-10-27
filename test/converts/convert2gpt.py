## Convert the QA dataset to the format required by GPT-3.5
## By default, we use this format for all our benchmark

import json
import os

def convert_to_gpt_format(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    output = []
    for scene_id, scene_content in data.items():
        key_frames = scene_content.get('key_frames', {})

        for frame_id, frame_content in key_frames.items():
            image_paths_dict = frame_content.get('image_paths', {})
            # Replace ".." with the actual path to your images directory
            image_paths = [image_paths_dict[key].replace("..", "data") for key in image_paths_dict.keys()]

            qa_sections = frame_content.get('QA', {})
            qa_categories = ['perception', 'prediction', 'planning', 'behavior']

            for category in qa_categories:
                qa_list = qa_sections.get(category, [])
                for idx, qa in enumerate(qa_list):
                    question = qa.get('Q', '')
                    answer = qa.get('A', '')

                    output_entry = {
                        "id": f"{scene_id}_{frame_id}_{category}_{idx}",
                        "images": image_paths,
                        "question": question,
                    }
                    output.append(output_entry)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    input_file = 'data/QA_dataset_nus/drivelm_val_norm_final.json'  # Update with your actual input file path
    output_file = 'data/test/test_gpt_norm_final.json'
    convert_to_gpt_format(input_file, output_file)