## Used to extract subset images

import os
import json
import shutil

def copy_images(json_file_path, destination_folder, corruption):
    """
    Copies all images specified in the 'image_paths' sections of the JSON file to a new folder.

    Parameters:
    - json_file_path (str): Path to the input JSON file.
    - destination_folder (str): Path to the folder where images will be copied.
    """

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Set to store all unique image paths
    image_paths = set()

    # Traverse the JSON data to extract image paths
    for scene_token, scene_data in data.items():
        key_frames = scene_data.get('key_frames', {})
        for key_frame_token, key_frame_data in key_frames.items():
            image_paths_dict = key_frame_data.get('image_paths', {})
            for camera, image_path in image_paths_dict.items():
                image_paths.add(image_path)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy each image to the destination folder
    for image_path in image_paths:
        # Ensure that the source image exists
        image_path = image_path.replace('../nuscenes/samples', f'data/val_data_corruption/{corruption}')
        if os.path.exists(image_path):
            # Extract the file name
            file_name = os.path.basename(image_path)
            # Define the destination path
            dest_path = image_path.replace('val_data_corruption', 'val_data_corruption_filter')
            if not os.path.exists(os.path.dirname(dest_path)):
                os.makedirs(os.path.dirname(dest_path))
            # Copy the file
            shutil.copy(image_path, dest_path)
            print(f"Copied {image_path} to {dest_path}")
        else:
            print(f"Image file {image_path} does not exist.")

    print("All images have been copied.")

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your JSON file
    json_file_path = 'data/QA_dataset_nus/drivelm_val_norm_final.json'
    # Replace with the desired destination folder path
    destination_folder = 'data/val_data_filter'

    # BitError    CameraCrash  Fog        H256ABRCompression      LowLight    Rain      Snow                   ZoomBlur
    # Brightness  ColorQuant   FrameLost  LensObstacleCorruption  MotionBlur  Saturate  WaterSplashCorruption

    corruptions = ['BitError', 'CameraCrash', 'Fog', 'H256ABRCompression', 'LowLight', 'Rain', 'Snow', 'ZoomBlur',
                    'Brightness', 'ColorQuant', 'FrameLost', 'LensObstacleCorruption', 'MotionBlur', 'Saturate', 'WaterSplashCorruption']

    for corruption in corruptions:
        copy_images(json_file_path, destination_folder, corruption)