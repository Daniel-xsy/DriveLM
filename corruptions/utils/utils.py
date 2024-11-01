import json

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
        if input_text.strip() == "":
            return None
        print(f'Fail to convert text to json: {input_text}')
        return None