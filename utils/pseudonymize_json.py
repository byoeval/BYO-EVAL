import json
import random
import string
import os

# Function to generate random strings
def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Function to process the JSON and replace only the piece keys
def replace_piece_keys(d):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            # Replace only keys in "pieces" that contain numbers after the underscore
            if isinstance(key, str) and '_' in key and key.split('_')[0] in ['rook', 'bishop', 'queen', 'king', 'pawn', 'knight']:
                # Generate a new random string for each piece key
                new_key = key.split('_')[0] + '_' + generate_random_string()
            else:
                new_key = key
            
            # Recurse into the value
            new_dict[new_key] = replace_piece_keys(value)
        return new_dict
    elif isinstance(d, list):
        # If it's a list, apply the function recursively to each element
        return [replace_piece_keys(item) for item in d]
    else:
        return d
    
# Function to load a JSON file, process it, and save a modified JSON file
def process_json_file(input_filename:str, output_filename:str):
    with open(input_filename, 'r') as f:
        json_data = json.load(f)

    processed_json = replace_piece_keys(json_data)

    with open(output_filename, 'w') as o:
        json.dump(processed_json, o, indent=2)

    print(f"Modified JSON has been saved to {output_filename}")

# Function to process all JSON files in a folder and save the modified versions to another folder
def process_all_json_files_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(filename)
        if filename.endswith('.json'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            print(input_filepath)
            process_json_file(input_filepath, output_filepath)
    
if __name__ == "__main__":
    path = os.getenv("PROJECT_PATH")
    # Example usage
    input_folder = 'annotation/legend_json' #'count_chess/count/legend_json' 
    output_folder = 'annotation/legend_pseudo_json' #'count_chess/count/legend_pseudo_json' 

    #process_json_file('count_chess/count/legend_json/count_img_00078.json','output.json')

    process_all_json_files_in_folder(input_folder, output_folder)   