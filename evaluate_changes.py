import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

your_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=your_api_key
)

with open('change_types.json', 'r') as f:
    change_types = json.load(f)

with open('data_points.json', 'r') as f:
    data_points = json.load(f)

def get_input_text_data(input_text_name):
    
    folder_path = "text_data\\" + input_text_name
    original_file = input_text_name + "_v0.txt"
    updated_file = input_text_name + "_v1.txt"

    original_file_path = os.path.join(folder_path, original_file)
    updated_file_path = os.path.join(folder_path, updated_file)

    with open(original_file_path, 'r') as file0:
        original_text = file0.read()

    with open(updated_file_path, 'r') as file1:
        updated_text = file1.read()

    return original_text, updated_text

input_text_name = "adam"

original_text, updated_text = get_input_text_data(input_text_name)