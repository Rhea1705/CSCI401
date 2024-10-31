import os
import zipfile  
from openai import OpenAI
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

your_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=your_api_key
)

with open('change_types.json', 'r') as f:
    change_types = json.load(f)

def get_change_type_and_degree(paragraph1, paragraph2):
    prompt = f"""
    Compare the following two paragraphs and classify the type of change (Recommended Action, Claim, Evaluation) and the degree of change. 
    Provide the change type and degree of change according to the following scales:

    Recommended Action Scale: {change_types['Recommended_Action']['scale']}
    Claim Scale: {change_types['Claim']['scale']}
    Evaluation Scale: {change_types['Evaluation']['scale']}

    Paragraph 1: "{paragraph1}"
    Paragraph 2: "{paragraph2}"
    
    Output the result in the following JSON format:
    {{
        "Type_of_Change": "Recommended_Action" or "Claim" or "Evaluation",
        "Original_Action": str,  // Only if type is Recommended_Action
        "Updated_Action": str,   // Only if type is Recommended_Action
        "Original_Claim": str,  // Only if type is Claim
        "Updated_Claim": str,   // Only if type is Claim
        "Original_Evaluation": str,  // Only if type is Evaluation
        "Updated_Evaluation": str,   // Only if type is Evaluation
        "Change_Level": int  // Scale from 0 to 5 based on the type-specific scale
    }}
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        temperature=0,
    )

    try:
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return None

def read_paragraphs_from_csv(file_path):
    df = pd.read_csv(file_path)  
    # get 'First Document' and 'Second Document' columns
    paragraphs = df[['First Document', 'Second Document']].values.tolist()  
    return paragraphs

# extract CSV file
def extract_csv_file(zip_file_path, extract_to_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

zip_file = 'paras.zip'  
extract_folder = 'extracted_csv' 
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

extract_csv_file(zip_file, extract_folder)

# get CSV file in the extraction folder
csv_files = [f for f in os.listdir(extract_folder) if f.endswith('.csv')]

if len(csv_files) != 1:
    raise ValueError("Expected exactly one CSV file in the extracted folder.")

csv_file_path = os.path.join(extract_folder, csv_files[0])

results = []
paragraphs = read_paragraphs_from_csv(csv_file_path)

# Compare each pair of paragraphs
for paragraph in paragraphs:
    # First Document
    paragraph1 = paragraph[0] 
    # Second Document 
    paragraph2 = paragraph[1]  

    result = get_change_type_and_degree(paragraph1, paragraph2)
    if result:  
        results.append(result)

output_file = 'results.json'
with open(output_file, 'w') as json_file:
    json.dump(results, json_file, indent=2)

print(f"Results have been written to {output_file}")