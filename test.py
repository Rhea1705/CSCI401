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
    
def get_change_type_and_degree(paragraph1, paragraph2):
    # Dynamically create the prompt based on the descriptions in the JSON file
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
    except:
        return None


results = []
for data_point in data_points:
    paragraph1 = data_point["paragraph1"]
    paragraph2 = data_point["paragraph2"]
    result = get_change_type_and_degree(paragraph1, paragraph2)
    if result:  
        results.append(result)

for result in results:
    print(json.dumps(result, indent=2))  
