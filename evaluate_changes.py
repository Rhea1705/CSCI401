import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI

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

def get_prompt(prompt_filepath):
    with open(prompt_filepath, 'r') as file:
        prompt_full = file.read()

    prompt = prompt_full.split("[BREAK]")

    return prompt

# def get_change_type_and_degree(paragraph1, paragraph2):
#     # Dynamically create the prompt based on the descriptions in the JSON file
#     prompt = f"""
#     Compare the following two paragraphs and classify the type of change (Recommended Action, Claim, Evaluation) and the degree of change. 
#     Provide the change type and degree of change according to the following scales:

#     Recommended Action Scale: {change_types['Recommended_Action']['scale']}
#     Claim Scale: {change_types['Claim']['scale']}
#     Evaluation Scale: {change_types['Evaluation']['scale']}

#     Paragraph 1: "{paragraph1}"
#     Paragraph 2: "{paragraph2}"
    
#     Output the result in the following JSON format:
#     {{
#         "Type_of_Change": "Recommended_Action" or "Claim" or "Evaluation",
#         "Original_Action": str,  // Only if type is Recommended_Action
#         "Updated_Action": str,   // Only if type is Recommended_Action
#         "Original_Claim": str,  // Only if type is Claim
#         "Updated_Claim": str,   // Only if type is Claim
#         "Original_Evaluation": str,  // Only if type is Evaluation
#         "Updated_Evaluation": str,   // Only if type is Evaluation
#         "Change_Level": int  // Scale from 0 to 5 based on the type-specific scale
#     }}
#     """
#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#     model="gpt-4",
#     messages=messages,
#     max_tokens=150,
#     temperature=0,
#     )

#     try:
#         return json.loads(response.choices[0].message.content.strip())
#     except:
#         return None
    
def create_chain(prompt):
    llm = OpenAI(api_key=your_api_key)

    # Build chain links using LLMChain
    chain_links = []

    for prompt_segment in prompt:
        # Create a LLMChain for each prompt segment
        prompt_template = PromptTemplate(template=prompt_segment, input_variables=["text"])
        chain_links.append(LLMChain(llm=llm, prompt=prompt_template))

    # Now we combine these LLMChains sequentially
    return chain_links


input_text_name = "adam"
prompt_filepath = "test_prompt.txt"

original_text, updated_text = get_input_text_data(input_text_name)

prompt = get_prompt(prompt_filepath)

chain_links = create_chain(prompt)

results = {}
input_text = "Hello!" 

for i, chain in enumerate(chain_links):
    results[i] = chain.invoke({"text": input_text})

print(results)



