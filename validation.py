import json
from scipy.stats import kendalltau
from typing import Dict, Any, List, ClassVar
import jsonschema
from jsonschema import validate
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from evaluate_changes import get_prompt, get_input_text_data, get_chat_prompt_template
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import BooleanOutputParser
from dotenv import load_dotenv
import os

class ScoringOutputParser:
    def __init__(self, samples: List[str]):
        self.schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                f"{sample} Total Score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3
                }
                for sample in samples
            },
            "required": [f"{sample} Total Score" for sample in samples]
        }

    def parse(self, text: str) -> bool:
        try:
            data = json.loads(text)
            validate(instance=data, schema=self.schema)
            return data
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError) as e:
            print(f"LLM response failed: Parsing or validation error. Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error. Error: {e}")  
            raise

class EvaluateChangesOutputParser(BooleanOutputParser):

    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "Recommended_Action": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            },
            "Claim": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            },
            "Evaluation": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            },
            "Fact Changing": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            },
            "Phrasing": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            },
            "Addition": {
                "type": "object",
                "properties": {
                    "Score": {
                        "type": "string",
                        "pattern": "^[0-5]|null$"
                    },
                    "Justification": {
                        "type": "string"
                    },
                    "List of changes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["Score", "Justification", "List of changes"]
            }
        },
    }
    
    def parse(self, text: str) -> bool:
        try:
            data = json.loads(text)
            validate(instance=data, schema=EvaluateChangesOutputParser.schema)
            return data
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError) as e:
            print(f"LLM response failed: Parsing or validation error. Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error. Error: {e}")  
            raise

class Evaluation:

    severity_mapping = {
        "Recommended_Action": 3,
        "Evaluation": 3,
        "Claim": 2,
        "Addition": 2,
        "Fact Changing": 1,
        "Phrasing": 1
    }

    @staticmethod
    def parse_human_ratings(input):
        pass

    @staticmethod
    def calculate_max_score(output):
        max_score = 0
        for action in output.keys():
            if int(output[action]['Score']) > 0:
                max_score = max(max_score, Evaluation.severity_mapping[action])
        return max_score

    @staticmethod
    def evaluate(llm_scores, human_scores):
        """Using severity scores, calculate correlation statistics and generate a confusion matrix"""

        tau, p_value = kendalltau(llm_scores, human_scores)
        print(f"Kendall's Tau: {tau}")
        print(f"P-value: {p_value}")

        conf_matrix = confusion_matrix(llm_scores, human_scores)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix, cmap="Blues")
        fig.colorbar(cax)

        plt.title("Confusion Matrix")
        plt.show()

class BatchJob:

    def __init__(self, samples, output_filepath, differences_prompt_filepath, score_prompt_filepath, max_retries=2):
        if not os.path.exists(score_prompt_filepath):
            raise FileNotFoundError(f"Score prompt file not found: {score_prompt_filepath}")
        
        if not os.path.exists(differences_prompt_filepath):
            raise FileNotFoundError(f"Differences prompt file not found: {differences_prompt_filepath}")

        self.samples: List[str] = samples
        self.output_filepath: str = output_filepath
        self.max_retries: int = max_retries
        
        with open(score_prompt_filepath, 'r') as file:
            self.score_prompt = file.read()
        
        with open(differences_prompt_filepath, 'r') as file:
            self.differences_prompt = file.read()
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(model="gpt-4")

    def run(self):
        df = None

        if os.path.exists(self.output_filepath):
            df = pd.read_csv(self.output_filepath)
        else:
            df = pd.DataFrame(columns=["ID", "Passage 1", "Passage 2",
                            "Human Max Score", "Human Gist Score",
                            "AI Diff List", "AI Gist Score", "AI Max Score"]) 
            df.to_csv(self.output_filepath, index=False) 

        covered_data = set()
        for _, row in df.iterrows():
            covered_data.add(row['ID'])
        
        changes_parser = EvaluateChangesOutputParser()
        for sample in self.samples:
            # Skip data points already covered
            if sample in covered_data:
                continue

            # Obtain list of differences and set up prompt
            original_text, updated_text = get_input_text_data(sample)
            user_prompt = """
                Passage 1:
                {passage1}

                Passage 2:
                {passage2}
            """
            prompt = get_chat_prompt_template(user_prompt, original_text, updated_text)

            llm_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_parser=changes_parser
            )

            attempts = 0
            while attempts < self.max_retries:
                try:
                    response = llm_chain.run({
                        "output_language": "English",
                        "system_prompt_holder": self.differences_prompt
                    })
                    if response:
                        break
                except Exception:
                    attempts += 1   
            
            if attempts == self.max_retries:
                continue
            
            # Obtain AI rating (TODO)

            # Obtain human ratings (TODO)

            # Append it to CSV file
            new_row = {"ID": sample, 
                       "Passage 1": original_text,
                       "Passage 2": updated_text,
                       "Human Max Score": -1,
                       "Human Gist Score": -1,
                       "AI Diff List": response,
                       "AI Gist Score": -1,
                       "AI Max Score": Evaluation.calculate_max_score(response)}
            new_row_df = pd.DataFrame([new_row])
            new_row_df.to_csv(self.output_filepath, mode='a', index=False, header=False)

# Test it with prompt
temp_output_filepath = "temp_output.csv"
prompt_filepath = "list_of_changes_prompt_v2.txt"
score_prompt_filepath = "prompt_draft.txt"
randomly_generated_samples = ["forecasting", "adam"]

gen = BatchJob(randomly_generated_samples, temp_output_filepath, prompt_filepath, score_prompt_filepath, 0.1)
gen.run()