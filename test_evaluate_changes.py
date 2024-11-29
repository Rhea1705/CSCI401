from validation import ScoringOutputParser, EvaluateChangesOutputParser, BatchJob
import os

# def test_scoring_output():
#     test_output = """{
#         "Sample 1 Total Score": 2,
#         "Sample 2 Total Score": 3,
#         "Sample 3 Total Score": 1
#     }"""
#     parser = ScoringOutputParser(['Sample 1', 'Sample 2', 'Sample 3'])
#     assert parser.parse(test_output) == True

# def test_malformed_scoring_output():
#     test_output = """{
#         "Sample 1 Total Score": 100,
#         "Samp
#     """
#     parser = ScoringOutputParser(['Sample 1', 'Sample 2', 'Sample 3'])
#     assert parser.parse(test_output) == False

# def test_diffs_output():
#     test_output = """
#         {Recommended Action Changes: {
#             score: 1,
#             justification: Blah blah,
#             list of changes: []
#         },
#         Addition Changes: {
#             score: 4,
#             justification: More blah,
#             list of changes: [{Description: She started jumping in this version}, {Description: She ate some ice cream}]
#         }}
#     """
#     parser = EvaluateChangesOutputParser()
#     assert parser.parse(test_output) == True

def test_evaluate_changes():
    temp_output_filepath = "temp_output.csv"
    prompt_filepath = "list_of_changes_prompt.txt"
    randomly_generated_samples = ["forecasting"]
    
    gen = BatchJob(randomly_generated_samples, temp_output_filepath, prompt_filepath, "prompt_draft.txt")
    gen.run()
