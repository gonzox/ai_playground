import json
from logging import exception
import os
from typing import List

# matplotlib for heatmap
import matplotlib.pyplot as plt
import numpy as np

#Pydantic
from pydantic import BaseModel, ValidationError

# Gemini model
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

#pandas
import pandas as pd

MODEL="gemini-2.0-flash"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

model = None

# Define the base model
class RepairQuesion(BaseModel):
  question: str
  answer: str
  equipment_problem: str
  tools_required: List[str]
  steps: List[str]
  safety_info: str
  tips: str

# retrieves the model and if not initilized, it initializes it with the model name
def get_model(generation_config: GenerationConfig = None):
    global model
    if model is None:
        model = genai.GenerativeModel(MODEL, generation_config=generation_config)
    return model

def generate_questions():
    # Vague Prompt
    vague_prompt = """Generate 20 questions on typical home DIY Repair, and return them as a valid JSON with the form of 
    { "question": "...", "answer": "...", "equipment_problem": "...", "tools_required": ["..."], "steps": ["..."], "safety_info": "...", "tips": "..." }
    for these categories:
    Appliance Repair ("appliance_repair"), Plumbing Repair ( plumbing_repair ), Electrical Repair ( electrical_repair ), HVAC Maintenance ( hvac_maintenance ), General Home Repair ( general_home_repair )
    """

    prompt = """
    <user_input>
    Generate 20 questions on typical home DIY Repair, and return them as a valid JSON with the form:
    </user_input>

    <format>
    { "question": "...", "answer": "...", "equipment_problem": "...", "tools_required": ["..."], "steps": ["..."], "safety_info": "...", "tips": "..." }
    </format>

    <instructions>
    The categories can be Appliance Repair ("appliance_repair"), Plumbing Repair ( plumbing_repair ), Electrical Repair ( electrical_repair ), HVAC Maintenance ( hvac_maintenance ), General Home Repair ( general_home_repair )

    Use these traits for each of them:

    1. Appliance Repair ( appliance_repair )
    Focus: Common household appliances

    Examples: Refrigerators, washing machines, dryers, dishwashers, ovens

    Expert Persona: Expert home appliance repair technician with 20+ years of experience

    Emphasis: Technical details and practical homeowner solutions

    2. Plumbing Repair ( plumbing_repair )
    Focus: Common plumbing issues

    Examples: Leaks, clogs, fixture repairs, pipe problems

    Expert Persona: Professional plumber with extensive residential experience

    Emphasis: Safety for homeowner attempts and realistic solutions

    3. Electrical Repair ( electrical_repair )
    Focus: SAFE homeowner-level electrical work

    Examples: Outlet replacement, switch repair, light fixture installation

    Expert Persona: Licensed electrician specializing in safe home electrical repairs

    Emphasis: Critical safety warnings and when to call professionals

    4. HVAC Maintenance ( hvac_maintenance )
    Focus: Basic HVAC maintenance and troubleshooting

    Examples: Filter changes, thermostat issues, vent cleaning, basic troubleshooting

    Expert Persona: HVAC technician specializing in homeowner maintenance

    Emphasis: Seasonal considerations and maintenance best practices

    5. General Home Repair ( general_home_repair )
    Focus: Common general repairs and maintenance

    Examples: Drywall repair, door/window problems, flooring issues, basic carpentry

    Expert Persona: Skilled handyperson with general home repair expertise

    Emphasis: Material specifications and practical DIY solutions
    </instructions>

    <context>
    Domain Expertise: Each template uses a specific expert persona

    Safety Focus: Strong emphasis on safety warnings and when to call professionals

    Practical Scope: Limited to repairs safe and practical for homeowners

    Structured Output: Consistent JSON format for downstream processing

    Realistic Scenarios: Focus on common, real-world repair situations

    </context>
    """

    # Configure and Instantiate the model
    generation_config = GenerationConfig(
        temperature=0.9, 
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
    )
    model = get_model(generation_config)

    # generate output
    # response = model.generate_content(prompt)
    response = model.generate_content(vague_prompt)

    # print the response
    print("Repair DYI questions data: ")
    print(response.text)

    # clean up to json text
    output = (
        response.text
        .strip()
        .removeprefix("```json")
        .removesuffix("```")
        .strip()
    )

    valid_outputs = []

    #validate structure of responses
    try:
        data = json.loads(output)
    except Exception as e:
        print("Bad Json format on the output")
        print(e)
    else:
        try:
            i=0
            for question in data:
                validated = RepairQuesion(**question)
                print("✅ Output for question " + str(i) + " is valid, adding to valid result list...")
                valid_outputs.append(question)
                i+=1
        except ValidationError as ve:
            print("Format validation failed:")
            print(ve)

    return valid_outputs

def evaluate_questions(questions: List[RepairQuesion], prompt:str):
    """
    Evaluates the questions based on the prompt
    """
    labeled_questions = []
    model = get_model()
    questions_str = json.dumps(questions)
    labeled_question = model.generate_content(prompt.format(QUESTIONS=questions_str))
    result=labeled_question.text.strip().removeprefix("```json").removesuffix("```").strip()
    #print("result: ", result)
    labeled_output = json.loads(result)

    
    return labeled_output

# creates a heatmap of evals based on the categories of evaluations
def visualize_evals(evals: pd.DataFrame):
    """
    Visualizes the evals based on the categories of evaluations
    """
    # convert evals to a numpy array summing the values
    data = evals.corr() # correlation matrix
   
    # print correlation matrix
    print("Correlation matrix: ")
    print(data)

    # create a heatmap of evals based on the categories of evaluations
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis')
    plt.colorbar(im)
    plt.show()

def to_dataframe(questions):
    """
    Converts a list of questions to a pandas dataframe
    """
    df = pd.DataFrame(data=questions, columns=["incomplete_answer", "safety_violations", "unrealistic_tools", "overcomplicated_solution", "missing_context", "poor_quality_tips"])
    return df


# generate questions from the model
# questions = generate_questions()

# load questions to a json file
questions = json.load(open('data/generated-questions.json'))

# human evaluations
human_evals = json.load(open('data/human-evals.json'))
#visualize_evals(to_dataframe(human_evals))

# seems to be some correlation between overcomplicated solution and unrealistic tools

first_pass_prompt = """
You are reviewing the quality of the support questions.
Given the answer and instructions generated for the question, add a field on each record for
Incomplete Answer (incomplete_answer),
Safety Violations (safety_violations),
Unrealistic Tools (unrealistic_tools),
Overcomplicated Solution (overcomplicated_solution),
Missing Context (missing_context),
Poor Quality Tips (poor_quality_tips). with 1 or 0 if you think that the failure applies or not respectively.

questions are a json array: {QUESTIONS} 

and each question is a json object where 'question' is the question and everything else is related to the answer

IMPORTANT: Return ONLY valid JSON format using double quotes (") for all strings, not single quotes ('). 
"""

# evaluate questions

# save first pass labeled questions to a json file if not already there
if not os.path.exists('data/first-pass.json'):
    first_pass_labeled_questions = evaluate_questions(questions, first_pass_prompt)
    with open('data/first-pass.json', 'w') as f:
        json.dump(first_pass_labeled_questions, f, indent=4)

# visualize evals
#visualize_evals(to_dataframe(first_pass_labeled_questions))

# load first pass labeled questions
second_pass_prompt = """
As you can see your evaluation is way more permissive than mine, keep in mind this is for an ordinary person at home
which won't have overcomplicated tools like a multi-meter, or ohmmeter, or other tools that are not common in a home.

Please evaluate the questions using these instructions:

You are reviewing the quality of the support questions.
Given the answer and instructions generated for the question, add a field on each record for
Incomplete Answer (incomplete_answer),
Safety Violations (safety_violations),
Unrealistic Tools (unrealistic_tools),
Overcomplicated Solution (overcomplicated_solution),
Missing Context (missing_context),
Poor Quality Tips (poor_quality_tips). with 1 or 0 if you think that the failure applies or not respectively.

be very strict on the unrealistic tools and overcomplicated solution, and incomplete answers. this is for inexperienced people at home.
And answer is incomplete if an inexperienced user can have follow up questions about the answer no matter how basic they are. Also make sure you analize the question with not that strictly to make sure are complete answers for inexperienced users.

Also over complicated solution is when the answer is too complex for an inexperienced user to follow despite how minimum or basic it appears.

questions are a json array: {QUESTIONS} 

and each question is a json object where 'question' is the question and everything else is related to the answer

IMPORTANT: Return ONLY valid JSON format using double quotes (") for all strings, not single quotes ('). 
"""

# evaluate questions
# save second pass labeled questions to a json file
if not os.path.exists('data/second-pass.json'):
    second_pass_labeled_questions = evaluate_questions(questions, second_pass_prompt)
    with open('data/second-pass.json', 'w') as f:
        json.dump(second_pass_labeled_questions, f, indent=4)

# visualize evals
#visualize_evals(to_dataframe(second_pass_labeled_questions))

# compare human evals with second pass labeled questions
human_evals = json.load(open('data/human-evals.json'))
first_pass_labeled_questions = json.load(open('data/first-pass.json'))
second_pass_labeled_questions = json.load(open('data/second-pass.json'))

def evaluate_similarity(human_evals, labeled_questions):
    """
    Evaluates the similarity between human evals and labeled questions
    """
    incomplete_answer_similarity = 0
    safety_violations_similarity = 0
    unrealistic_tools_similarity = 0
    overcomplicated_solution_similarity = 0
    missing_context_similarity = 0
    poor_quality_tips_similarity = 0
    for q in range(0, len(human_evals)-1):
        if int(human_evals[q]['incomplete_answer']) == labeled_questions[q]['incomplete_answer']:
            incomplete_answer_similarity += 1
        if int(human_evals[q]['safety_violations']) == labeled_questions[q]['safety_violations']:
            safety_violations_similarity += 1
        if int(human_evals[q]['unrealistic_tools']) == labeled_questions[q]['unrealistic_tools']:
            unrealistic_tools_similarity += 1
        if int(human_evals[q]['overcomplicated_solution']) == labeled_questions[q]['overcomplicated_solution']:
            overcomplicated_solution_similarity += 1
        if int(human_evals[q]['missing_context']) == labeled_questions[q]['missing_context']:
            missing_context_similarity += 1
        if int(human_evals[q]['poor_quality_tips']) == labeled_questions[q]['poor_quality_tips']:
            poor_quality_tips_similarity += 1

    print(f"Incomplete answer similarity: {incomplete_answer_similarity / len(human_evals)}")
    print(f"Safety violations similarity: {safety_violations_similarity / len(human_evals)}")
    print(f"Unrealistic tools similarity: {unrealistic_tools_similarity / len(human_evals)}")
    print(f"Overcomplicated solution similarity: {overcomplicated_solution_similarity / len(human_evals)}")
    print(f"Missing context similarity: {missing_context_similarity / len(human_evals)}")
    print(f"Poor quality tips similarity: {poor_quality_tips_similarity / len(human_evals)}")

    return (incomplete_answer_similarity + safety_violations_similarity + unrealistic_tools_similarity + overcomplicated_solution_similarity + missing_context_similarity + poor_quality_tips_similarity) / len(human_evals)

# use human evals as ground truth and compare the similarities for each question and each field and comput a percentage of similarity in total based on hit or miss
# compute the percentage of similarity in total based on hit or miss
first_pass_similarity = evaluate_similarity(human_evals, first_pass_labeled_questions)
print("--------------------------------")
second_pass_similarity = evaluate_similarity(human_evals, second_pass_labeled_questions)
print("--------------------------------")
print(f"First pass similarity: {first_pass_similarity / 6}")
print(f"Second pass similarity: {second_pass_similarity / 6}")
    
