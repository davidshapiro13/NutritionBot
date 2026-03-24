import requests
from llmproxy import LLMProxy
from prompts import open_response, multi_choice
import random
import ast

class Model():
    def __init__(self, name):
        self.name = name
        self.multi_correct = 0
        self.best_open = 0
        self.answers = []
        self.multiple_choice_answers = []

    def increase_correct_multi(self):
        self.multi_correct += 1
    def increase_best_open(self):
        self.best_open += 1
    def reset_best_open(self):
        self.best_open = 0

    def answer_open_response(self):
        for question in open_response:
            session_id_value = "convo" + str(random.random())
            response = client.generate(
                        model = self.name,
                        system = open_response_instructions,
                        query = question["question"],
                        temperature = temperature_value,
                        lastk = last_queries,
                        session_id = session_id_value,
                        rag_usage = False)["result"]
            self.answers.append(response)

    def answer_multiple_choice(self, question):
        session_id_value = "convo" + str(random.random())
        response = client.generate(
            model = self.name,
            system = multiple_choice_instructions,
            query = question["question"],
            temperature = temperature_value,
            lastk = last_queries,
            session_id = session_id_value,
            rag_usage = False)
        self.multiple_choice_answers.append(response['result'])
        
    def __repr__(self):  
        return self.name

client = LLMProxy()
models = [Model('gpt-5-mini'), Model('us.anthropic.claude-3-haiku-20240307-v1:0'), Model('google.gemma-3-27b-it'), Model('us.meta.llama3-2-3b-instruct-v1:0')]
judge_models = [Model('gpt-5-mini'), Model('us.anthropic.claude-3-haiku-20240307-v1:0'), Model('google.gemma-3-27b-it'), Model('us.meta.llama3-2-3b-instruct-v1:0')]

multiple_choice_instructions = (
            """You are an expert nutritionist located in Boston, MA. This is a multiple choice question. Only answer with a, b, c, d, e, f (depending on number of options in question).
                Never write more than 1 letter as the answer. Do not include new lines either.
                
                <Example>
                a
                </Example>""")

open_response_instructions = (
            """You are an expert nutritionist located in Boston, MA. This is an open response question. Write a friendly, informational, accurate answer to the following question.
            """)

judge_instructions = (
    """You are an expert nutritionist judging answers to helath questions.
    Answer that you like 1 or 2 better where 1 is the first option and 2 is the second option and provide your reasoning

    <Answer Criteria>
    1. Is the answer clear and consise?
    2. Is the answer accurate?
    </Answer Criteria>
    
    Do not include new lines. Follow the JSON format. Never put extra curly braces. Otherwise, you have failed your mission. Never write ``` Make sure to use escape characters for quotes or apostrophes 
    { "vote": String (1 or 2),
      "reason": String (why you chose it)
    }

    <Example>
    {"vote": "1", "reason": "The first response gave clear examples and provided a more concise answer"}
    {"vote": "2", "reason": "The second response had more information relevant to the specific situation of the user"}
    </Example>"""
)

invalid_json_prompt = (
    "This JSON has an error. Please fix it to be proper JSON structure: "
)
    
temperature_value = 0.0
last_queries = 2
rag_enabled = True

def multiple_choice_section():
    for question in multi_choice:
        for model in models:
            model.answer_multiple_choice(question)
    write_to_file(models, "multi_choice")
    for model in models:
        for i, answer in enumerate(model.multiple_choice_answers):
            if answer == multi_choice[i]["answer"]:
                model.increase_correct_multi()
    for model in models:
        print(model.name, ": ", model.multi_correct)

def write_to_file(models, type):
    if type == "open_response":
        for i, question in enumerate(open_response):
            with open(f"model_open_response{i}.txt", "w") as file:
                file.write("QUESTION: " + question["question"] + "\n\n")
                for model in models:
                    file.write("MODEL: " + model.name + "\n")
                    file.write(model.answers[i] + "\n")
    else:
        with open("model_multiple_choice.txt", "w") as file:
            for i, question in enumerate(multi_choice):
                file.write("QUESTION: " + question["question"] + "\n\n")
                for model in models:
                    file.write("MODEL: " + model.name + "\n")
                    file.write("Provided: " + model.multiple_choice_answers[i] + " | Correct: " + question["answer"] + "\n\n")

def LLM_as_Jury(models, flip_order=False):
    for i, question in enumerate(open_response):
        for judge in judge_models:
            if not flip_order:
                model_1 = models[0]
                model_2 = models[1]
                model_3 = models[2]
                model_4 = models[3]
                file_name = f"model_open_reasoning{i}.txt"
            else:
                model_1 = models[1]
                model_2 = models[0]
                model_3 = models[3]
                model_4 = models[2]
                file_name = f"model_open_reasoning{i}_flipped.txt"
            response1 = model_1.answers[i]
            response2 = model_2.answers[i]
            response3 = model_3.answers[i]
            response4 = model_4.answers[i]
            round1_winner, round1_res, reason1 = judge_vote(judge.name, question, model_1, model_2, response1, response2)
            round2_winner, round2_res, reason2 = judge_vote(judge.name, question, model_3, model_4, response3, response4)
            round3_winner, _, reason3 = judge_vote(judge.name, question, round1_winner, round2_winner, round1_res, round2_res)
            round3_winner.increase_best_open()

            with open(file_name, "a") as file:
                file.write(judge.name + " Judgement:\n\n")
                file.write(model_1.name + " vs. " + model_2.name + "\n")
                file.write(reason1 + "\n\n")
                file.write(model_3.name + " vs. " + model_4.name + "\n")
                file.write(reason2 + "\n\n")
                file.write(round1_winner.name + " vs. " + round2_winner.name + "\n")
                file.write(reason3 + "\n\n")
    best_model = None
    top_score = 0
    for model in models:
        if model.best_open > top_score:
            top_score = model.best_open
            best_model = model
    return best_model, top_score

            

def judge_vote(judge_model, question, model1, model2, ans1, ans2):
    session_id_value = "convo" + str(random.random())
    response = client.generate(
        model = judge_model,
        system = judge_instructions + "Question: " + question["question"] + " | Rubric: " + question["rubric"],
        query = "Option 1: " + ans1 + "\n\n | Option 2: " + ans2,
        temperature = temperature_value,
        lastk = last_queries,
        session_id = session_id_value,
        rag_usage = False)["result"]
    not_proper_json = True
    while not_proper_json:
        try:
            response = ast.literal_eval(response)
            not_proper_json = False
        except:
            response = client.generate(
                model = judge_model,
                system = invalid_json_prompt,
                query = response,
                temperature = temperature_value,
                lastk = last_queries,
                session_id = "fix json",
                rag_usage = False)["result"]
    
    if response["vote"] == '1':
        return model1, ans1, response["reason"]
    if response["vote"] == '2':
        return model2, ans2, response["reason"]
    print(response)
    return None
                

def open_response_section():
    for model in models:
        model.answer_open_response()
    write_to_file(models, "open_response")
    best_model, top_score = LLM_as_Jury(models)
    print("First Round")
    print("Best Model: ", best_model, " | Open Score: ", top_score)
    for model in models:
        print("Model: ", model.name, " | Open Score: ", model.best_open)
        model.reset_best_open()

    print("Flipped Round - to eliminate bias")
    best_model, top_score = LLM_as_Jury(models, flip_order=True)
    print("Best Model: ", best_model, " | Open Score: ", top_score)
    for model in models:
        print("Model: ", model.name, " | Open Score: ", model.best_open)

multiple_choice_section()
open_response_section()