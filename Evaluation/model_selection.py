import requests
from llmproxy import LLMProxy
from prompts import open_response, multi_choice
import random

class Model():
    def __init__(self, name):
        self.name = name
        self.multi_correct = 0
        self.best_open = 0
        self.answers = []

    def increase_correct_multi(self):
        self.multi_correct += 1
    def increase_best_open(self):
        self.best_open += 1
    def generate_answers(self):
        for question in open_response:
            session_id_value = "convo" + str(random.random())
            response = client.generate(
                        model = self.name,
                        system = open_response_instructions,
                        query = question,
                        temperature = temperature_value,
                        lastk = last_queries,
                        session_id = session_id_value,
                        rag_usage = False)["result"]
            
            self.answers.append(response)

    def __repr__(self):  
        return "Model: " + self.name + "| Multiple Choice: " + str(self.multi_correct) + "\n"

client = LLMProxy()
models = [Model('4o-mini'), Model('us.anthropic.claude-3-haiku-20240307-v1:0'), Model('azure-phi3'), Model('us.meta.llama3-2-3b-instruct-v1:0')]
judge_models = [Model('4o-mini'), Model('us.anthropic.claude-3-haiku-20240307-v1:0'), Model('azure-phi3'), Model('us.meta.llama3-2-3b-instruct-v1:0')]

multiple_choice_instructions = (
            "You are an expert nutritionist located in Boston, MA. This is a multiple choice question. Only answer with a, b, c, d, e, f (depending on number of options in question)")

open_response_instructions = (
            "You are an expert nutritionist located in Boston, MA. This is an open response question. Write a friendly, informational, accurate answer to the following question."
)
judge_instructions = (
    """You are an expert nutritionist judging answers to helath questions.
    Answer that you like 1 or 2 better where 1 is the first option and 2 is the second option.
    Base your answer on the accuracy of the response, the clarity, the completeness and the tone (we want a kind, friendly tone)"""
)
    
temperature_value = 0.0
last_queries = 2
rag_enabled = True

def multiple_choice_section():
    for question in multi_choice:
        for model in models:
            session_id_value = "convo" + str(random.random())
            response = client.generate(
                        model = model.name,
                        system = multiple_choice_instructions,
                        query = question["question"],
                        temperature = temperature_value,
                        lastk = last_queries,
                        session_id = session_id_value,
                        rag_usage = False)
            if question["answer"] == response["result"]:
                model.increase_correct_multi()
    print(models)

def write_to_file(models):
    with open("model_eval_results.txt", "w") as file:
        for i, question in enumerate(open_response):
            file.write("QUESTION: " + question + "\n\n")
            for model in models:
                file.write("MODEL: " + model.name + "\n")
                file.write(model.answers[i] + "\n")

def LLM_as_Jury(models):
    for i, question in enumerate(open_response):
        for judge in judge_models:
            response1 = models[0].answers[i]
            response2 = models[1].answers[i]
            response3 = models[2].answers[i]
            response4 = models[3].answers[i]
            round1_winner, round1_res = judge_vote(judge.name, models[0], models[1], response1, response2)
            round2_winner, round2_res = judge_vote(judge.name, models[0], models[1], response3, response4)
            round3_winner, _ = judge_vote(judge.name, round1_winner, round2_winner, round1_res, round2_res)
            round3_winner.increase_best_open()
    best_model = None
    top_score = 0
    for model in models:
        if model.best_open > top_score:
            top_score = model.best_open
            best_model = model
    return best_model, top_score

            

def judge_vote(judge_model, model1, model2, ans1, ans2):
    session_id_value = "convo" + str(random.random())
    response = client.generate(
        model = judge_model,
        system = judge_instructions,
        query = "Option 1: " + ans1 + "\n\n | Option 2: " + ans2,
        temperature = temperature_value,
        lastk = last_queries,
        session_id = session_id_value,
        rag_usage = False)["result"]
    if response == '1':
        return model1, ans1
    if response == '2':
        return model2, ans2
    return None

                

def open_response_section():
    for model in models:
        model.generate_answers()
    write_to_file(models)
    best_model, top_score = LLM_as_Jury(models)
    print("Best Model: ", best_model, " | Score: ", top_score)

multiple_choice_section()
open_response_section()