import requests
from llmproxy import LLMProxy
from prompts import queries
import random

client = LLMProxy()
models = ['4o-mini', 'us.anthropic.claude-3-haiku-20240307-v1:0', 'azure-phi3', 'us.meta.llama3-2-3b-instruct-v1:0']

system_instructions = (
            "You are an expert nutritionist that specializes in developing regions.")
    
temperature_value = 0.0
last_queries = 2
rag_enabled = True
    
with open("model_eval_results.txt", "w") as file:
    for question in queries:
        file.write("QUESTION: " + question + "\n\n")
        print(question)

        for model_name in models:
            print(model_name)
            session_id_value = "convo" + str(random.random())
            file.write("MODEL: " + model_name + "\n")

            response = client.generate(
                model = model_name,
                system = system_instructions,
                query = question,
                temperature = temperature_value,
                lastk = last_queries,
                session_id = session_id_value,
                rag_usage = False)
            
            file.write(response['result'] + "\n") 
