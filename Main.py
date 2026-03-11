from AI import AI
from prompts import main_system_prompt
import random
model = AI()

print("Hi, I'm your Nutrition Bot")
query_prompt = input("You: ")

#Temporary - figure out what to do with this
session_id = "Session" + str(random.random())

while "EXIT" not in query_prompt:
    answer = model.ask(main_system_prompt, query_prompt, session_id)
    print("Nutribot: ", answer)
    query_prompt = input("You: ")