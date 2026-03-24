from AI import AI
from prompts import main_system_prompt, button_creator_prompt
import random
import ast
model = AI()

class Button():
    def __init__(self, id, title):
        self.id = id
        self.title = title

def convertButtons(buttonsJSON):
    buttonsJSON = ast.literal_eval(buttonsJSON)
    buttons = []
    for buttonJSON in buttonsJSON:
        button_info = ast.literal_eval(buttonJSON)
        print(button_info)
        new_button = Button(id=button_info["id"], title=button_info["title"])
        buttons.append(new_button)
        print(buttons)
    return buttons

print("Hi, I'm your Nutrition Bot")
query_prompt = input("You: ")

#Temporary - figure out what to do with this
session_id = "Session" + str(random.random())

while "EXIT" not in query_prompt:
    response = model.ask(main_system_prompt, query_prompt, session_id)
    buttonsJSON = model.ask(button_creator_prompt, response, session_id)
    buttons = convertButtons(buttonsJSON)
    print("Nutribot: ", response)
    print("BUTTONS", buttons)
    query_prompt = input("You: ")

