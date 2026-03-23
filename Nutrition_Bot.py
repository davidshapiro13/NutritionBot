from AI import AI
from prompts import main_system_prompt, button_creator_prompt, invalid_json_prompt
import random
import ast
model = AI()

from wa_service_sdk import BaseEvent, TextEvent, InteractiveEvent, create_message, create_buttoned_message, Button

#Temporary
session_id = "Session" + str(random.random())

async def handle_event(event: BaseEvent):
    if isinstance(event, InteractiveEvent):
        user_input = event.interaction_title
    if isinstance(event, TextEvent):
        user_input = event.text
    response = model.ask(main_system_prompt, user_input, session_id)
    buttonsJSON = model.ask(button_creator_prompt, response, session_id)
    buttons = convertButtons(buttonsJSON)
    return create_buttoned_message(
            user_id=event.user_id,
            text=response,
            buttons=buttons,
    )

def convertButtons(buttonsJSON):
    not_proper_json = True
    while not_proper_json:
        try:
            buttons_info = ast.literal_eval(buttonsJSON)
            not_proper_json = False
        except:
            buttonsJSON = model.ask(invalid_json_prompt, str(buttonsJSON), "fix json")
    buttons = []

    for buttonJSON in buttons_info:
        new_button = Button(id=buttonJSON["id"], title=buttonJSON["title"])
        buttons.append(new_button)
    return buttons