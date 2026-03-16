from AI import AI
from prompts import main_system_prompt, button_creator_prompt
import random
import ast
model = AI()

from wa_service_sdk import BaseEvent, TextEvent, ReplyEvent, create_message, create_buttoned_message, Button

#Temporary
session_id = "Session" + str(random.random())

async def handle_event(event: BaseEvent):
    if isinstance(event, ReplyEvent):
        return create_message(
            user_id=event.user_id,
            text=f'You chose the option: ' + event.text,
        )
    if isinstance(event, TextEvent):
        response = model.ask(main_system_prompt, event.text, session_id)
        buttonsJSON = model.ask(button_creator_prompt, response, session_id)
        buttons = convertButtons(buttonsJSON)
        return create_buttoned_message(
            user_id=event.user_id,
            text=response,
            buttons=buttons,
        )

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