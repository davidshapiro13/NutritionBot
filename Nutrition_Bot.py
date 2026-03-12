from AI import AI
from prompts import main_system_prompt
import random
model = AI()

from wa_service_sdk import BaseEvent, TextEvent, create_message

#Temporary
session_id = "Session" + str(random.random())

async def handle_event(event: BaseEvent):
    if isinstance(event, TextEvent):
        response = model.ask(main_system_prompt, event.text, session_id)
        return create_message(user_id=event.user_id, text=response)
    
"""
print("Hi, I'm your Nutrition Bot")
query_prompt = input("You: ")

#Temporary - figure out what to do with this
session_id = "Session" + str(random.random())

while "EXIT" not in query_prompt:
    answer = model.ask(main_system_prompt, query_prompt, session_id)
    print("Nutribot: ", answer)
    query_prompt = input("You: ")
"""