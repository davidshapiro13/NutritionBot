from benchmarks import Benchmark
from agent import NutritionAgent
from user_memory import UserMemory
import random

benchmark = Benchmark()
#benchmark.evaluate()

class Our_Model():
    def __init__(self):
        self.name = 'gpt-5-mini'
        self.agent = NutritionAgent()
        self._reset_session()

    def _reset_session(self):
        self.session_id = "OurModel" + str(random.random())
        self._pass_disclaimer_gate()

    def _pass_disclaimer_gate(self):
        text, buttons = self.agent.run("hi", self.session_id)
        if not buttons:
            return
        for button in buttons:
            if getattr(button, "id", None) == "disclaimer_agree":
                self.agent.run_tool("disclaimer_agree", self.session_id)
                return

    def answer(self, questions, memory=None):
        self._reset_session()
        response = ""
        for question in questions:
            if memory == None:
                response = self.agent.run(question, self.session_id)[0]
            else:
                print("MEMORY" + memory)
                response = self.agent.run(question + "[MEMORY]" + memory, self.session_id)[0]
        return response
    
    def onboard(self):
        self._reset_session()
        
        profile = {
        "name": "Fred",
        "age_group": "adult",
        "gender": "male",
        "health_conditions": "diabetes",
        "allergies": "peanuts",
        "asking_for": "self"
        }
        mem = UserMemory(embed_model=None)
        mem.save_profile(self.session_id, profile)
        return mem.load_all(self.session_id)
        

if __name__ == "__main__":
    agent = Our_Model()
    benchmark.evaluate(agent, "4-10 sycophancy results.txt")
