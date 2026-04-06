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
        self.session_id = "OurModel" + str(random.random())

    def answer(self, questions):
        for question in questions:
            response = self.agent.run(question, self.session_id)[0]
        return response
    
    def onboard(self):
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
        

agent = Our_Model()
benchmark.evaluate(agent, "4-04 Benchmark results.txt")