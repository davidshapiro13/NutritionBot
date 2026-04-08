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

    def answer(self, questions, memory=None):
        response = ""
        for question in questions:
            if memory == None:
                response = self.agent.run(question, self.session_id)[0]
            else:
                print("MEMORY" + memory)
                response = self.agent.run(question + "[MEMORY]" + memory, self.session_id)[0]
        return response
    
    def onboard(self):
        self.session_id = "OurModel" + str(random.random())
        
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
        

agent = Our_Model()
benchmark.evaluate(agent, "4-07 Benchmark results.txt")