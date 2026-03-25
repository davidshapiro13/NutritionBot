from benchmarks import Benchmark
from agent import NutritionAgent

benchmark = Benchmark()
benchmark.evaluate()

class Our_Model():
    def __init__(self):
        self.name = 'gpt-5-mini'
        self.agent = NutritionAgent()
        self.session_id = "OurModel"

    def answer(self, questions):
        for question in questions:
            response = self.agent.run(question, self.session_id)[0]
        return response

agent = Our_Model()
benchmark.evaluate(agent, "3-24 Benchmark results.txt")