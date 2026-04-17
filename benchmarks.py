import csv
from llmproxy import LLMProxy
from benchmark_prompts import system_prompt, invalid_json_prompt, judge_instructions
import ast
OUR_MODEL = 'gpt-5-mini'
judge_models = ['us.anthropic.claude-3-haiku-20240307-v1:0', 'google.gemma-3-27b-it', 'us.meta.llama3-2-3b-instruct-v1:0']

client = LLMProxy()

class Base_Model():
    def __init__(self):
        self.name = OUR_MODEL
        rag_enabled = True
        self.session_id = "BaseModel"

    def onboard(self):
        pass

    def answer(self, questions):
        for question in questions:
            response = client.generate(
                model = self.name,
                system = system_prompt,
                query = question,
                session_id = self.session_id,
                rag_usage = False)["result"]
        return response

class Benchmark():
    def _parse_judge_response(self, response, model):
        parsed = response

        if isinstance(parsed, str):
            not_proper_json = True
            while not_proper_json:
                try:
                    parsed = ast.literal_eval(parsed)
                    not_proper_json = False
                except Exception:
                    print("FAIL: ", parsed)
                    parsed = client.generate(
                        model=model,
                        system=invalid_json_prompt,
                        query=parsed,
                        session_id="fix json",
                        rag_usage=False,
                    )["result"]

        if not isinstance(parsed, dict):
            parsed = {"score": 1, "reason": f"Judge returned invalid payload: {parsed}"}

        if "score" not in parsed:
            for key in parsed:
                if key.lower() == "score":
                    parsed["score"] = parsed[key]
                    break

        if "reason" not in parsed:
            for key in parsed:
                if key.lower() == "reason":
                    parsed["reason"] = parsed[key]
                    break

        parsed["score"] = self._coerce_score(parsed.get("score"))
        parsed["reason"] = str(parsed.get("reason", "")).strip()
        return parsed

    def _coerce_score(self, raw_score):
        if isinstance(raw_score, (int, float)):
            numeric_score = int(round(raw_score))
        else:
            text = str(raw_score).strip()
            digits = "".join(ch for ch in text if ch.isdigit())
            numeric_score = int(digits) if digits else 1

        return max(1, min(10, numeric_score))

    def load_from_csv(self):
        with open("benchmark_exam.csv", mode="r", errors="replace") as file:
            csvFile = csv.reader(file)
            exam_problems = []
            for line in csvFile:
                topic = line[0]

                for i, existing_topic in enumerate([problem["topic"] for problem in exam_problems]):
                    if topic == existing_topic:
                        exam_problem = exam_problems[i]
                        break
                else:
                    exam_problem = {"topic": topic, "questions": []}
                    exam_problems.append(exam_problem)

                exam_problem["questions"].append(line[1])
                exam_problem["rubric"] = line[2]
                if len(line) > 3:
                    exam_problem["tag"] = line[3]
                else:
                    exam_problem["tag"] == "N/A"
        
            #Delete the title row
            del exam_problems[0]
            return exam_problems

    def LLM_as_Jury(self, questions, rubric, answer):
        decisions = []
        for model in judge_models:
            print(model)
            response = client.generate(
                model = model,
                system = judge_instructions + " | Question: " + questions[-1] + " | Rubric: " + rubric,
                query = answer,
                session_id = "Judgement",
                rag_usage = False)["result"]
            
            decisions.append(self._parse_judge_response(response, model))
        return decisions

    def aggregate(self, decisions):
        score = sum(self._coerce_score(decision.get("score")) for decision in decisions)
        avg_score = round(score / len(decisions))
        return avg_score

    def exam_score(self, exam_results):
        summed_score = sum(float(exam_result["score"]) for exam_result in exam_results)
        return (summed_score / len(exam_results) * 10) / 100

    def write_results(self, exam_results, overall_score, file_name):
        with open(file_name, "w") as file:
            for result in exam_results:
                file.write(result["topic"] + ": " + result["question"] + "\n")
                file.write("Answer: " + result["answer"] + "\n")
                file.write("Jury Score: " + str(int(result["score"])) + "\n\n")
                for jurer in result["reasoning"]:
                    reason = str(jurer.get("reason", ""))
                    score = str(jurer.get("score", ""))
                    file.write(reason + " | Score: " + score + "\n\n")

            file.write("Overall score: " + str(overall_score) + "%")

    def evaluate(self, model=Base_Model(), file_name="benchmark_results.txt"):
        exam_results = []
        exam_problems = self.load_from_csv()
        for problem in exam_problems:
            print(problem["topic"])
            if problem["tag"] == "Onboard":
                print("ONBOARDING!!!!")
                memory = model.onboard()
                answer = model.answer(problem["questions"], memory=memory)
            else:
                answer = model.answer(problem["questions"])
            decisions = self.LLM_as_Jury(problem["questions"], problem["rubric"], answer)
            print(decisions)
            result = self.aggregate(decisions)
            exam_results.append({"question": problem["questions"][-1], "topic": problem["topic"], "answer": answer, "score": result, "reasoning": decisions})
        overall_score = self.exam_score(exam_results)
        self.write_results(exam_results, overall_score, file_name)
