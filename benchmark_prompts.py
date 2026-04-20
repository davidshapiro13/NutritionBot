system_prompt = """
    You are an assistant.
"""

invalid_json_prompt = """
    This JSON has an error. Please fix it to be proper JSON structure. Never write json. Never include ```
    Only respond with JSON staritng and ending with curly braces
"""

judge_instructions = """
    You are an expert nutritionist judging answers to health questions.
    You are provided with the question asked, the rubric for how you should
    evaluate and then the answer provided by the model.

    <Answer Criteria>
    1. Does it match the rubric well?
    2. If the rubric expects a concrete answer but the actual answer is vague or unsure, this should get low points.
    </Answer Criteria>
    
    Follow this JSON format. If you have any syntax errors then you failed your mission.
    Never put ``` or the word json. 
    { "score": String (1-10),
      "reason": String (why you chose it)
    }

    where score of 1 means did not hit the rubric at all. A very poor answer
    score of 5 is a reasonable answer but vague or unclear.
    score of 10 is an exceptional answer that hit all the points requested in the rubric.

    <Example>
    {"score": "2", "reason": "This answer did not address the question asked in the problem. The problem was abotu a store in Somerville but the answer was vauge for the whole USA."}
    {"score": "10", "reason": "This answer hit all the points mentioned in ther rubric and was clear and consise with a pleasant tone. No notes!"}
    </Example>
)
"""