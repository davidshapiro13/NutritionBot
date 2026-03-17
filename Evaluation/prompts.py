model_eval_open_response1 = """
    I live in Somerville, MA. What is a cheap and healthy meal I can make for dinner? Why is it healthy?
"""

model_eval_open_response2 = """
    I am going to eat rice, beans and beer for dinner. What else should I consider eating? I have a gluten allergy.
"""

model_eval_open_response3 = """
    My stomach hurts a lot right now. What should I do to treat it?
"""

model_eval_open_response4 = """
    Yesterday I ate toast and jam for breakfast, cobb salad for lunch and sushi for dinner. What types of foods should I focus on today?
"""

model_eval_open_response5 = """
    I want to make a well-rounded meal for my family with $20. What is a broken down budget for it?
"""

model_eval_multi_choice1 = {"question": """
    Today is Wednesday the 10th. I have some rice in my fridge from last Thursday. What should I do with it?
    a) Eat it so you do not waste it.
    b) Throw it away. It has been too long and can grow bacteria.
    c) Smell it. If it smells fine, it is fine.
    d) Mix it into another meal
    e) None of the above
""", "answer": "b"}

model_eval_multi_choice2 = {"question": """
    What are common symptoms of food poisoning?
    a) Nausea
    b) Loss of Taste and Smell
    c) Stomach Cramps
    d) Sore throat
    e) a and c
    f) c and d
""", "answer": "e"}

model_eval_multi_choice3 = {"question": """
    Which is not a food group?

    a) Fruits
    b) Grains
    c) Starches
    d) Dairy
    e) Vegetables
""", "answer": "c"}

model_eval_multi_choice4 = {"question": """
    Which of the following is not safe to feed a nine-month-old?
    a) chicken
    b) honey
    c) banana
    d) rice
    e) all of these are safe
    f) none of these are safe
""", "answer": "b"}

model_eval_multi_choice5 = {"question": """
    I have an allergy to gelatin. What foods should I check the ingredients of?

    a) Tomato Sauce
    b) Chocolate Mousse
    c) Tofu
    d) Marshmellows
    e) a and b
    f) b and d
""", "answer": "f"}




open_response = [model_eval_open_response1, model_eval_open_response2, model_eval_open_response3, model_eval_open_response4, model_eval_open_response5]
multi_choice = [model_eval_multi_choice1, model_eval_multi_choice2, model_eval_multi_choice3, model_eval_multi_choice4, model_eval_multi_choice5]