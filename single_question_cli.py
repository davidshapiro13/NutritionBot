import argparse
import random

from agent import NutritionAgent
from user_memory import UserMemory


def _bootstrap_agent(agent: NutritionAgent, user_id: str) -> None:
    text, buttons = agent.run("hi", user_id)
    if not buttons:
        return
    for button in buttons:
        if getattr(button, "id", None) == "disclaimer_agree":
            agent.run_tool("disclaimer_agree", user_id)
            return


def _seed_demo_profile(user_id: str) -> str:
    profile = {
        "name": "Fred",
        "age_group": "adult",
        "gender": "male",
        "health_conditions": "diabetes",
        "allergies": "peanuts",
        "asking_for": "self",
    }
    mem = UserMemory(embed_model=None)
    mem.save_profile(user_id, profile)
    return mem.load_all(user_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one NutritionAgent question from terminal (Evaluator-style setup)."
    )
    parser.add_argument("question", help="Question to send to NutritionAgent.run(...)")
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional user_id. Defaults to a randomized Evaluator-style session id.",
    )
    parser.add_argument(
        "--memory",
        default=None,
        help="Optional memory text appended as '[MEMORY]<text>' (matches Evaluator behavior).",
    )
    parser.add_argument(
        "--seed-profile",
        action="store_true",
        help="Seed the same demo profile used in Evaluator.onboard().",
    )
    args = parser.parse_args()

    user_id = args.user_id or f"OurModel{random.random()}"
    agent = NutritionAgent()
    _bootstrap_agent(agent, user_id)

    if args.seed_profile:
        profile_text = _seed_demo_profile(user_id)
        print(f"[seeded_profile]\n{profile_text}")

    prompt = args.question
    if args.memory is not None:
        prompt = f"{prompt}[MEMORY]{args.memory}"

    text, buttons = agent.run(prompt, user_id)
    print(f"[user_id] {user_id}")
    print("[response]")
    print(text)
    print("[buttons]")
    if buttons == "request_location":
        print("request_location")
    elif not buttons:
        print("(none)")
    else:
        for button in buttons:
            print(f"- {getattr(button, 'id', '')}: {getattr(button, 'title', '')}")


if __name__ == "__main__":
    main()
