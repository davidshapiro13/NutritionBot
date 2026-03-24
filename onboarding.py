"""Onboarding helper for NutritionBot.

Collects structured user profile fields and writes them to user_memory files.

Fields:
- name
- age_group (child / adult / elder)
- gender
- health_conditions
- allergies
- medications
- asking_for (self / child / parent / spouse / other)
- main_goal
"""

from pathlib import Path
from user_memory import UserMemory


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    sel = None
    while sel is None:
        options = ", ".join(choices)
        raw = input(f"{prompt} [{options}]" + (f" (default: {default})" if default else "") + ": ").strip()
        if raw == "" and default:
            return default
        if raw == "" and not default:
            print("Please enter a value.")
            continue
        for c in choices:
            if raw.lower() == c.lower():
                return c
        # Accept free text for open-ended fields when explicitly asked
        if prompt.startswith("Gender") or prompt.startswith("Asking"):
            return raw
        print("Invalid choice. Please choose from:", options)


def _prompt_free_text(prompt: str, required: bool = False) -> str:
    while True:
        value = input(f"{prompt}: ").strip()
        if value or not required:
            return value
        print("This field is required.")


def run_onboarding(user_id: str | None = None) -> dict[str, str]:
    """Run interactive onboarding and save data to user memory."""
    if not user_id:
        user_id = input("Enter a user ID (e.g. phone number): ").strip()
    if not user_id:
        raise ValueError("user_id is required")

    profile = {
        "name": _prompt_free_text("Name", required=True),
        "age_group": _prompt_choice("Age Group", ["child", "adult", "elder"], default="adult"),
        "gender": _prompt_free_text("Gender", required=False),
        "health_conditions": _prompt_free_text("Health conditions (heart disease, diabetes, pregnancy, etc)", required=False),
        "allergies": _prompt_free_text("Allergies", required=False),
        "medications": _prompt_free_text("Medications", required=False),
        "asking_for": _prompt_choice("Who are you asking for", ["self", "child", "parent", "spouse", "other"], default="self"),
        "main_goal": _prompt_free_text("Main Goal", required=True),
    }

    mem = UserMemory(embed_model=None)
    mem.save_profile(user_id, profile)

    print("\nOnboarding data saved for user_id=", user_id)
    print("Saved profile:")
    for k, v in profile.items():
        print(f"- {k}: {v}")

    return profile


if __name__ == "__main__":
    run_onboarding()
