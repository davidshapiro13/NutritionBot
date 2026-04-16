from dataclasses import dataclass, field


@dataclass
class AgentState:
    pending_store_type: dict[str, str] = field(default_factory=dict)
    pending_store_search: dict[str, dict] = field(default_factory=dict)
    user_last_location: dict[str, tuple[float, float]] = field(default_factory=dict)
    nutrition_ob_state: dict[str, dict] = field(default_factory=dict)
    nutrition_ob_done_users: set[str] = field(default_factory=set)
    eligibility_state: set[str] = field(default_factory=set)
    wic_eligibility_users: set[str] = field(default_factory=set)
    # Button-driven WIC screening: step name per user; answers collected so far
    wic_eligibility_steps: dict[str, str] = field(default_factory=dict)
    wic_eligibility_answers: dict[str, dict] = field(default_factory=dict)
    food_safety_flow_users: set[str] = field(default_factory=set)
    resources_mode_users: set[str] = field(default_factory=set)
    resources_conversation_summary: dict[str, str] = field(default_factory=dict)
    accepted_disclaimer_users: set[str] = field(default_factory=set)


STATE = AgentState()
