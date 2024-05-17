""" Question Form Generator"""
import json
from typing import Dict, Tuple

from src.config import PATH_QUESTION_TEMPLATES 


def get_question_form(
    scenario: Dict, character: Dict, question_type: str, question_ordering: int, system_instruction: bool
) -> Tuple[Dict, Dict]:
    """Get question form for a given scenario, question_type and question_ordering"""

    # (0) Set option ordering
    if question_ordering == 0 : 
        decision_mapping = {"A": "decision1", "B": "decision2", "C":"decision3"}
    elif question_ordering == 1 :
        decision_mapping = {"A": "decision1", "B": "decision3", "C":"decision2"}
    elif question_ordering == 2 :
        decision_mapping = {"A": "decision2", "B": "decision1", "C":"decision3"}
    elif question_ordering == 3 :
        decision_mapping = {"A": "decision2", "B": "decision3", "C":"decision1"}
    elif question_ordering == 4 :
        decision_mapping = {"A": "decision3", "B": "decision1", "C":"decision2"}
    elif question_ordering == 5 :
        decision_mapping = {"A": "decision3", "B": "decision2", "C":"decision1"}

    # (2) Generate question form
    with open(f"{PATH_QUESTION_TEMPLATES}/{question_type}.json", encoding="utf-8") as f:
        question_config = json.load(f)

    question_form = {
        "question": question_config["question"].format(
            scenario["context"],
            scenario[decision_mapping["A"]],
            scenario[decision_mapping["B"]],
            scenario[decision_mapping["C"]],
        ),
        "question_header": question_config["question_header"].format( 
            character["name"],
            character["main value category"],
            character["sub values"],
            character["name"]
        )
        if system_instruction
        else "",
    }

    return (question_form, decision_mapping)
