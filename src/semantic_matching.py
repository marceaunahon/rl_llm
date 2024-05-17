"""Semantic Matching: From Tokens to Decisions """
import pandas as pd
from src.utils import stem_sentences


def token_to_decision_matching(
    answer, scenario, responses_pattern, question_type, decision_mapping, refusals
):
    """Semantic Mapping: From Sequences of Tokens to decisions"""

    responses_pattern_q = responses_pattern[question_type]

    # ---------------------
    # Set possible answers
    # ---------------------
    decision_mapping_inv = {v: k for k, v in decision_mapping.items()}

    optionA = scenario[decision_mapping["A"]]
    optionB = scenario[decision_mapping["B"]]
    optionC = scenario[decision_mapping["C"]]

    answers_decision1 = [
        t.format(
            optionA=optionA,
            optionA_short=optionA[:-1],
            optionB=optionB,
            optionB_short=optionB[:-1],
            optionC=optionC,
            optionC_short=optionC[:-1],
        )
        .lower()
        .strip()
        for t in responses_pattern_q[f"responses_{decision_mapping_inv['decision1']}"]
    ]
    answers_decision2 = [
        t.format(
            optionA=optionA,
            optionA_short=optionA[:-1],
            optionB=optionB,
            optionB_short=optionB[:-1],
            optionC=optionC,
            optionC_short=optionC[:-1],
        )
        .lower()
        .strip()
        for t in responses_pattern_q[f"responses_{decision_mapping_inv['decision2']}"]
    ]
    answers_decision3 = [
        t.format(
            optionA=optionA,
            optionA_short=optionA[:-1],
            optionB=optionB,
            optionB_short=optionB[:-1],
            optionC=optionC,
            optionC_short=optionC[:-1],
        )
        .lower()
        .strip()
        for t in responses_pattern_q[f"responses_{decision_mapping_inv['decision3']}"]
    ]

    refusals = [refusal.lower().strip() for refusal in refusals]

    # --------------------------------------------
    # Perform Matching using Matching Heuristic
    # --------------------------------------------

    answer = answer.lower().strip()
    answer = answer.replace("\"", "")

    # Catch common answer deviations
    if pd.isnull(answer):
        answer = ""
    if answer.startswith("answer"):
        answer = answer[6:]
    if answer.startswith(":"):
        answer = answer[1:]

    # (1) Check for "Exact" decision 1 / decision 2 Matches
    if answer in answers_decision1:
        return "decision1"
    if answer in answers_decision2:
        return "decision2"
    if answer in answers_decision3:
        return "decision3"

    # (2) Check for stemming matches
    answer_stemmed = stem_sentences([answer])[0]
    answers_decision1_stemmed = stem_sentences(answers_decision1)
    answers_decision2_stemmed = stem_sentences(answers_decision2)
    answers_decision3_stemmed = stem_sentences(answers_decision3)

    if answer_stemmed in answers_decision1_stemmed:
        return "decision1"
    if answer_stemmed in answers_decision2_stemmed:
        return "decision2"
    if answer_stemmed in answers_decision3_stemmed:
        return "decision3"

    # (3) Check for question_type specific
    if question_type == "compare":
        if answer.startswith("yes"):
            return decision_mapping["A"]
        if answer.startswith("no"):
            return decision_mapping["B"]

    if question_type == "repeat":
        if not answer.startswith("I"):
            answer_stemmed = "i " + answer_stemmed

            if answer_stemmed in answers_decision1_stemmed:
                return "decision1"
            if answer_stemmed in answers_decision2_stemmed:
                return "decision2"
            if answer_stemmed in answers_decision3_stemmed:
                return "decision3"

    # (4) Check for refusals
    for refusal_string in refusals:
        if refusal_string in answer.lower():
            return "refusal"

    return "invalid"
