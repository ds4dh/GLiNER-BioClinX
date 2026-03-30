from typing import List, Dict
from .task_prompts import (
    # User inputs (few-shot examples)
    UI_1,
    UI_2,

    # NER
    NER_SYS_PROMPT,
    NER_AO_1,
    NER_AO_2,

    # HARD NER
    HARD_NER_SYS_PROMPT,
    HARD_NER_AO_1,
    HARD_NER_AO_2,

    # RELATION EXTRACTION
    REL_SYS_PROMPT,
    REL_AO_1,
    REL_AO_2,

    # EXTRACTIVE QA
    EX_QA_SYS_PROMPT,
    EX_QA_AO_1,
    EX_QA_AO_2,

    # OPEN TASK
    OPEN_TASK_SYS_PROMPT,
    OPEN_TASK_AO_1,
    OPEN_TASK_AO_2,
)

Message = Dict[str, str]

def _wrap_user_text(text: str) -> str:
    return f"""Text:\n\"\"\"\n{text.strip()}\n\"\"\""""

# NER
def create_ner_messages(text: str) -> List[Message]:
    return [
        {"role": "system", "content": NER_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": NER_AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": NER_AO_2},
        {"role": "user", "content": _wrap_user_text(text)},
    ]


# HARD NER
def create_hard_ner_messages(text: str) -> List[Message]:
    return [
        {"role": "system", "content": HARD_NER_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": HARD_NER_AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": HARD_NER_AO_2},
        {"role": "user", "content": _wrap_user_text(text)},
    ]


# RELATION EXTRACTION
def create_rel_messages(text: str) -> List[Message]:
    return [
        {"role": "system", "content": REL_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": REL_AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": REL_AO_2},
        {"role": "user", "content": _wrap_user_text(text)},
    ]


# EXTRACTIVE QA
def create_extractive_qa_messages(text: str) -> List[Message]:
    return [
        {"role": "system", "content": EX_QA_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": EX_QA_AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": EX_QA_AO_2},
        {"role": "user", "content": _wrap_user_text(text)},
    ]


# OPEN TASK
def create_open_task_messages(text: str) -> List[Message]:
    return [
        {"role": "system", "content": OPEN_TASK_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": OPEN_TASK_AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": OPEN_TASK_AO_2},
        {"role": "user", "content": _wrap_user_text(text)},
    ]