# NER
SCHEMA_NER = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["span", "label"],
        "properties": {
            "span": {"type": "string", "minLength": 1},
            "label": {"type": "string", "minLength": 1},
        },
    },
}

# HARD NER
SCHEMA_HARD_NER = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "entities"],
    "properties": {
        "prompt": {"type": "string", "minLength": 1},
        "entities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

# RELATION EXTRACTION
SCHEMA_REL = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["subject", "relation", "object"],
        "properties": {
            "subject": {"type": "string", "minLength": 1},
            "relation": {"type": "string", "minLength": 1},
            "object": {"type": "string", "minLength": 1},
        },
    },
}

# EXTRACTIVE QA
SCHEMA_EX_QA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "answers"],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "answers": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
    },
}

# OPEN TASK
SCHEMA_OPEN_TASK = {
    "type": "object",
    "additionalProperties": False,
    "required": ["task_prompt", "output"],
    "properties": {
        "task_prompt": {"type": "string", "minLength": 1},
        "output": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
    },
}