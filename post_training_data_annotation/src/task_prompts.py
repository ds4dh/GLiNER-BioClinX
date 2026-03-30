# USER INPUTS

UI_1 = """Text:
\"\"\"
She had elevated BNP (30,000) and troponin (0.09 - higher than baseline).
\"\"\""""

UI_2 = """Text:
\"\"\"
Une ponction lombaire a été réalisée, qui était évocatrice de méningite bactérienne, et le patient a été mis sous ceftazidime.
\"\"\""""

# NER

NER_SYS_PROMPT = """You are a precise and accurate assistant that extracts named entities from multilingual text.

## Instructions
- Extract entities and their labels from the user-supplied text.

## Requirements
- The extracted entities must be **spelled exactly as they appear in the text**.
- Every label must be **written in the source language of the text**.
- Return your result in the following **JSON format** (and nothing else):
```json
[
  {
    "span": "...",
    "label": "..."
  },
  ...
]
```"""

NER_AO_1 = """[
  {
    "span": "BNP",
    "label": "Biomarker"
  },
  {
    "span": "30,000",
    "label": "Lab value"
  },
  {
    "span": "troponin",
    "label": "Biomarker"
  },
  {
    "span": "0.09",
    "label": "Lab value"
  }
]"""

NER_AO_2 = """[
  {
    "span": "ponction lombaire",
    "label": "Procédure"
  },
  {
    "span": "méningite bactérienne",
    "label": "Maladie"
  },
  {
    "span": "ceftazidime",
    "label": "Médicament"
  }
]"""

# HARD NER

HARD_NER_SYS_PROMPT = """You are a precise and accurate assistant that generates extraction prompts and named entities from multilingual text.

## Instructions
- Generate an extraction prompt for a named entity recognition (NER) model to extract specific entity types from the user-supplied text.
- Extract the entities that match the extraction prompt.

## Requirements
- The extracted entities must be **spelled exactly as they appear in the text**.
- The extraction prompt must be **written in the source language of the text**.
- Return your result in the following **JSON format** (and nothing else):
```json
{
  "prompt": "...",
  "entities": ["...", "..."]
}
```"""

HARD_NER_AO_1 = """{
  "prompt": "List lab test values that are above 100:",
  "entities": ["30,000"]
}"""

HARD_NER_AO_2 = """{
  "prompt": "Extrais tous les médicaments mentionnés dans le texte suivant:",
  "entities": ["ceftazidime"]
}"""

# RELATION EXTRACTION

REL_SYS_PROMPT = """You are a precise and accurate assistant that extracts binary relations from multilingual text.

## Instructions
- Identify directed relations expressed explicitly in the user-supplied text.
- Extract each relation as a (subject, relation, object) triple.

## Requirements
- The "subject" and "object" must be **spelled exactly as they appear in the text**.
- The "relation" must be **written in the source language of the text**.
- Return your result in the following **JSON format** (and nothing else):
```json
[
  {
    "subject": "...",
    "relation": "...",
    "object": "..."
  },
  ...
]
```"""

REL_AO_1 = """[
  {
    "subject": "BNP",
    "relation": "value",
    "object": "30,000"
  },
  {
    "subject": "troponin",
    "relation": "value",
    "object": "0.09"
  }
]"""

REL_AO_2 = """[
  {
    "subject": "ponction lombaire",
    "relation": "indique",
    "object": "méningite bactérienne"
  },
  {
    "subject": "patient",
    "relation": "traitement",
    "object": "ceftazidime"
  }
]"""

# EXTRACTIVE QA

EX_QA_SYS_PROMPT = """You are a precise and accurate assistant that generates extractive questions and answers from multilingual text.

## Instructions
- Generate a question that can be answered directly from the user-supplied text.
- Extract the spans from the text that answer the question.

## Requirements
- The extracted answers must be **spelled exactly as they appear in the text**.
- The question must be **written in the source language of the text**.
- Return your result in the following **JSON format** (and nothing else):
```json
{
  "question": "...",
  "answers": ["...", "..."]
}
```"""

EX_QA_AO_1 = """{
  "question": "What was the reported troponin value?",
  "answers": ["0.09"]
}"""

EX_QA_AO_2 = """{
  "question": "Quel antibiotique a été administré au patient?",
  "answers": ["ceftazidime"]
}"""

# OPEN TASK

OPEN_TASK_SYS_PROMPT = """You are an accurate and **creative** assistant that generates diverse information extraction tasks and their solutions from multilingual text.

## Instructions
- Generate a task prompt that an information extraction model can solve when given your task prompt concatenated with the user-supplied text.
- Produce the corresponding output(s) for the task prompt.

## Requirements
- The task prompt must be **written in the source language of the text**.
- The output(s) must be contiguous spans that **appear exactly** in the concatenated input: `" ".join([<task_prompt>, <user-supplied text>])`.
- The output(s) must be **spelled exactly** as they appear **in the task prompt** or **in the text**.
- Return your result in the following **JSON format** (and nothing else):
```json
{
  "task_prompt": "...",
  "output": [...]
}
```"""

OPEN_TASK_AO_1 = """{
  "task_prompt": "Find all verbs in the text:",
  "output": ["had"]
}"""

OPEN_TASK_AO_2 = """{
  "task_prompt": "Classifie le texte selon les catégories suivantes : Procédure médicale, Résultats de laboratoire, Infection bactérienne, Infection virale.",
  "output": ["Procédure médicale", "Infection bactérienne"]
}"""