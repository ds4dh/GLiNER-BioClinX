from typing import Any, Dict, List, Callable, Optional, Tuple, Set
from openai import AsyncOpenAI
import httpx
import stanza

# ---- Your prompts & schemas ----
from .task_conversations import (
    # prompt builders
    create_ner_messages,
    create_hard_ner_messages,
    create_rel_messages,
    create_extractive_qa_messages,
    create_open_task_messages,
)
from .task_json_schema import (
    # schemas
    SCHEMA_NER,
    SCHEMA_HARD_NER,
    SCHEMA_REL,
    SCHEMA_EX_QA,
    SCHEMA_OPEN_TASK,
)

# ===== vLLM / OpenAI-compatible client =====
CLIENT = AsyncOpenAI(api_key="EMPTY",
                     base_url="http://0.0.0.0:8847/v1",
                     max_retries=0,
                     timeout=httpx.Timeout(3600.0, connect=30.0, read=3600.0)) #AsyncOpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8000/v1")
MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"

# Optionally toggle Qwen3 “thinking” per request
DEFAULT_CHAT_TEMPLATE_KWARGS = {} #{"enable_thinking": True}

# ===== Generic helpers =====

def _messages_from_builder(builder: Callable[[str], List[Dict[str, str]]], text: str):
    # Your builders already place system + few-shot + final user turn.
    return builder(text)

async def _chat_call(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    guided_schema: Optional[Dict[str, Any]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "seed": seed,
        "top_k": top_k,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    if guided_schema is not None:
        # vLLM structured output: pass your JSON Schema here
        body["guided_json"] = guided_schema
    if extra_body:
        body.update(extra_body)

    return await CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra_body=body,
    )

# ===== Task-specific convenience wrappers =====

async def run_ner_chat_call(
    text: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    chat_template_kwargs: Optional[Dict[str, Any]] = DEFAULT_CHAT_TEMPLATE_KWARGS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = _messages_from_builder(create_ner_messages, text)
    return await _chat_call(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        guided_schema=SCHEMA_NER,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
    )

async def run_hard_ner_chat_call(
    text: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    chat_template_kwargs: Optional[Dict[str, Any]] = DEFAULT_CHAT_TEMPLATE_KWARGS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = _messages_from_builder(create_hard_ner_messages, text)
    return await _chat_call(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        guided_schema=SCHEMA_HARD_NER,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
    )

async def run_rel_chat_call(
    text: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    chat_template_kwargs: Optional[Dict[str, Any]] = DEFAULT_CHAT_TEMPLATE_KWARGS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = _messages_from_builder(create_rel_messages, text)
    return await _chat_call(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        guided_schema=SCHEMA_REL,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
    )

async def run_extractive_qa_chat_call(
    text: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    chat_template_kwargs: Optional[Dict[str, Any]] = DEFAULT_CHAT_TEMPLATE_KWARGS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = _messages_from_builder(create_extractive_qa_messages, text)
    return await _chat_call(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        guided_schema=SCHEMA_EX_QA,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
    )

async def run_open_task_chat_call(
    text: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: Optional[int] = 42,
    chat_template_kwargs: Optional[Dict[str, Any]] = DEFAULT_CHAT_TEMPLATE_KWARGS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = _messages_from_builder(create_open_task_messages, text)
    return await _chat_call(
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        guided_schema=SCHEMA_OPEN_TASK,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
    )

class TokenFixedLanguageSplitter:
    
    def __init__(self, lang_code: str):
        stanza.download(lang_code, verbose=False)
        self.pipeline = stanza.Pipeline(
            lang=lang_code,
            processors="tokenize",
            tokenize_no_ssplit=True,
            verbose=False,
        )

    def split(self, text: str) -> List[Dict[str, int | str]]:
        """
        Always returns tokens: list of dicts with keys:
        'text', 'start', 'end', 'idx'
        """
        if not text.strip():
            return []

        doc = self.pipeline(text)
        out: List[Dict[str, int | str]] = []

        for tidx, tok in enumerate(doc.iter_tokens()):
            out.append(
                {
                    "text": tok.text,
                    "start": tok.start_char,
                    "end": tok.end_char,
                    "idx": tidx,
                }
            )
        return out

def build_ner_gliner_format(
    passage: str,
    tokens: List[Dict[str, object]],
    labels: List[Dict[str, str]],
    *,
    greedy: bool = True,
    case_sensitive: bool = False,
    allow_nested: bool = True,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, object], bool]:
    if not passage:
        return ({"tokenized_text": [], "ner": []}, len({d.get("span","") for d in labels if d.get("span")}) == 0)

    tokens_sorted = sorted(tokens, key=lambda t: int(t["start"]))
    tokenized_text = [str(t["text"]) for t in tokens_sorted]

    start2tok: Dict[int, int] = {}
    end2tok: Dict[int, int] = {}
    for gi, tok in enumerate(tokens_sorted):
        s = int(tok["start"]); e = int(tok["end"])
        start2tok[s] = gi; end2tok[e] = gi

    n_tokens = len(tokens_sorted)

    def snap_span_to_tokens(s: int, e: int) -> Tuple[int, int] | None:
        if n_tokens == 0:
            return None
        ts = None
        for gi in range(n_tokens):
            t = tokens_sorted[gi]; ts_start, ts_end = int(t["start"]), int(t["end"])
            if ts_start <= s < ts_end or s <= ts_start:
                ts = gi; break
        if ts is None:
            return None
        te = None
        for gi in range(n_tokens - 1, -1, -1):
            t = tokens_sorted[gi]; te_start, te_end = int(t["start"]), int(t["end"])
            if te_start < e <= te_end or te_end <= e:
                te = gi; break
        if te is None or ts > te:
            return None
        return ts, te

    hay = passage if case_sensitive else passage.lower()

    # Track unique targets (by the actual matching "needle") and which are found after filtering.
    unique_targets: Set[str] = set()
    # spans_labeled entries carry the target key to compute coverage after filtering
    spans_labeled: List[Tuple[int, int, str, str]] = []  # (s,e,label,target_key)

    def add_occurrences(span_raw: str, label: str):
        if not span_raw:
            return
        needle = span_raw if case_sensitive else span_raw.lower()
        if not needle:
            return
        unique_targets.add(needle)
        L = len(span_raw)
        if L == 0:
            return
        if greedy:
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans_labeled.append((idx, idx + L, label, needle))
                p = idx + L
        else:
            idx = hay.find(needle)
            if idx != -1:
                spans_labeled.append((idx, idx + L, label, needle))

    for item in labels:
        add_occurrences(item.get("span", ""), item.get("label", ""))

    # Dedup exact (s,e,label,key)
    spans_labeled = sorted(set(spans_labeled))

    if require_token_boundaries:
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if s in start2tok and e in end2tok]

    if not allow_nested and spans_labeled:
        bounds = sorted({(s, e) for (s, e, _, _) in spans_labeled}, key=lambda x: -(x[1] - x[0]))
        kept_bounds: List[Tuple[int, int]] = []
        for s, e in bounds:
            if any(not (e <= S or s >= E) for S, E in kept_bounds):
                continue
            kept_bounds.append((s, e))
        kept_bounds = sorted(kept_bounds)
        allowed = set(kept_bounds)
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if (s, e) in allowed]

    ner_entries: List[List[object]] = []
    kept_keys: Set[str] = set()
    for s, e, lab, key in spans_labeled:
        if require_token_boundaries:
            ts = start2tok[s]; te = end2tok[e]
        else:
            snapped = snap_span_to_tokens(s, e)
            if snapped is None:
                continue
            ts, te = snapped
        ner_entries.append([ts, te, lab])
        kept_keys.add(key)

    payload = {"tokenized_text": tokenized_text, "ner": ner_entries}
    min_expected = len(unique_targets)
    passed = (min_expected == 0) or (len(kept_keys) >= min_expected)
    return payload, passed

def build_hard_ner_gliner_format(
    passage: str,
    tokens: List[Dict[str, object]],
    labels: Dict[str, object],
    *,
    greedy: bool = True,
    case_sensitive: bool = False,
    allow_nested: bool = True,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, object], bool]:
    if not passage:
        ents = labels.get("entities", []) if isinstance(labels, dict) else []
        return ({"tokenized_text": [], "ner": []}, len([e for e in ents if isinstance(e, str) and e]) == 0)

    tokens_sorted = sorted(tokens, key=lambda t: int(t["start"]))
    tokenized_text = [str(t["text"]) for t in tokens_sorted]
    start2tok, end2tok = {}, {}
    for gi, tok in enumerate(tokens_sorted):
        s = int(tok["start"]); e = int(tok["end"])
        start2tok[s] = gi; end2tok[e] = gi
    n_tokens = len(tokens_sorted)

    def snap_span_to_tokens(s: int, e: int):
        if n_tokens == 0:
            return None
        ts = None
        for gi in range(n_tokens):
            t = tokens_sorted[gi]; ts_start, ts_end = int(t["start"]), int(t["end"])
            if ts_start <= s < ts_end or s <= ts_start:
                ts = gi; break
        if ts is None:
            return None
        te = None
        for gi in range(n_tokens - 1, -1, -1):
            t = tokens_sorted[gi]; te_start, te_end = int(t["start"]), int(t["end"])
            if te_start < e <= te_end or te_end <= e:
                te = gi; break
        if te is None or ts > te:
            return None
        return ts, te

    entity_strings = labels.get("entities", []) if isinstance(labels, dict) else []
    hay = passage if case_sensitive else passage.lower()

    unique_targets: Set[str] = set()
    spans_labeled: List[Tuple[int, int, str, str]] = []  # (s,e,"match",key)

    def add_occurrences(span_raw: str):
        if not isinstance(span_raw, str) or not span_raw:
            return
        needle = span_raw if case_sensitive else span_raw.lower()
        if not needle:
            return
        unique_targets.add(needle)
        L = len(span_raw)
        if L == 0:
            return
        if greedy:
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans_labeled.append((idx, idx + L, "match", needle))
                p = idx + L
        else:
            idx = hay.find(needle)
            if idx != -1:
                spans_labeled.append((idx, idx + L, "match", needle))

    for span_raw in entity_strings:
        add_occurrences(span_raw)

    spans_labeled = sorted(set(spans_labeled))

    if require_token_boundaries:
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if s in start2tok and e in end2tok]

    if not allow_nested and spans_labeled:
        bounds = sorted({(s, e) for (s, e, _, _) in spans_labeled}, key=lambda x: -(x[1] - x[0]))
        kept_bounds: List[Tuple[int, int]] = []
        for s, e in bounds:
            if any(not (e <= S or s >= E) for S, E in kept_bounds):
                continue
            kept_bounds.append((s, e))
        kept_bounds = sorted(kept_bounds)
        allowed = set(kept_bounds)
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if (s, e) in allowed]

    ner_entries: List[List[object]] = []
    kept_keys: Set[str] = set()
    for s, e, lab, key in spans_labeled:
        if require_token_boundaries:
            ts = start2tok[s]; te = end2tok[e]
        else:
            snapped = snap_span_to_tokens(s, e)
            if snapped is None:
                continue
            ts, te = snapped
        ner_entries.append([ts, te, lab])
        kept_keys.add(key)

    payload = {"tokenized_text": tokenized_text, "ner": ner_entries}
    min_expected = len(unique_targets)
    passed = (min_expected == 0) or (len(kept_keys) >= min_expected)
    return payload, passed

def build_rel_gliner_format(
    passage: str,
    tokens: List[Dict[str, object]],
    labels: List[Dict[str, str]],
    *,
    greedy: bool = True,
    case_sensitive: bool = False,
    allow_nested: bool = True,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, object], bool]:
    if not passage:
        objs = [it.get("object","") for it in labels]
        return ({"tokenized_text": [], "ner": []}, len({o for o in objs if o}) == 0)

    tokens_sorted = sorted(tokens, key=lambda t: int(t["start"]))
    tokenized_text = [str(t["text"]) for t in tokens_sorted]
    start2tok, end2tok = {}, {}
    for gi, tok in enumerate(tokens_sorted):
        s = int(tok["start"]); e = int(tok["end"])
        start2tok[s] = gi; end2tok[e] = gi
    n_tokens = len(tokens_sorted)

    def snap_span_to_tokens(s: int, e: int):
        if n_tokens == 0:
            return None
        ts = None
        for gi in range(n_tokens):
            t = tokens_sorted[gi]; ts_start, ts_end = int(t["start"]), int(t["end"])
            if ts_start <= s < ts_end or s <= ts_start:
                ts = gi; break
        if ts is None:
            return None
        te = None
        for gi in range(n_tokens - 1, -1, -1):
            t = tokens_sorted[gi]; te_start, te_end = int(t["start"]), int(t["end"])
            if te_start < e <= te_end or te_end <= e:
                te = gi; break
        if te is None or ts > te:
            return None
        return ts, te

    hay = passage if case_sensitive else passage.lower()
    unique_targets: Set[str] = set()
    spans_labeled: List[Tuple[int, int, str, str]] = []  # (s,e,label,key)

    def add_occurrences(obj_raw: str, subj: str, rel: str):
        if not obj_raw:
            return
        needle = obj_raw if case_sensitive else obj_raw.lower()
        if not needle:
            return
        unique_targets.add(needle)
        L = len(obj_raw)
        if L == 0:
            return
        label = f"{subj} <> {rel}"
        if greedy:
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans_labeled.append((idx, idx + L, label, needle))
                p = idx + L
        else:
            idx = hay.find(needle)
            if idx != -1:
                spans_labeled.append((idx, idx + L, label, needle))

    for item in labels:
        add_occurrences(item.get("object", ""), item.get("subject", ""), item.get("relation", ""))

    spans_labeled = sorted(set(spans_labeled))

    if require_token_boundaries:
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if s in start2tok and e in end2tok]

    if not allow_nested and spans_labeled:
        bounds = sorted({(s, e) for (s, e, _, _) in spans_labeled}, key=lambda x: -(x[1] - x[0]))
        kept_bounds: List[Tuple[int, int]] = []
        for s, e in bounds:
            if any(not (e <= S or s >= E) for S, E in kept_bounds):
                continue
            kept_bounds.append((s, e))
        kept_bounds = sorted(kept_bounds)
        allowed = set(kept_bounds)
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if (s, e) in allowed]

    ner_entries: List[List[object]] = []
    kept_keys: Set[str] = set()
    for s, e, lab, key in spans_labeled:
        if require_token_boundaries:
            ts = start2tok[s]; te = end2tok[e]
        else:
            snapped = snap_span_to_tokens(s, e)
            if snapped is None:
                continue
            ts, te = snapped
        ner_entries.append([ts, te, lab])
        kept_keys.add(key)

    payload = {"tokenized_text": tokenized_text, "ner": ner_entries}
    min_expected = len(unique_targets)
    passed = (min_expected == 0) or (len(kept_keys) >= min_expected)
    return payload, passed

def build_extractive_qa_gliner_format(
    passage: str,
    tokens: List[Dict[str, object]],
    labels: Dict[str, object],
    *,
    greedy: bool = True,
    case_sensitive: bool = False,
    allow_nested: bool = True,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, object], bool]:
    answers = labels.get("answers", []) if isinstance(labels, dict) else []
    if not passage:
        return ({"tokenized_text": [], "ner": []}, len([a for a in answers if a]) == 0)

    tokens_sorted = sorted(tokens, key=lambda t: int(t["start"]))
    tokenized_text = [str(t["text"]) for t in tokens_sorted]
    start2tok, end2tok = {}, {}
    for gi, tok in enumerate(tokens_sorted):
        s = int(tok["start"]); e = int(tok["end"])
        start2tok[s] = gi; end2tok[e] = gi
    n_tokens = len(tokens_sorted)

    def snap_span_to_tokens(s: int, e: int):
        if n_tokens == 0:
            return None
        ts = None
        for gi in range(n_tokens):
            t = tokens_sorted[gi]; ts_start, ts_end = int(t["start"]), int(t["end"])
            if ts_start <= s < ts_end or s <= ts_start:
                ts = gi; break
        if ts is None:
            return None
        te = None
        for gi in range(n_tokens - 1, -1, -1):
            t = tokens_sorted[gi]; te_start, te_end = int(t["start"]), int(t["end"])
            if te_start < e <= te_end or te_end <= e:
                te = gi; break
        if te is None or ts > te:
            return None
        return ts, te

    hay = passage if case_sensitive else passage.lower()
    unique_targets: Set[str] = set()
    spans_labeled: List[Tuple[int, int, str, str]] = []  # (s,e,"answer",key)

    def add_occurrences(ans_raw: Any):
        if ans_raw is None:
            return
        ans_str = str(ans_raw)
        if not ans_str:
            return
        needle = ans_str if case_sensitive else ans_str.lower()
        if not needle:
            return
        unique_targets.add(needle)
        L = len(ans_str)
        if L == 0:
            return
        if greedy:
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans_labeled.append((idx, idx + L, "answer", needle))
                p = idx + L
        else:
            idx = hay.find(needle)
            if idx != -1:
                spans_labeled.append((idx, idx + L, "answer", needle))

    for a in answers if isinstance(answers, list) else []:
        add_occurrences(a)

    spans_labeled = sorted(set(spans_labeled))

    if require_token_boundaries:
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if s in start2tok and e in end2tok]

    if not allow_nested and spans_labeled:
        bounds = sorted({(s, e) for (s, e, _, _) in spans_labeled}, key=lambda x: -(x[1] - x[0]))
        kept_bounds: List[Tuple[int, int]] = []
        for s, e in bounds:
            if any(not (e <= S or s >= E) for S, E in kept_bounds):
                continue
            kept_bounds.append((s, e))
        kept_bounds = sorted(kept_bounds)
        allowed = set(kept_bounds)
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if (s, e) in allowed]

    ner_entries: List[List[object]] = []
    kept_keys: Set[str] = set()
    for s, e, lab, key in spans_labeled:
        if require_token_boundaries:
            ts = start2tok[s]; te = end2tok[e]
        else:
            snapped = snap_span_to_tokens(s, e)
            if snapped is None:
                continue
            ts, te = snapped
        ner_entries.append([ts, te, lab])
        kept_keys.add(key)

    payload = {"tokenized_text": tokenized_text, "ner": ner_entries}
    min_expected = len(unique_targets)
    passed = (min_expected == 0) or (len(kept_keys) >= min_expected)
    return payload, passed

def build_open_task_gliner_format(
    passage: str,
    tokens: List[Dict[str, object]],
    labels: Dict[str, object],
    *,
    greedy: bool = True,
    case_sensitive: bool = False,
    allow_nested: bool = True,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, object], bool]:
    outputs = labels.get("output", []) if isinstance(labels, dict) else []
    if not passage:
        return ({"tokenized_text": [], "ner": []}, len([o for o in outputs if isinstance(o, str) and o]) == 0)

    tokens_sorted = sorted(tokens, key=lambda t: int(t["start"]))
    tokenized_text = [str(t["text"]) for t in tokens_sorted]
    start2tok, end2tok = {}, {}
    for gi, tok in enumerate(tokens_sorted):
        s = int(tok["start"]); e = int(tok["end"])
        start2tok[s] = gi; end2tok[e] = gi
    n_tokens = len(tokens_sorted)

    def snap_span_to_tokens(s: int, e: int):
        if n_tokens == 0:
            return None
        ts = None
        for gi in range(n_tokens):
            t = tokens_sorted[gi]; ts_start, ts_end = int(t["start"]), int(t["end"])
            if ts_start <= s < ts_end or s <= ts_start:
                ts = gi; break
        if ts is None:
            return None
        te = None
        for gi in range(n_tokens - 1, -1, -1):
            t = tokens_sorted[gi]; te_start, te_end = int(t["start"]), int(t["end"])
            if te_start < e <= te_end or te_end <= e:
                te = gi; break
        if te is None or ts > te:
            return None
        return ts, te

    hay = passage if case_sensitive else passage.lower()
    unique_targets: Set[str] = set()
    spans_labeled: List[Tuple[int, int, str, str]] = []  # (s,e,"match",key)

    def add_occurrences(span_raw: str):
        if not isinstance(span_raw, str) or not span_raw:
            return
        needle = span_raw if case_sensitive else span_raw.lower()
        if not needle:
            return
        unique_targets.add(needle)
        L = len(span_raw)
        if L == 0:
            return
        if greedy:
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans_labeled.append((idx, idx + L, "match", needle))
                p = idx + L
        else:
            idx = hay.find(needle)
            if idx != -1:
                spans_labeled.append((idx, idx + L, "match", needle))

    for span_raw in outputs:
        add_occurrences(span_raw)

    spans_labeled = sorted(set(spans_labeled))

    if require_token_boundaries:
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if s in start2tok and e in end2tok]

    if not allow_nested and spans_labeled:
        bounds = sorted({(s, e) for (s, e, _, _) in spans_labeled}, key=lambda x: -(x[1] - x[0]))
        kept_bounds: List[Tuple[int, int]] = []
        for s, e in bounds:
            if any(not (e <= S or s >= E) for S, E in kept_bounds):
                continue
            kept_bounds.append((s, e))
        kept_bounds = sorted(kept_bounds)
        allowed = set(kept_bounds)
        spans_labeled = [(s, e, lab, key) for (s, e, lab, key) in spans_labeled if (s, e) in allowed]

    ner_entries: List[List[object]] = []
    kept_keys: Set[str] = set()
    for s, e, lab, key in spans_labeled:
        if require_token_boundaries:
            ts = start2tok[s]; te = end2tok[e]
        else:
            snapped = snap_span_to_tokens(s, e)
            if snapped is None:
                continue
            ts, te = snapped
        ner_entries.append([ts, te, lab])
        kept_keys.add(key)

    payload = {"tokenized_text": tokenized_text, "ner": ner_entries}
    min_expected = len(unique_targets)
    passed = (min_expected == 0) or (len(kept_keys) >= min_expected)
    return payload, passed