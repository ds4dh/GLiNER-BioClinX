from typing import Any, Dict, List, Optional, Tuple, Literal
from openai import AsyncOpenAI
from collections import defaultdict
import stanza
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42


class GENConfig:

    model = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
    base_url = "http://localhost:8000/v1"
    api_key = "EMPTY"

    seed = 42
    top_k = 20
    temperature = 0.6
    top_p = 0.95
    enable_thinking = True


SCHEMA_LABELING = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["ID", "span", "hypernym"],
        "properties": {
            "ID": {"type": "integer", "minimum": 1},
            "span": {"type": "string", "minLength": 1},
            "hypernym": {
                "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]
            },
        },
    },
}


async def one_chat(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    *,
    cfg: Optional[GENConfig] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    top_k: Optional[int] = None,
    guided_json: Optional[Dict[str, Any]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Any:

    # 1) Resolve parameters, falling back to cfg when present
    resolved_model = model or (cfg.model if cfg else None)
    resolved_temp = (temperature if temperature is not None else (cfg.temperature if cfg else None))
    resolved_top_p = top_p if top_p is not None else (cfg.top_p if cfg else None)
    resolved_seed = seed if seed is not None else (cfg.seed if cfg else None)
    resolved_top_k = top_k if top_k is not None else (cfg.top_k if cfg else None)
    resolved_think = (
        chat_template_kwargs
        if chat_template_kwargs is not None
        else ({"enable_thinking": cfg.enable_thinking} if cfg else {})
    )

    # 2) Build extra_body payload
    body: Dict[str, Any] = {
        "seed": resolved_seed,
        "top_k": resolved_top_k,
        "chat_template_kwargs": resolved_think,
    }
    if guided_json is not None:
        body["guided_json"] = guided_json
    if extra_body:
        body.update(extra_body)

    # 3) Call the API
    return await client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=resolved_temp,
        top_p=resolved_top_p,
        extra_body=body,
    )


class LanguageSplitter:
    """
    Split text into sentences or tokens, with start/end char offsets.
    Uses stanza for robust, multilingual tokenization (incl. Chinese/Japanese).
    """

    def __init__(self):
        self._pipelines: Dict[str, stanza.Pipeline] = {}

    def _normalize_lang(self, code: str) -> str:
        # Map langdetect codes to stanza codes when they differ
        mapping = {"zh-cn": "zh"}
        return mapping.get(code, code)

    def _get_pipeline(self, lang_code: str) -> stanza.Pipeline:
        lang_code = self._normalize_lang(lang_code)
        if lang_code not in self._pipelines:
            stanza.download(lang_code, verbose=False)
            self._pipelines[lang_code] = stanza.Pipeline(
                lang=lang_code,
                processors="tokenize",
                verbose=False,
            )
        return self._pipelines[lang_code]

    def detect_language(self, text: str) -> str:
        if not text.strip():
            raise ValueError("Cannot detect language of empty text.")
        return detect(text)

    def split(
        self,
        text: str,
        level: Literal["sentence", "token"] = "sentence",
        lang_code: str | None = None,
    ) -> List[Dict[str, int | str]]:
        """
        Returns a list of dicts with keys: 'text', 'start', 'end', 'idx'

        level='sentence' → stanza sentence spans
        level='token'    → stanza token spans (word-level even for zh/ja)
        """
        if not text.strip():
            return []

        if lang_code is None:
            lang_code = self.detect_language(text)
        pipe = self._get_pipeline(lang_code)
        doc = pipe(text)

        out = []
        if level == "sentence":
            for i, sent in enumerate(doc.sentences):
                if not sent.tokens:
                    continue
                start = sent.tokens[0].start_char
                end = sent.tokens[-1].end_char
                out.append(
                    {"text": text[start:end], "start": start, "end": end, "idx": i}
                )
        elif level == "token":
            tidx = 0
            for sent in doc.sentences:
                for tok in sent.tokens:
                    out.append(
                        {
                            "text": tok.text,
                            "start": tok.start_char,
                            "end": tok.end_char,
                            "idx": tidx,
                        }
                    )
                    tidx += 1
        else:
            raise ValueError("level must be 'sentence' or 'token'.")

        return out


def build_id_map(
    text: str,
    entities: List[Dict[str, object]],
    greedy: bool = False,
    case_sensitive: bool = False,
    allow_nested: bool = False,
    tokens: List[Dict[str, object]] | None = None,
    require_token_boundaries: bool = True,
) -> Tuple[Dict[str, dict], List[Tuple[int, str]], Dict[int, List[str]]]:
    """
    Return:
        id_map               – {'E1': {'text', 'start', 'end', 'uid'}, ...}
        unique_ordered_spans – [(uid, text), ...]       # document order of first encounter
        uid_to_tags          – {uid: ['E1', 'E7', ...], ...}
    """
    # ---------- helper --------------------------------------------------
    def build_boundary_sets(tok_list):
        return ({t["start"] for t in tok_list}, {t["end"] for t in tok_list})

    # ---------- span discovery -----------------------------------------
    hay = text if case_sensitive else text.lower()
    spans: List[Tuple[int, int]] = []

    if greedy:
        if any("text" not in e for e in entities):
            raise ValueError("Greedy mode needs 'text' in each entity.")
        for ent in entities:
            needle_raw = ent["text"]
            needle = needle_raw if case_sensitive else needle_raw.lower()
            L = len(needle_raw)
            p = 0
            while True:
                idx = hay.find(needle, p)
                if idx == -1:
                    break
                spans.append((idx, idx + L))
                p = idx + L  # skip over this hit
    else:
        spans.extend((int(e["start"]), int(e["end"])) for e in entities)

    # ---------- deduplicate & filter -----------------------------------
    spans = sorted(set(spans))

    if require_token_boundaries and tokens:
        start_set, end_set = build_boundary_sets(tokens)
        spans = [(s, e) for (s, e) in spans if s in start_set and e in end_set]

    if not allow_nested and spans:
        spans_sorted = sorted(spans, key=lambda x: -(x[1] - x[0]))  # longest first
        kept: List[Tuple[int, int]] = []
        for s, e in spans_sorted:
            if any(not (e <= S or s >= E) for S, E in kept):
                continue
            kept.append((s, e))
        spans = sorted(kept)

    # ---------- build id_map -------------------------------------------
    id_map: Dict[str, dict] = {}
    for i, (s, e) in enumerate(spans, start=1):
        tag = f"E{i}"
        id_map[tag] = {"text": text[s:e], "start": s, "end": e}  # uid filled in later

    # ---------- canonical‐text grouping --------------------------------
    canon_to_uid: Dict[str, int] = {}
    uid_to_tags: Dict[int, List[str]] = defaultdict(list)
    unique_ordered_spans: List[Tuple[int, str]] = []

    # iterate in tag order so “first occurrence in doc” is respected
    for tag in sorted(id_map, key=lambda t: int(t[1:])):
        span_text = id_map[tag]["text"]
        canon = span_text if case_sensitive else span_text.lower()

        # assign / reuse uid
        if canon not in canon_to_uid:
            uid = len(canon_to_uid) + 1
            canon_to_uid[canon] = uid
            unique_ordered_spans.append((uid, span_text))
        else:
            uid = canon_to_uid[canon]

        id_map[tag]["uid"] = uid
        uid_to_tags[uid].append(tag)

    return id_map, unique_ordered_spans, uid_to_tags