import re
import stanza
from langdetect import detect, LangDetectException, DetectorFactory
from collections import Counter
import logging
import time

DetectorFactory.seed = 42

# Set of officially supported language codes by your application.
# These are the languages you are willing to process.
SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "it", "es",
    "nl", "uk", "ru", "zh-cn", "ja", "pt"
}

# Stanza uses 'zh' for simplified Chinese, while langdetect may return 'zh-cn'.
# This map ensures the correct model is requested from Stanza.
LANG_MAP = {
    "zh-cn": "zh"
}

# Configure logging to show stanza download/loading status
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache for Stanza pipelines to avoid re-initializing for the same language
STANZA_PIPELINES = {}

def get_stanza_pipeline(lang: str):
    """
    Initializes and caches a Stanza pipeline for a given language.
    Downloads the model if it's not already present.
    Retries initialization up to 10 times on failure.
    """
    if lang not in STANZA_PIPELINES:
        logging.info(f"Initializing Stanza pipeline for language: {lang}...")
        for attempt in range(10): # Retry up to 10 times
            try:
                #stanza.download(lang, verbose=False, logging_level='WARN')
                STANZA_PIPELINES[lang] = stanza.Pipeline(lang, processors='tokenize', verbose=False, logging_level='WARN')
                logging.info(f"Pipeline for '{lang}' initialized successfully.")
                return STANZA_PIPELINES[lang] # Success, exit loop and function
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/10 failed to initialize Stanza pipeline for '{lang}': {e}")
                if attempt < 9: # Don't sleep after the last attempt
                    logging.info("Retrying in 5 seconds...")
                    time.sleep(5)
        
        logging.error(f"Failed to initialize Stanza pipeline for language '{lang}' after 10 attempts.")
        return None # All retries failed
        
    return STANZA_PIPELINES[lang]


def compute_quality_scores(text: str, tokens: list, sentences: list) -> dict:
    """
    Compute all heuristic scores for a given text in a language-agnostic way.
    """
    total_words = len(tokens)
    total_sentences = len(sentences)

    # Ratio of tokens that are not purely alphanumeric.
    non_alnum_words = sum(1 for word in tokens if not word.isalnum())
    special_char_ratio = non_alnum_words / total_words if total_words > 0 else 0

    # Average words per sentence
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

    # Capitalization ratio. For scripts without case, this will be 0.
    uppercase_count = sum(1 for char in text if char.isupper())
    total_letters = sum(1 for char in text if char.isalpha())
    capitalization_ratio = uppercase_count / total_letters if total_letters > 0 else 0

    # Lexical diversity (Type-Token Ratio)
    unique_words = set(tokens)
    lexical_diversity = len(unique_words) / total_words if total_words > 0 else 0

    # Repetition score (based on the most frequent token)
    word_freq = Counter(tokens)
    max_freq = max(word_freq.values(), default=0)
    repetition_score = max_freq / total_words if total_words > 0 else 0

    # Newline to sentence ratio
    newline_groups = re.findall(r'\n+', text)
    newline_to_sentence_ratio = len(newline_groups) / total_sentences if total_sentences > 0 else 0

    return {
        "special_char_ratio": special_char_ratio,
        "sentence_count": total_sentences,
        "avg_words_per_sentence": avg_words_per_sentence,
        "capitalization_ratio": capitalization_ratio,
        "lexical_diversity": lexical_diversity,
        "repetition_score": repetition_score,
        "newline_to_sentence_ratio": newline_to_sentence_ratio,
    }


def process_single_text(text: str, expected_lang: str, thresholds: dict) -> (bool, dict):
    """
    Process a single text, assert its language, and check quality thresholds.
    Returns a tuple: (passed: bool, scores_and_metadata: dict).
    The dict contains scores if passed, or a reason for failure if not.
    """
    if not isinstance(text, str) or not text.strip():
        return False, {"reason": "empty_or_invalid"}

    try:
        detected_lang = detect(text)
    except LangDetectException:
        logging.warning("Could not detect language. Skipping quality checks.")
        return False, {"reason": "language_detection_failed"}

    # Assert that the detected language matches the user-provided one.
    if detected_lang != expected_lang:
        #logging.warning(f"Language mismatch. Expected '{expected_lang}', but detected '{detected_lang}'. Skipping.")
        return False, {"reason": "language_mismatch", "detected_language": detected_lang}

    # Map to the correct stanza model name and get the pipeline
    stanza_lang = LANG_MAP.get(expected_lang, expected_lang)
    nlp = get_stanza_pipeline(stanza_lang)
    if nlp is None:
        return False, {"reason": "pipeline_failure"}

    # Process the document
    doc = nlp(text)
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    sentences = [sentence.text for sentence in doc.sentences]

    scores = compute_quality_scores(text, tokens, sentences)
    scores["detected_language"] = detected_lang

    # Check against quality thresholds
    passed = True
    for heuristic, (comp_type, threshold) in thresholds.items():
        score = scores.get(heuristic)
        if score is None: continue

        if (comp_type == 'min' and score < threshold) or \
           (comp_type == 'max' and score > threshold):
            passed = False
            scores["reason"] = f"failed_threshold_{heuristic}"
            break

    return passed, scores