
import re
from pathlib import Path

# Project-relative defaults; edit directly for a different storage layout:
#
#     <repo>/
#       data/
#         training_cohort.jsonl
#         val_cohort.jsonl
#         notes_pre_annotation.jsonl
#         annotated_cohort.jsonl
#         eval_cohorts/<diagnosis>_eval_cohort.jsonl
#       models/checkpoint-*/
#       cache/tokenized/
#       privacy_eval_results/checkpoint-*/<diagnosis>/<prior>__<probe>.jsonl
#       analysis/content_classification_audit/
#       figures/

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR              = BASE_DIR / "data"
TRAIN_COHORT_PATH      = DATA_DIR / "training_cohort.jsonl"
VAL_COHORT_PATH        = DATA_DIR / "val_cohort.jsonl"
PRE_ANNOTATION_PATH    = DATA_DIR / "notes_pre_annotation.jsonl"
ANNOTATED_COHORT_PATH  = DATA_DIR / "annotated_cohort.jsonl"
EVAL_COHORT_DIR        = DATA_DIR / "eval_cohorts"
# Diagnosis definitions (ICD-10 codes / medications / symptoms) read by
# 0_cohort_creation/annotate_for_sensitive_dx.py.
SENSITIVE_DX_CSV       = BASE_DIR / "1_priors" / "sensitive_diagnosis.csv"

# Base (pre-continual-pretraining) model — HF hub id or local path.
BASE_MODEL_PATH = "Qwen/Qwen3.5-9B"
# transformers class name for BASE_MODEL_PATH's architecture, and the
# matching decoder-layer class name for FSDP transformer_layer_cls_to_wrap.
MODEL_CLASS     = "Qwen3_5ForCausalLM"
FSDP_LAYER_CLS  = "Qwen3_5DecoderLayer"
RUN_NAME        = "qwen3.5-9b-clinical-cpt"
CPT_MODEL_BASE  = BASE_DIR / "models" / "cpt"
TOKENIZER_PATH = BASE_MODEL_PATH
TOKENIZED_CACHE_DIR = BASE_DIR / "cache" / "tokenized"
TRAINING_OUTPUT_DIR = CPT_MODEL_BASE

WANDB_ENTITY  = ""
WANDB_PROJECT = "clinical-privacy-eval"


RESULTS_DIR = BASE_DIR / "privacy_eval_results"
DEFAULT_CHECKPOINT = "final"
AUDIT_DIR = BASE_DIR / "analysis" / "content_classification_audit"
FIGURES_DIR = BASE_DIR / "figures"

# The full candidate panel of sensitive diagnoses considered during
# cohort construction. Populate 1_priors/sensitive_diagnosis.csv with the matching
# ICD-10 codes / medications / symptoms

SENSITIVE_DIAGNOSES = [
    "major_depressive_disorder",
    "ptsd",
    "anxiety",
    "bipolar_disorder",
    "abortion",
    "hiv",
]

# Note-boundary regexes used to derive priors from full notes

# Marks the boundary right after the encounter/demographic header block.
ENCOUNTER_BOUNDARY_PATTERNS = [
    re.compile(r'electronically', re.IGNORECASE),
    re.compile(r'subjective',     re.IGNORECASE),
]

# Marks the boundary right after the "encounter_info_cc" prior (through the
# chief-complaint / HPI header).
NOTE_START_BOUNDARY_PATTERN = re.compile(r'\n \n hpi:', re.IGNORECASE)

# Marks the boundary right after the HPI body (start of Review of Systems /
# preventive screening).
NOTE_HPI_BOUNDARY_PATTERNS = [
    re.compile(r'\n ros: \nconstitutional:',           re.IGNORECASE),
    re.compile(r'\n \n preventative screening tests:', re.IGNORECASE),
]

# Section-header taxonomy + templated-documentation regexes
# Used by verbatim_memorization/content_classification/ to assign each
# memorized region to a clinical note section (find_span_headers.py) and to
# classify tokens within a section as templated vs. clinically revealing
# (content_classification.py).

# The 73 section headers from the SOAP-style note template used in the
# training corpus.
KNOWN_HEADERS = [
    "visit date", "provider", "location",
    "subjective", "cc", "hpi",
    "ros",
    "constitutional", "e/n/t", "cardiovascular", "respiratory",
    "gastrointestinal", "genitourinary", "musculoskeletal",
    "integumentary", "allergic/immunologic", "psychiatric",
    "eyes", "lymphatic", "general", "nose", "neck", "skin", "neurologic",
    "past medical history / family history / social history",
    "past medical history",
    "current medical providers",
    "preventive health maintenance",
    "surgical history", "family history", "social history",
    "occupation", "marital status", "children", "hobbies/recreation",
    "exercise", "functional status",
    "tobacco/alcohol/supplements", "caffeine",
    "substance abuse history", "mental health history",
    "communicable diseases (eg stds)",
    "current problems", "immunizations",
    "allergies", "current medications",
    "objective", "vitals", "physical exam",
    "ht", "wt", "bmi", "bp", "p", "r", "sat",
    "lab/test results",
    "assessment", "plan", "medications", "patient recommendations",
    "charge capture", "primary diagnosis", "orders",
    "vaccine", "exams", "alcohol", "prescriptions",
    "gynecological history", "hospitalizations",
    "history",
    "hematologic/lymphatic", "endocrine",
]

# Section-header taxonomy: header text -> (role, parent-or-None).
# Single configurable table find_span_headers.py's classify_header_match and
# resolve_section_for_region both look up generically — a different corpus's
# SOAP template may define an entirely different header hierarchy; edit this
# one table, not the resolver logic.
#
#   role:
#     "parent"    a parent-defining header (e.g. "ros", "physical exam");
#                 parent is the resolved section prefix ("pe"/"ros").
#     "pe_leaf"   leaf header that always resolves under "pe".
#     "ros_leaf"  leaf header that always resolves under "ros".
#     "both_leaf" leaf header that could be under either "pe" or "ros";
#                 parent is None until disambiguated by the nearest
#                 preceding parent header in the source note.
#     "vitals"    vitals sub-marker: grouped under "pe" for backward parent
#                 resolution, but not itself "pe/"-prefixed in its own
#                 section label (kept bare, e.g. "bp" not "pe/bp").
#   headers with no entry here classify as ("other", None).
SECTION_HEADER_TAXONOMY = {
    "ros":           ("parent", "ros"),
    "physical exam": ("parent", "pe"),
    "objective":     ("parent", "pe"),

    "general":    ("pe_leaf", "pe"),
    "eyes":       ("pe_leaf", "pe"),
    "nose":       ("pe_leaf", "pe"),
    "neck":       ("pe_leaf", "pe"),
    "lymphatic":  ("pe_leaf", "pe"),
    "skin":       ("pe_leaf", "pe"),
    "neurologic": ("pe_leaf", "pe"),

    "constitutional":       ("ros_leaf", "ros"),
    "genitourinary":        ("ros_leaf", "ros"),
    "integumentary":        ("ros_leaf", "ros"),
    "allergic/immunologic": ("ros_leaf", "ros"),

    "e/n/t":            ("both_leaf", None),
    "cardiovascular":   ("both_leaf", None),
    "respiratory":      ("both_leaf", None),
    "gastrointestinal": ("both_leaf", None),
    "musculoskeletal":  ("both_leaf", None),
    "psychiatric":      ("both_leaf", None),

    "vitals": ("vitals", "pe"),
    "ht":     ("vitals", "pe"),
    "wt":     ("vitals", "pe"),
    "bmi":    ("vitals", "pe"),
    "bp":     ("vitals", "pe"),
    "p":      ("vitals", "pe"),
    "r":      ("vitals", "pe"),
    "sat":    ("vitals", "pe"),
}

# Sections forced entirely templated regardless of token-level classification.
BLANKET_TEMPLATED_SECTIONS = {
    "past medical history / family history / social history",
}

# Templated-documentation regexes (content_classification.py's
# find_templated_spans). Section headers themselves (e.g. "Cardiovascular:")
# are matched separately via HEADER_PREFIX_RE, built from KNOWN_HEADERS.

# Negated review-of-systems templates, e.g. "Negative for chest pain,
# shortness of breath".
NEGATIVE_FOR_RE = re.compile(
    r"(?ix)"
    r"(?:^|(?<=\n))\s*"
    r"(?:[a-z][a-z /\-]*?:\s*)?"
    r"negative\s+for\s+"
    r"[^.\n]*"
    r"\.?",
    re.DOTALL,
)

# Last-reviewed annotations, e.g. "last reviewed 01/02/2026 by Dr. Smith, MD".
LAST_REVIEWED_RE = re.compile(
    r"(?ix)"
    r"(?:^|(?<=\n))\s*"
    r"last\s+reviewed[^\n]*",
    re.DOTALL,
)

# Sub-case of "last-reviewed annotations": bare date-led signature/revision
# lines that don't literally say "last reviewed" (LAST_REVIEWED_RE requires
# that phrasing; this catches the same class of chart-maintenance line when
# it's written as just a date).
DATE_TIME_LINE_RE = re.compile(
    r"(?ix)"
    r"(?:^|(?<=\n))\s*"
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"[^\n]*",
    re.DOTALL,
)

# Sub-case of "last-reviewed annotations": bare "by <provider name>"
# signature lines not preceded by "last reviewed" text on the same line.
BY_NAME_LINE_RE = re.compile(
    r"(?ix)"
    r"(?:^|(?<=\n))\s*"
    r"(?:[ap]m\s+)?"
    r"by\s+"
    r"(?:"
        r"(?:dr|mr|mrs|ms)\.?\s+[a-z][a-z\-']+"
        r"|"
        r"[a-z][a-z\-']+\s*,\s*[a-z][a-z\-' \.]{1,25}"
        r"|"
        r"[a-z][a-z\-']+\s+(?:md|do|np|pa|rn|m\.?d\.?|d\.?o\.?)\b"
    r")"
    r"[^\n]{0,5}"
    r"(?=\s*$|\s*\n)",
    re.DOTALL,
)

# Cross-references to other sections, e.g. "gynecological history: see HPI".
SEE_REFERENCE_RE = re.compile(
    r"(?ix)"
    r"\bsee\s+"
    r"(?:"
        r"hpi|history|ros|pe|p\.?e\.?|exam|"
        r"physical(?:\s+exam(?:ination)?)?|"
        r"note|notes|chart|"
        r"assessment|plan|a\s*/\s*p|"
        r"above|below|prior|previous|"
        r"attached|attachment"
    r")"
    r"\b[^.\n]*\.?",
    re.DOTALL,
)

# Single configurable mapping content_classification.py's find_templated_spans
# iterates over generically (rule name -> compiled regex), rather than the
# function hardcoding one loop per named regex. A different corpus/template
# may need a different rule set entirely (more, fewer, or differently-defined
# rules) — add/remove/edit entries here without touching that function.
TEMPLATED_ARTIFACT_RULES = {
    "NEGATIVE_ROS":    NEGATIVE_FOR_RE,
    "LAST_REVIEWED":   LAST_REVIEWED_RE,
    "DATE_TIME_LINE":  DATE_TIME_LINE_RE,
    "BY_NAME_LINE":    BY_NAME_LINE_RE,
    "SEE_REFERENCE":   SEE_REFERENCE_RE,
}
