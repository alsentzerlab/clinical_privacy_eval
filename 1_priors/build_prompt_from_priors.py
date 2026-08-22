import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config 


ENCOUNTER_BOUNDARY_PATTERNS = config.ENCOUNTER_BOUNDARY_PATTERNS
NOTE_START_BOUNDARY_PATTERN = config.NOTE_START_BOUNDARY_PATTERN
NOTE_HPI_BOUNDARY_PATTERNS  = config.NOTE_HPI_BOUNDARY_PATTERNS


STRUCTURED_PRIORS = [
    "public",
    "public_named",
    "public_named_meds",
]

UNSTRUCTURED_PRIORS = [
    "encounter_info",          # encounter header only (name, DOB, visit date, provider, location)
    "encounter_info_cc",       # encounter header through chief complaint / HPI header
    "encounter_info_cc_hpi",   # encounter header through end of HPI body
]

PRIOR_TIERS = STRUCTURED_PRIORS + UNSTRUCTURED_PRIORS

# note_continuation:    append end marker (section boundary)
# patient_note:         append "patient note:"

PROBE_TYPES = [
    "note_continuation",
    "patient_note",
]

INVALID_COMBINATIONS = {
    ("structured",   "note_continuation"),
    ("unstructured", "patient_note"),
}


def is_valid_combination(prior_tier: str, probe_type: str) -> bool:
    prior_type = "unstructured" if prior_tier in UNSTRUCTURED_PRIORS else "structured"
    return (prior_type, probe_type) not in INVALID_COMBINATIONS


def _earliest_match(text: str, patterns: list):
    best = None
    for p in patterns:
        m = p.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def derive_encounter_marker(note_start: str) -> tuple[str, str | None]:
    """
    Given a note_start string, return (prefix_before_boundary, boundary_marker).
    Returns ("", None) if no encounter boundary is found.
    """
    if not note_start:
        return "", None
    m = _earliest_match(note_start, ENCOUNTER_BOUNDARY_PATTERNS)
    if not m:
        return "", None
    start, end = m.span()
    lead = start
    while lead > 0 and note_start[lead - 1] in (" ", "\t", "\n"):
        lead -= 1
    before = note_start[:start].rstrip()
    marker = note_start[lead:end]
    if end < len(note_start) and note_start[end] == ":":
        marker += ":"
    return before, marker


def derive_note_start_marker(full_note: str, note_start: str) -> str | None:
    """Boundary right after note_start (through the chief-complaint / HPI header)."""
    if not full_note or not note_start:
        return None
    idx = full_note.find(note_start)
    if idx == -1:
        idx = full_note.find(note_start.rstrip())
        if idx == -1:
            return None
        after = full_note[idx + len(note_start.rstrip()):]
    else:
        after = full_note[idx + len(note_start):]
    m = NOTE_START_BOUNDARY_PATTERN.match(after)
    if not m:
        m = NOTE_START_BOUNDARY_PATTERN.search(after)
        if not m:
            return None
    return after[m.start(): m.end()]


def derive_note_hpi_marker(full_note: str, note_hpi: str) -> str | None:
    """Boundary right after the HPI body (start of ROS / preventive screening)."""
    if not full_note or not note_hpi:
        return None
    idx = full_note.find(note_hpi)
    if idx == -1:
        idx = full_note.find(note_hpi.rstrip())
        if idx == -1:
            return None
        after = full_note[idx + len(note_hpi.rstrip()):]
    else:
        after = full_note[idx + len(note_hpi):]
    m = _earliest_match(after, NOTE_HPI_BOUNDARY_PATTERNS)
    if not m:
        return None
    return after[m.start(): m.end()]


def format_public_fields(rec: dict) -> str:
    parts = []
    if rec.get("age") is not None:      parts.append(f"age: {rec['age']}")
    if rec.get("gender"):               parts.append(f"gender: {rec['gender']}")
    if rec.get("marital_status"):       parts.append(f"marital status: {rec['marital_status']}")
    if rec.get("children") is not None: parts.append(f"number of children: {rec['children']}")
    if rec.get("occupation"):           parts.append(f"occupation: {rec['occupation']}")
    return "\n".join(parts)


def format_prefix(rec: dict, prior_tier: str) -> str | None:
    """Return the structured prior string, or None if required fields are absent."""
    base = format_public_fields(rec)
    if not base:
        return None

    if prior_tier == "public":
        return base

    elif prior_tier == "public_named":
        if not rec.get("name"):
            return None
        return f"name: {rec['name']}\n{base}"

    elif prior_tier == "public_named_meds":
        if not rec.get("name") or not rec.get("current_medications"):
            return None
        return f"name: {rec['name']}\n{base}\nmedications: {rec['current_medications']}"

    else:
        raise ValueError(f"Unknown structured prior tier: {prior_tier}")


def get_unstructured_prefix_and_marker(rec: dict, prior_tier: str) -> tuple[str | None, str]:
    """
    Resolve (prefix_text, end_marker) for an unstructured prior tier by
    locating section boundaries in last_note_text. Returns (None, "") if the
    prefix cannot be derived.
    """
    full_note = rec.get("last_note_text", "") or ""

    if prior_tier == "encounter_info":
        note_start = rec.get("note_start", "") or ""
        prefix, marker = derive_encounter_marker(note_start)
        if not prefix or marker is None:
            return None, ""
        return prefix, marker

    elif prior_tier == "encounter_info_cc":
        prefix = rec.get("note_start", "") or ""
        if not prefix or not full_note:
            return None, ""
        marker = derive_note_start_marker(full_note, prefix)
        if marker is None:
            return None, ""
        return prefix, marker

    elif prior_tier == "encounter_info_cc_hpi":
        note_start = rec.get("note_start", "") or ""
        note_hpi   = rec.get("note_hpi", "") or ""
        if not note_start or not note_hpi or not full_note:
            return None, ""
        ns_marker  = derive_note_start_marker(full_note, note_start)
        hpi_marker = derive_note_hpi_marker(full_note, note_hpi)
        if ns_marker is None or hpi_marker is None:
            return None, ""
        return note_start + ns_marker + note_hpi, hpi_marker

    else:
        raise ValueError(f"Unknown unstructured prior tier: {prior_tier}")


def build_probe_suffix(probe_type: str, end_marker: str = "") -> str:
    """
    Return the text appended after the prior.

    note_continuation:    appends the end marker (section boundary) to continue
                          the note naturally — end_marker comes from the
                          derived boundary marker.
    patient_note:         appends "\\npatient note:" (no end marker).
    """
    if probe_type == "note_continuation":
        return end_marker
    elif probe_type == "patient_note":
        return "\npatient note:"
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def build_prompt_record(
    rec:        dict,
    prior_tier: str,
    probe_type: str,
) -> dict | None:
    """
    Build a complete prompt record for one (rec, prior_tier, probe_type) triple.

    The returned dict always contains:
        patient_id, encounter_date, prior_tier, probe_type,
        eval_dx_assignment, eval_dx_label,
        num_train_notes, ground_truth, last_note_text, prompt.
    """
    uid = rec.get("patient_id", "?")
    tag = f"uid={uid} prior={prior_tier} probe={probe_type}"

    if not is_valid_combination(prior_tier, probe_type):
        print(f"    [prompt_none] {tag} — invalid combination")
        return None

    sd = rec["eval_dx_assignment"]

    if prior_tier in UNSTRUCTURED_PRIORS:
        prefix_text, end_marker = get_unstructured_prefix_and_marker(rec, prior_tier)
        if prefix_text is None:
            print(f"    [prompt_none] {tag} — {prior_tier} could not be derived "
                  f"from note_start/note_hpi/last_note_text")
            return None

    else:
        prefix_text = format_prefix(rec, prior_tier)
        if prefix_text is None:
            missing = []
            pub = format_public_fields(rec)
            if not pub:
                missing.append("all_public_fields(age/gender/marital/occupation)")
            if prior_tier in ("public_named", "public_named_meds") and not rec.get("name"):
                missing.append("name")
            if prior_tier == "public_named_meds" and not rec.get("current_medications"):
                missing.append("current_medications")
            print(f"    [prompt_none] {tag} — format_prefix=None  missing={missing}")
            return None
        end_marker = ""

    probe_suffix = build_probe_suffix(probe_type, end_marker=end_marker)
    prompt       = prefix_text + probe_suffix

    record = {
        "patient_id":         rec["patient_id"],
        "encounter_date":     rec["encounter_date"],
        "prior_tier":         prior_tier,
        "probe_type":         probe_type,
        "eval_dx_assignment": sd,
        "eval_dx_label":      rec.get("eval_dx_label"),
        "num_train_notes":    rec["num_train_notes"],
        "ground_truth": {
            k[: -len("_present")]: v
            for k, v in rec.items()
            if k.endswith("_present")
        },
        "last_note_text": rec["last_note_text"],
        "prompt":         prompt,
    }

    return record
