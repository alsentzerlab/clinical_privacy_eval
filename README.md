# Clinical Privacy Evaluation Framework

A clinically grounded framework for auditing privacy leakage in language
models trained on clinical notes. It evaluates leakage along a **graded axis
of adversarial access** — from publicly inferable demographics up to a
leaked note fragment — and measures two complementary forms of disclosure at
each access tier:

- **Verbatim memorization** — exact-match reproduction of a target patient's
  training text, attributed back to its source note(s) and decomposed into
  templated-documentation vs. clinically revealing content.
- **Semantic leakage** — disclosure of a sensitive diagnosis (directly, or
  through symptoms/medications), measured with a matched train/non-train
  cohort so disclosure attributable to training-set membership can be
  distinguished from population-level inference.

To apply this framework, you need:

1. A JSONL corpus of clinical notes (patient id, note text, encounter date per row), and per-patient demographic/note fields extracted from it (name, DOB, gender, marital status, occupation, children, current medications, and the note-fragment/HPI text used by the note-fragment priors).
2. An LM to evaluate.
3. Documentation template: the section-header list and templated-documentation regexes.
4. LLM judge client.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
```

## Repository layout
```
config.py                     # Need to edit to apply the framework to your repo. 
aihub_client.py               # Need to edit with LLM judge client

0_cohort_creation/
  regex_extract.sql            # extracts per-patient demographic/note fields from raw source
                                # data (name, DOB, gender, marital status, occupation, number of children,
                                # medications, note_start/note_hpi) - may need to adapt for
                                # your own schema and note template
  annotate_for_sensitive_dx.py # per-patient sensitive-dx ground truth, chief-complaint/HPI
                                # leakage flags, and medication scrubbing (LLM judge)

1_priors/
  sensitive_diagnosis.csv      # sensitive-diagnosis to evaluate: ICD-10 codes, meds, symptoms
  build_prompt_from_priors.py  # generates adversarial priors 
  generate.py                  # generates model completions per (patient, dx, prior, probe)

2_privacy_measures/
  verbatim_memorization/
    verbatim_memorization.py       # tau=30-gram verbatim-match detection
    source_note_attribution.py     # attributes memorized regions to source note(s)
    content_classification/
      find_span_headers.py         # note-section resolution (header taxonomy)
      content_classification.py    # templated- vs. clinically-revealing token classification
      manual_review/               # human-review sampling
  semantic_leakage/
    llm_judge.py                   # sensitive-diagnosis disclosure judge over generations
    manual_review/                 # human-review sampling and adjudication in the paper 

3_matched_eval_cohort_design/
  build_eval_cohort.py         # PSM-matched train/non-train x positive/negative eval cohorts

4_model_training/
  build_train_cohort.py        # training cohort selection
  build_val_cohort.py          # validation cohort selection
  pretokenize.py               # tokenize + cache the training corpus
  train.py                     # continued pretraining (FSDP)

5_analysis/
  sensitive_diagnosis_auroc_ppv.py     # AUROC/PPV tables, train-attributable delta plots, ROC curves
  verbatim_memorization/
    by_prior.py                        # memorization % and hit-rate by prior
    content_classification.py          # composition of memorized tokens breakdown by section
    recurrence.py                      # cross-patient and within-patient recurrence of regions
    memorized_region_stats.py          # distinct source notes per generation; cross-note stitching %
```

## Adapting the framework to a new model or corpus

You need to edit config.py with your filepaths, sensitive diagnoses, and regex templates. 

| Section in `config.py` | What it controls |
|---|---|
| 1. `SENSITIVE_DIAGNOSES` | the diagnosis panel audited for semantic leakage |
| 2. Note-boundary regexes (`ENCOUNTER_BOUNDARY_PATTERNS`, `NOTE_START_BOUNDARY_PATTERN`, `NOTE_HPI_BOUNDARY_PATTERNS`) | mark where each prior tier ends in a full note |
| 3. `KNOWN_HEADERS` + `SECTION_HEADER_TAXONOMY` + `BLANKET_TEMPLATED_SECTIONS` | the note-section headers used by your corpus's template | 
| 4. `TEMPLATED_ARTIFACT_RULES` | regexes identifying boilerplate/templated documentation | 


## Data schema

**Training corpus** (`config.TRAIN_COHORT_PATH`) — one row per note:
```json
{"PatientUid": "...", "Note": "full note text...", "Encounter_Date": "2024-01-15"}
```

**Pre-annotation cohort** (`config.PRE_ANNOTATION_PATH`, output of
`0_cohort_creation/regex_extract.sql` or your equivalent extraction
 — one row per patient, with the demographic/note
fields prompt construction consumes: `extracted_name`, `extracted_dob`,
`patient_gender`, `marital_status`, `occupation`, `children`,
`medications`, `note_start` (everything up to the HPI header for last note), `note_hpi` (note_start + HPI for last note), `last_note`. 

**Annotated cohort** (`config.ANNOTATED_COHORT_PATH`, output of
`0_cohort_creation/annotate_for_sensitive_dx.py`) — the pre-annotation cohort
plus, per diagnosis, the fields the LLM judge fills in: `{dx}_present`
(cohort-labeling ground truth), `{dx}_in_note_start`/`{dx}_in_hpi`
(chief-complaint/HPI leakage flags used to exclude patients from the
note-fragment eval tiers), `{dx}_medications_filtered`, `{dx}_medications_to_remove`, and the
evidence spans behind each judgment (`{dx}_diagnosis_spans`,
`{dx}_icd10_spans`, `{dx}_symptom_spans`, `{dx}_medication_spans`).
## Running the pipeline

```bash
# 0a. Extract per-patient demographic/note fields from your raw source data
#     (0_cohort_creation/regex_extract.sql is the query used for this paper)

# 0b. Annotate patients for sensitive-diagnosis evidence, HPI/CC leakage, and medication scrubbing
python 0_cohort_creation/annotate_for_sensitive_dx.py

# 1. Generate model completions for every (prior, probe, diagnosis).
for prior in public public_named public_named_meds; do
  python 1_priors/generate.py --prior_tier "$prior" --probe_type patient_note --checkpoint final
done
for prior in encounter_info encounter_info_cc encounter_info_cc_hpi; do
  python 1_priors/generate.py --prior_tier "$prior" --probe_type note_continuation --checkpoint final
done

# 2. Verbatim-memorization detection, source-note attribution, and content classification
python 2_privacy_measures/verbatim_memorization/verbatim_memorization.py --apply
python 2_privacy_measures/verbatim_memorization/source_note_attribution.py --apply
python 2_privacy_measures/verbatim_memorization/content_classification/find_span_headers.py --apply
python 2_privacy_measures/verbatim_memorization/content_classification/content_classification.py --apply

# 2a. Manual review: validate the templated/revealing classifier against human judgment
python 2_privacy_measures/verbatim_memorization/content_classification/manual_review/sample_spans_for_review.py
# fill in human_label on each row of review_blind.csv 
python 2_privacy_measures/verbatim_memorization/content_classification/manual_review/score_review.py \
    --blind analysis/manual_review/review_blind.csv --key analysis/manual_review/review_key.csv

# 2b. Sensitive-diagnosis disclosure judge over the generations
python 2_privacy_measures/semantic_leakage/llm_judge.py

# 2c. Manual review: validate the LLM judge against human annotation (inter-rater agreement) (may want to do thi before 2b)
python 2_privacy_measures/semantic_leakage/manual_review/sample_for_annotation.py
# fill in the manual_* columns of annotation_sheet.csv...
python 2_privacy_measures/semantic_leakage/manual_review/compute_kappa.py

# 3. Build matched train/non-train x positive/negative eval cohorts (PSM)
python 3_matched_eval_cohort_design/build_eval_cohort.py

# 4. Select training / validation cohorts, then continually pretrain the model
python 4_model_training/build_train_cohort.py
python 4_model_training/build_val_cohort.py
python 4_model_training/pretokenize.py
python 4_model_training/train.py

# 5. Reproduce the paper's figures/tables from computed results
python 5_analysis/verbatim_memorization/by_prior.py
python 5_analysis/verbatim_memorization/content_classification.py
python 5_analysis/verbatim_memorization/recurrence.py
python 5_analysis/verbatim_memorization/memorized_region_stats.py
python 5_analysis/sensitive_diagnosis_auroc_ppv.py
```


## Citation

Sasha Ronaghi, Sana Tonekaboni, Lena Stempfle, Vivian Utti, Jordan Li Cahoon,
Nathaniel Hendrix, Ayin Vala, Marzyeh Ghassemi, Emily Alsentzer. **Clinically
Grounded Privacy Evaluation of Medical LMs.** EMNLP 2026.
[arXiv:2606.09590](https://arxiv.org/abs/2606.09590)

```bibtex
@inproceedings{ronaghi2026clinicallygroundedprivacyevaluation,
  title={Clinically Grounded Privacy Evaluation of Medical LMs}, 
  author={Sasha Ronaghi and Sana Tonekaboni and Lena Stempfle and Vivian Utti and Jordan Li Cahoon and Nathaniel Hendrix and Ayin Vala and Marzyeh Ghassemi and Emily Alsentzer},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.09590}
}

```
