CREATE OR REPLACE TABLE `pre_annotation_cohort` AS

WITH filtered AS (
  SELECT
    PatientUid,
    Encounter_Date,
    Note
  FROM `original_data'
  WHERE LOWER(TRIM(Section_Name)) = "doctor note"
    AND Encounter_Date >= TIMESTAMP("2019-01-01")
    AND Note IS NOT NULL
    AND TRIM(Note) != ""
    AND LENGTH(Note) < 32000
),

last_note_agg AS (
  SELECT
    PatientUid,
    ARRAY_AGG(Note ORDER BY Encounter_Date DESC LIMIT 1)[OFFSET(0)] AS last_note
  FROM filtered
  GROUP BY PatientUid
),

occupation_agg AS (
  SELECT
    PatientUid,
    TRIM(REGEXP_EXTRACT(
      ARRAY_AGG(Note ORDER BY Encounter_Date DESC LIMIT 1)[OFFSET(0)],
      r'(?i)occupation\s*:\s*([\s\S]+?)\s*(?:;|marital status\s*:)'
    )) AS occupation
  FROM filtered
  WHERE REGEXP_CONTAINS(Note, r'(?i)occupation\s*:')
  GROUP BY PatientUid
),

marital_agg AS (
  SELECT
    PatientUid,
    TRIM(REGEXP_EXTRACT(
      ARRAY_AGG(Note ORDER BY Encounter_Date DESC LIMIT 1)[OFFSET(0)],
      r'(?i)marital status\s*:\s*([\s\S]+?)\s*children\s*:'
    )) AS marital_status
  FROM filtered
  WHERE REGEXP_CONTAINS(Note, r'(?i)marital status\s*:')
  GROUP BY PatientUid
),

children_agg AS (
  SELECT
    PatientUid,
    TRIM(REGEXP_EXTRACT(
      ARRAY_AGG(Note ORDER BY Encounter_Date DESC LIMIT 1)[OFFSET(0)],
      r'(?i)children\s*:\s*([^\n]+)'
    )) AS children
  FROM filtered
  WHERE REGEXP_CONTAINS(Note, r'(?i)children\s*:')
  GROUP BY PatientUid
),

extracted AS (
  SELECT
    n.PatientUid,
    n.last_note,

    TRIM(REGEXP_EXTRACT(n.last_note,
      r'(?s)^\s*([A-Za-z][A-Za-z\s,\.\-]+?)\s+\d{2}/\d{2}/\d{4}'
    )) AS extracted_name,

    REGEXP_EXTRACT(n.last_note,
      r'\b(\d{2}/\d{2}/\d{4})\b'
    ) AS extracted_dob,

    REGEXP_EXTRACT(n.last_note,
      r'(?is)^([\s\S]+?)\n \n hpi:'
    ) AS note_start,

    -- note_hpi: everything AFTER 'hpi:' up to first end marker, trimmed
    TRIM(REGEXP_EXTRACT(n.last_note,
      r'(?i)\n \n hpi:\s*([\s\S]+?)(?:\n ros: \nconstitutional:|\n \n preventative screening tests:)'
    )) AS note_hpi,

    ARRAY_TO_STRING(
      ARRAY(
        SELECT TRIM(line)
        FROM UNNEST(
          SPLIT(
            TRIM(REGEXP_EXTRACT(n.last_note,
              r'(?is)current\s+medications?\s*[:\-][^\n]*\n([\s\S]+?)(?:\n\s*\n|$)'
            )),
            '\n'
          )
        ) AS line
        WHERE TRIM(line) != ''
          AND NOT REGEXP_CONTAINS(line, r'(?i)^last\s+reviewed\s+on')
      ),
      '; '
    ) AS medications,

    m.marital_status,
    o.occupation,
    c.children

  FROM last_note_agg n
  LEFT JOIN marital_agg m  USING (PatientUid)
  LEFT JOIN occupation_agg o USING (PatientUid)
  LEFT JOIN children_agg c  USING (PatientUid)
),

with_patient AS (
  SELECT
    e.*,
    p.practiceid,
    p.gender AS patient_gender
  FROM extracted e
  INNER JOIN `patient_table` p
    ON LOWER(e.PatientUid) = LOWER(p.patientuid)
  WHERE p.practiceid IS NOT NULL
),

with_race AS (
  SELECT
    wp.*,
    CASE
      WHEN LOWER(TRIM(re.raceeth)) IN (
        'asian_indian', 'vietnamese', 'filipino', 'korean',
        'pakistani', 'bangladeshi', 'japanese', 'cambodian', 'chinese'
      ) THEN 'other_asian'
      ELSE LOWER(TRIM(re.raceeth))
    END AS raceeth
  FROM with_patient wp
  INNER JOIN `race_ethnicity_table` re
    ON LOWER(wp.PatientUid) = LOWER(re.patientuid)
  WHERE re.raceeth IS NOT NULL
    AND LOWER(TRIM(re.raceeth)) != "unknown"
    AND LOWER(TRIM(re.raceeth)) NOT IN ('multi', 'other')
),

final AS (
  SELECT DISTINCT
    PatientUid,
    practiceid,
    raceeth,
    patient_gender,
    extracted_name,
    extracted_dob,
    marital_status,
    occupation,
    children,
    note_start,
    note_hpi,
    medications,
    last_note
  FROM with_race
  WHERE extracted_name IS NOT NULL
    AND extracted_dob IS NOT NULL
    AND marital_status IS NOT NULL
    AND occupation IS NOT NULL
    AND children IS NOT NULL
    AND raceeth IS NOT NULL
    AND patient_gender IS NOT NULL
    AND practiceid IS NOT NULL
    AND note_start IS NOT NULL
    AND note_hpi IS NOT NULL
    AND medications IS NOT NULL 
    AND medications != ""
)

SELECT *
FROM final
WHERE PatientUid NOT IN (
  SELECT PatientUid
  FROM final
  GROUP BY PatientUid
  HAVING COUNT(*) > 1
)
ORDER BY PatientUid