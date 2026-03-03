EN_SYS_PROMPT = """## Instructions
Your task is to assign **standardized**, **human-friendly** hypernyms to user-supplied spans in their context. Your task is to go through all the spans, one at a time, and apply Step 1, Step 2 and Step 3 sequentially.

### Step 1: Span removal
- Your first task is to determine whether the span is worth annotating, i.e., whether it is generic or a valuable piece of information. For example, "protein", "gene", and "symptom" are generic and uninformative. On the other hand, the actual occurrences of such generic terms are very important. For example, "lipoprotein", "MT-TC", or "headache" are informative spans, i.e., **specific**, **identifiable** items.
- You must be strict. A span to which no **clear**, **self-contained** hypernym can be assigned is undoubtedly an irrelevant span.
- If you have several occurrences of the same interval in different forms, if one of them is deemed relevant, all other mentions are relevant, i.e., if "Lycée Madame de Staël" is relevant, then "Lycée" and "Staël" are also relevant.
- If a span is considered irrelevant, the hypernym defaults to `null`. If relevant, the span is eligible for Step 2.

### Step 2: Assigning hypernyms to relevant spans
- Every hypernym must be **written in the source language of the text**.
- **Never reuse the span’s own wording** (e.g., in "tumor", the hypernym could be "Neoplastic mass").
- Specificity must **never** come at the cost of accuracy. Sometimes, generic hypernyms are much more meaningful.
- Do **not** skip any span. You must either assign the value `null` or assign a hypernym.
- Acronyms are no different from other spans. If relevant, they should be annotated according to the same principles.
- Return solely a JSON list of objects in this format:
```json
[
  {
    "ID": 1,
    "span": "...",
    "hypernym": "..." # If relevant
  },
  {
    "ID": 2,
    "span": "...",
    "hypernym": null # If irrelevant
  }
]
```

### Step 3: Standardization
- Your last task is to standardize all the hypernyms:
    - **No variance**: If multiple spans refer to the same entity type, assign exactly the same hypernym. Avoid synonyms, rephrasings, or variations, i.e., ["7zip", "Excel"] → "logiciel" and not ["7zip"] → "logiciel", ["Excel"] → "programme informatique".
    - **Highest-level hypernym**: If spans overlap semantically, assign the most general hypernym that is still accurate for all of them, i.e., ["foie", "rate"] → "organe" and not ["foie"] → "organe abdominal", ["rate"] → "organe lymphoïde"."""

UI_1 = """Context:
\"\"\"
PURPOSE: We compared oncological outcomes according to tumor volume (TV) thresholds defining both classical and updated insignificant prostate cancer (IPC), since the TV threshold can be used as clinical parameter for active surveillance. METHODS: Between 2001 and 2012, we retrospectively analyzed 331 organ-confined prostate cancer patients who had preoperative Gleason score 6, preoperative PSA under 10 ng/mL and pathologic TV less than 1.3 mL. Among them, 81 of 331 (24.5%) had Gleason grade 4/5 disease postoperatively. Patients were stratified into two groups: (1) TV less than 0.5 mL, using the classical definition; and (2) TV between 0.5 mL and 1.3 mL, using the range of updated definition. We compared biochemical recurrence (BCR)-free survival and identified independent predictors of BCR in each group. RESULTS: Group 2 had more Gleason grade 4/5 disease than group 1 (P<0.001). On multivariate analysis, Gleason grade 4/5 disease was not associated with BCR in group 1 (P=0.132). However, it was an independent predictor for BCR in group 2 (P=0.042). BCR-free survival were not significantly different according to the presence of Gleason grade 4/5 disease in group 1 (P=0.115). However, in group 2, it was significantly different according to the presence of Gleason grade 4/5 disease (P=0.041). CONCLUSIONS: Although the TV thresholds of the two definitions of IPC vary only slightly, this difference was enough to result in different clinical course if Gleason grade 4/5 disease was present. Therefore, the updated IPC TV threshold should be carefully applied as clinical parameter for active surveillance.
\"\"\"

Candidate (ID, span) for Step 1, Step 2 and Step 3:
```
[(1, 'tumor volume'), (2, 'TV'), (3, 'classical'), (4, 'insignificant prostate cancer'), (5, 'prostate cancer'), (6, 'IPC'), (7, 'TV threshold'), (8, 'active surveillance'), (9, 'PSA'), (10, 'Gleason grade 4/5 disease'), (11, 'updated definition. We compared biochemical recurrence'), (12, 'BCR'), (13, 'Group 2'), (14, 'BCR-free survival'), (15, 'TV thresholds')]
```"""

AO_1 = """[
  {
    "ID": 1,
    "span": "tumor volume",
    "hypernym": "clinical measurement"
  },
  {
    "ID": 2,
    "span": "TV",
    "hypernym": "clinical measurement"
  },
  {
    "ID": 3,
    "span": "classical",
    "hypernym": null
  },
  {
    "ID": 4,
    "span": "insignificant prostate cancer",
    "hypernym": null
  },
  {
    "ID": 5,
    "span": "prostate cancer",
    "hypernym": "cancer type"
  },
  {
    "ID": 6,
    "span": "IPC",
    "hypernym": null
  },
  {
    "ID": 7,
    "span": "TV threshold",
    "hypernym": "clinical parameter"
  },
  {
    "ID": 8,
    "span": "active surveillance",
    "hypernym": null
  },
  {
    "ID": 9,
    "span": "PSA",
    "hypernym": "biomarker"
  },
  {
    "ID": 10,
    "span": "Gleason grade 4/5 disease",
    "hypernym": "tumor grade"
  },
  {
    "ID": 11,
    "span": "updated definition. We compared biochemical recurrence",
    "hypernym": null
  },
  {
    "ID": 12,
    "span": "BCR",
    "hypernym": "clinical endpoint"
  },
  {
    "ID": 13,
    "span": "Group 2",
    "hypernym": null
  },
  {
    "ID": 14,
    "span": "BCR-free survival",
    "hypernym": "clinical endpoint metric"
  },
  {
    "ID": 15,
    "span": "TV thresholds",
    "hypernym": "clinical criteria"
  }
]"""

UI_2 = """Context:
\"\"\"
Objectif Concevoir un système de télé-échographie en temps différé, entre un site expert et patient, où se trouve un opérateur non initié en échographie. Matériels et méthodes Un porte sonde motorisé imprimant un mouvement de TILT (±40°) à une sonde d’échographe 2D permet à un non initié de capturer l’ensemble des images d’un organe du patient. Ce volume d’images est envoyé au centre expert via internet. La séquence vidéo est décomposée en images au format JPEG par le programme VirtualDub. Un programme de navigation virtuelle (ECHO-CNES) permet à l’expert de trouver les incidences nécessaires au diagnostic. Ce système a été testé sur 50 patients au CHU Trousseau de Tours (France). Résultats Les organes investigués chez les patients étaient le foie, le tronc porte, la vésicule biliaire, les reins (droit et gauche) et la rate. Le temps d’acquisition des volumes d’image était de quatreminutes maximum et permettait de reconstruire les images nécessaires au diagnostic dans 80 % des cas. Le temps de balayage était de quatre secondes en moyenne, le temps de traitement du volume d’images était de 15minutes environ. Ce système est en cours de validation en situation réelle au Togo en vue d’une utilisation en routine dans la prise en charge médicale. Purpose Results
\"\"\"

Candidate (ID, span) for Step 1, Step 2 and Step 3:
```
[(1, 'Objectif'), (2, 'télé-échographie'), (3, 'patient'), (4, 'non initié'), (5, 'échographie'), (6, 'Matériels et méthodes'), (7, 'porte sonde motorisé'), (8, 'sonde d’échographe'), (9, 'images'), (10, 'internet'), (11, 'JPEG'), (12, 'VirtualDub'), (13, 'ECHO-CNES'), (14, 'CHU Trousseau de Tours'), (15, 'France'), (16, 'organes'), (17, 'foie'), (18, 'tronc porte'), (19, 'vésicule biliaire'), (20, 'reins'), (21, 'rate'), (22, 'Togo')]
```"""

AO_2 = """[
  {
    "ID": 1,
    "span": "Objectif",
    "hypernym": null
  },
  {
    "ID": 2,
    "span": "télé-échographie",
    "hypernym": "technique d'imagerie médicale"
  },
  {
    "ID": 3,
    "span": "patient",
    "hypernym": null
  },
  {
    "ID": 4,
    "span": "non initié",
    "hypernym": null
  },
  {
    "ID": 5,
    "span": "échographie",
    "hypernym": "technique d'imagerie médicale"
  },
  {
    "ID": 6,
    "span": "Matériels et méthodes",
    "hypernym": null
  },
  {
    "ID": 7,
    "span": "porte sonde motorisé",
    "hypernym": "accessoire de dispositif médical"
  },
  {
    "ID": 8,
    "span": "sonde d’échographe",
    "hypernym": "dispositif médical"
  },
  {
    "ID": 9,
    "span": "images",
    "hypernym": null
  },
  {
    "ID": 10,
    "span": "internet",
    "hypernym": null
  },
  {
    "ID": 11,
    "span": "JPEG",
    "hypernym": "format de fichier"
  },
  {
    "ID": 12,
    "span": "VirtualDub",
    "hypernym": "logiciel"
  },
  {
    "ID": 13,
    "span": "ECHO-CNES",
    "hypernym": "logiciel"
  },
  {
    "ID": 14,
    "span": "CHU Trousseau de Tours",
    "hypernym": "hôpital"
  },
  {
    "ID": 15,
    "span": "France",
    "hypernym": "pays"
  },
  {
    "ID": 16,
    "span": "organes",
    "hypernym": null
  },
  {
    "ID": 17,
    "span": "foie",
    "hypernym": "organe"
  },
  {
    "ID": 18,
    "span": "tronc porte",
    "hypernym": "vaisseau sanguin"
  },
  {
    "ID": 19,
    "span": "vésicule biliaire",
    "hypernym": "organe"
  },
  {
    "ID": 20,
    "span": "reins",
    "hypernym": "organe"
  },
  {
    "ID": 21,
    "span": "rate",
    "hypernym": "organe"
  },
  {
    "ID": 22,
    "span": "Togo",
    "hypernym": "pays"
  }
]"""