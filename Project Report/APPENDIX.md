# Appendix: Important Code Snippets and Logic

## NextStep AI — Intelligent Career Prediction System

---

## A.1 Model Loading with Caching

The model is loaded once and cached for the entire server lifecycle using Streamlit's `@st.cache_resource`:

```python
@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    path = os.path.join(BASE_DIR, "career_model.pkl")
    if not os.path.exists(path):
        return None, "career_model.pkl not found"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"
```

**Logic:** If the `.pkl` file is missing, the system returns `None` and activates demo mode instead of crashing.

---

## A.2 Quiz-to-Feature Vector Engineering (`quiz_to_features`)

This function transforms 15 quiz answers into a 23-dimensional feature vector. The process involves three stages:

### Stage 1: Latent Signal Extraction

Five quiz questions (Q2, Q5, Q10, Q12, Q14, Q15) contribute to four domain signals:

```python
tech_lat = (
    (1.0 if a[1] == 0 else 0.0) +   # Weekend: coding
    (1.0 if a[4] == 0 else 0.0) +   # Subject: CS/Math
    (1.0 if a[11] == 0 else 0.0) +  # Build: app/tool
    (1.0 if a[13] == 0 else 0.0) +  # Motivation: tech
    (1.0 if a[14] == 0 else 0.0)    # Vision: tech lead
) / 5.0  # Normalised to [0, 1]

# Similarly: art_lat, biz_lat, res_lat
```

### Stage 2: Feature Computation

Academic, personality, skill, and interest features are derived from quiz answers:

```python
# Academic features
academic_base = [17, 14, 11, 8][a[8]]       # From Q9 (performance)
math_score    = [9, 7, 5, 3][a[3]]           # From Q4 (math comfort)
self_learning = [1.0, 0.8, 0.5, 0.2][a[9]]  # From Q10 (self-study)

grade1      = clip(round(academic_base + (math_score - 6) * 0.6), 0, 20)
final_grade = clip(round((grade1 + grade2) / 2 + self_learning * 1.5), 0, 20)

# Big Five Personality
openness          = clip(0.35 + art_lat * 0.35 + res_lat * 0.2, 0, 1)
conscientiousness = clip(0.3 + self_learning * 0.4 + deadline_ctrl * 0.3, 0, 1)
extraversion      = clip(0.2 + (3 - a[5]) * 0.18 + (1 if a[2]==2 else 0) * 0.15, 0, 1)

# Skills
coding_skill     = int(clip(round(2 + tech_lat * 7), 0, 9))
analytical_skill = int(clip(round(2 + (1 if a[0]==0 else 0)*3 + (math_score/10)*4), 0, 9))

# Interest scores
tech_interest     = int(clip(round(tech_lat * 9), 0, 9))
art_interest      = int(clip(round(art_lat * 9), 0, 9))
business_interest = int(clip(round(biz_lat * 9), 0, 9))
```

### Stage 3: Dominant-Intent Calibration

The strongest domain signal is boosted above model decision thresholds:

```python
dominant = max(
    [("tech", tech_lat), ("art", art_lat), ("biz", biz_lat), ("res", res_lat)],
    key=lambda x: x[1],
)[0]

if dominant == "tech":
    coding_skill  = max(coding_skill, 8)
    tech_interest = max(tech_interest, 8)
elif dominant == "art":
    art_interest = max(art_interest, 8)
elif dominant == "biz":
    business_interest = max(business_interest, 8)
else:  # research
    analytical_skill = max(analytical_skill, 8)
    final_grade      = max(final_grade, 13)
```

**Purpose:** The training labels use rule-based thresholds (e.g., `coding_skill > 7`). Without calibration, ambiguous quiz profiles would default to the majority class (Research Scientist). This step ensures distinct quiz patterns yield distinct career predictions.

---

## A.3 ML Inference (`predict`)

```python
def predict(model, feature_vector, label_encoder=None):
    raw   = model.predict(feature_vector)
    top_c = _decode_career_label(raw[0], label_encoder)
    top3  = []
    conf  = None

    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(feature_vector)[0]
        idx_top = np.argsort(probs)[::-1][:3]
        for idx in idx_top:
            name = _decode_career_label(model.classes_[idx], label_encoder)
            top3.append((name, round(float(probs[idx]) * 100, 1)))
        if top3:
            conf = top3[0][1]

    return top_c, conf, top3
```

**Logic:** Returns the top career, its confidence percentage, and the top-3 ranked careers. Gracefully handles models without `predict_proba`.

---

## A.4 Resume Scoring Algorithm (`analyze_resume`)

```python
def analyze_resume(text, career):
    keywords   = RESUME_KEYWORDS.get(career.lower(), RESUME_KEYWORDS["default"])
    text_lower = text.lower()

    matched = [k for k in keywords if k.lower() in text_lower]
    missing = [k for k in keywords if k.lower() not in text_lower]

    # Keyword score (max 49 points)
    kw_score = (len(matched) / max(len(keywords), 1)) * 49

    # Structural bonus signals (max 51 points)
    bonus = sum([
        ("github" in text_lower)  * 9,          # GitHub link
        bool(re.search(r"\d+\s*%|...", text)) * 11,  # Quantified impact
        has_action_verbs * 8,                    # Action verbs
        has_education    * 5,                    # Degree info
        has_contact_info * 4,                    # Contact details
        14 if (180 < word_count < 750) else 3,   # Optimal length
    ])

    score = max(0, min(100, int(kw_score + bonus)))
    grade = "A" if score >= 80 else ("B" if score >= 60 else
            ("C" if score >= 40 else "D"))
```

**Scoring Breakdown:**

| Component | Points | Check |
|---|---|---|
| Keyword match | 0–49 | `matched / total × 49` |
| GitHub link | +9 | `"github" in text` |
| Quantified impact | +11 | Regex: `%`, `x`, `$`, `₹` |
| Action verbs | +8 | built, deployed, led, etc. |
| Education | +5 | B.Tech, BSc, CGPA, etc. |
| Contact info | +4 | email, phone, LinkedIn |
| Optimal length | +14/+3 | 180–750 words |
| **Total** | **0–100** | |

---

## A.5 Chatbot Intent Detection (`chatbot_response`)

```python
INTENT_MAP = {
    "skills":    ["skill", "learn", "language", "tool", "technology"],
    "start":     ["start", "begin", "roadmap", "first step"],
    "projects":  ["project", "portfolio", "build", "create"],
    "interview": ["interview", "prepare", "crack", "job"],
    "salary":    ["salary", "pay", "earn", "package", "ctc"],
    # ... 10 intent categories total
}

def detect_intent(msg):
    msg_l = msg.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(k in msg_l for k in keywords):
            return intent
    return "unknown"

def chatbot_response(user_msg, career):
    intent     = detect_intent(user_msg)
    kb_career  = CHATBOT_KB.get(career.lower(), {})
    kb_general = CHATBOT_KB["_general"]

    if intent != "unknown" and intent in kb_career:
        return kb_career[intent]          # Career-specific response
    if intent != "unknown" and intent in kb_general:
        return kb_general[intent]          # General response
    return f"I can help with: skills, getting started, projects..."
```

**Logic:** Two-tier lookup — career-specific knowledge base first, general fallback second.

---

## A.6 Model Training Pipeline (LightGBM + Optuna + SHAP)

```python
# 1. Load and inject noise
df = pd.read_csv("final_career_dataset.csv")
noise_indices = np.random.choice(df.index, size=int(len(df)*0.10), replace=False)
df.loc[noise_indices, 'career'] = np.random.choice(df['career'].unique(), size=n_noise)

# 2. Encode and split
le = LabelEncoder()
df['career'] = le.fit_transform(df['career'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# 3. SMOTE balancing
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. SHAP feature selection
baseline = lgb.LGBMClassifier(random_state=42, verbose=-1)
baseline.fit(X_train_res, y_train_res)
explainer = shap.TreeExplainer(baseline)
shap_values = explainer.shap_values(X_train_res.sample(1000, random_state=42))
mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
selected_features = X.columns[np.argsort(mean_abs_shap)[::-1][:15]].tolist()

# 5. Optuna hyperparameter tuning
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves':    trial.suggest_int('num_leaves', 20, 100),
        'max_depth':     trial.suggest_int('max_depth', 3, 15),
        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
        # ... additional parameters
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_selected, y_train_res)
    y_pred_proba = model.predict_proba(X_test_selected)
    return roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)

# 6. Final model with best parameters
final_model = lgb.LGBMClassifier(**study.best_params)
final_model.fit(X_train_selected, y_train_res)
```

---

## A.7 Career Label Assignment (Training Data)

The rule-based function used during training to assign career labels:

```python
def assign_career(row):
    if row['coding_skill'] > 7 and row['tech_interest'] > 6:
        return 'Software Developer'
    elif row['analytical_skill'] > 7 and row['final_grade'] > 12:
        return 'Data Scientist'
    elif row['art_interest'] > 7:
        return 'UI/UX Designer'
    elif row['business_interest'] > 7:
        return 'Entrepreneur'
    else:
        return 'Research Scientist'
```

**Note:** This deterministic mapping creates clear decision boundaries. The 10% noise injection partially disrupts these boundaries to simulate real-world label ambiguity.

---

## A.8 Flask REST API Endpoint

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    num_features = ['grade1', 'grade2', 'final_grade', 'study_time',
                    'failures', 'absences', 'openness', 'conscientiousness',
                    'extraversion', 'agreeableness', 'neuroticism',
                    'coding_skill', 'communication_skill', 'analytical_skill',
                    'study_hours', 'consistency', 'participation',
                    'tech_interest', 'art_interest', 'business_interest',
                    'family_income']

    features_list = [float(data.get(feat, 0.0)) for feat in num_features]

    # One-hot encode internet_access
    internet = str(data.get('internet_access', 'yes')).lower()
    features_list.extend([1.0, 0.0] if internet == 'no' else [0.0, 1.0])

    prediction = model.predict(np.array([features_list]))
    return jsonify({"status": "success", "career": str(prediction[0])})
```

**Purpose:** Provides a lightweight `/predict` REST endpoint accepting a 23-feature JSON payload and returning the predicted career. Serves as an alternative to the Streamlit UI.
