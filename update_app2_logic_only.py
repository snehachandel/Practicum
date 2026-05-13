import re

with open('app2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update model and feature loading
loading_logic = """
@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    path = os.path.join(BASE_DIR, "career_model_7.pkl")
    if not os.path.exists(path):
        return None, "career_model_7.pkl not found — place it in the same folder as app2.py"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"

@st.cache_resource(show_spinner=False)
def load_features() -> tuple:
    path = os.path.join(BASE_DIR, "features_7.pkl")
    if not os.path.exists(path):
        return None, "features_7.pkl not found"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load features: {exc}"

@st.cache_resource(show_spinner=False)
def load_label_encoder() -> tuple:
    path = os.path.join(BASE_DIR, "label_encoder_7.pkl")
    if not os.path.exists(path):
        return None, "label_encoder_7.pkl not found"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load label encoder: {exc}"
"""

content = re.sub(
    r'@st\.cache_resource\(show_spinner=False\).*?def load_label_encoder\(\) -> tuple:.*?return None, f"Failed to load label encoder: \{exc\}"',
    loading_logic.strip(),
    content,
    flags=re.DOTALL
)

# 2. Update quiz_to_features
quiz_func = """
def quiz_to_features(answers: list[int], model, features) -> pd.DataFrame:
    a = answers
    def clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    tech_lat = ((1.0 if a[1] == 0 else 0.0) + (1.0 if a[4] == 0 else 0.0) + (1.0 if a[11] == 0 else 0.0) + (1.0 if a[13] == 0 else 0.0) + (1.0 if a[14] == 0 else 0.0)) / 5.0
    art_lat = ((1.0 if a[1] == 1 else 0.0) + (1.0 if a[4] == 1 else 0.0) + (1.0 if a[11] == 1 else 0.0) + (1.0 if a[13] == 1 else 0.0) + (1.0 if a[14] == 1 else 0.0)) / 5.0
    biz_lat = ((1.0 if a[1] == 2 else 0.0) + (1.0 if a[4] == 2 else 0.0) + (1.0 if a[11] == 2 else 0.0) + (1.0 if a[13] == 2 else 0.0) + (1.0 if a[14] == 2 else 0.0)) / 5.0
    res_lat = ((1.0 if a[1] == 3 else 0.0) + (1.0 if a[4] == 3 else 0.0) + (1.0 if a[11] == 3 else 0.0) + (1.0 if a[13] == 3 else 0.0) + (1.0 if a[14] == 3 else 0.0)) / 5.0

    academic_base = [17, 14, 11, 8][a[8]]
    math_score    = [9, 7, 5, 3][a[3]]
    self_learning = [1.0, 0.8, 0.5, 0.2][a[9]]
    deadline_ctrl = [1.0, 0.8, 0.5, 0.2][a[6]]

    grade1      = clip(round(academic_base + (math_score - 6) * 0.6), 0, 20)
    grade2      = clip(round(academic_base + (deadline_ctrl - 0.6) * 3), 0, 20)
    final_grade = clip(round((grade1 + grade2) / 2 + self_learning * 1.5), 0, 20)

    study_time = int(clip(round(1 + self_learning * 3), 1, 4))
    failures   = int(clip(round((20 - final_grade) / 6), 0, 3))
    absences   = int(clip(round(20 - (academic_base * 0.8) + (3 - a[6]) * 2), 0, 30))

    openness          = clip(0.35 + art_lat * 0.35 + res_lat * 0.2, 0, 1)
    conscientiousness = clip(0.3 + self_learning * 0.4 + deadline_ctrl * 0.3, 0, 1)
    extraversion      = clip(0.2 + (3 - a[5]) * 0.18 + (1 if a[2] == 2 else 0) * 0.15, 0, 1)
    agreeableness     = clip(0.35 + (1 if a[2] in [1, 2] else 0) * 0.25 + (1 if a[12] == 2 else 0) * 0.2, 0, 1)
    neuroticism       = clip(0.2 + (a[6] / 3) * 0.6, 0, 1)

    coding_skill        = int(clip(round(2 + tech_lat * 7), 0, 9))
    communication_skill = int(clip(round(2 + (extraversion * 3.5) + (1 if a[2] == 2 else 0) * 2), 0, 9))
    analytical_skill    = int(clip(round(2 + (1 if a[0] == 0 else 0) * 3 + (math_score / 10) * 4), 0, 9))

    study_hours   = int(clip(round(3 + self_learning * 6), 1, 10))
    consistency   = clip(0.25 + self_learning * 0.4 + deadline_ctrl * 0.25, 0, 1)
    participation = clip(0.2 + (1 if a[2] in [1, 2] else 0) * 0.25 + (extraversion * 0.35), 0, 1)

    tech_interest     = int(clip(round(tech_lat * 9), 0, 9))
    art_interest      = int(clip(round(art_lat * 9), 0, 9))
    business_interest = int(clip(round(biz_lat * 9), 0, 9))

    dominant = max([("tech", tech_lat), ("art", art_lat), ("biz", biz_lat), ("res", res_lat)], key=lambda x: x[1])[0]
    
    if dominant == "tech":
        coding_skill  = max(coding_skill, 8)
        tech_interest = max(tech_interest, 8)
    elif dominant == "art":
        art_interest = max(art_interest, 8)
    elif dominant == "biz":
        business_interest = max(business_interest, 8)
    else:
        analytical_skill = max(analytical_skill, 8)
        final_grade      = max(final_grade, 13)

    family_income = int(clip(round(2 + (0.5 if a[14] in [0, 1] else -0.2)), 1, 4))
    has_internet  = 1 if (self_learning >= 0.5 or a[10] in [0, 1, 2]) else 0

    base_row = {
        "grade1": grade1, "grade2": grade2, "final_grade": final_grade, "study_time": study_time,
        "failures": failures, "absences": absences, "openness": openness, "conscientiousness": conscientiousness,
        "extraversion": extraversion, "agreeableness": agreeableness, "neuroticism": neuroticism,
        "coding_skill": coding_skill, "communication_skill": communication_skill, "analytical_skill": analytical_skill,
        "study_hours": study_hours, "consistency": consistency, "participation": participation,
        "tech_interest": tech_interest, "art_interest": art_interest, "business_interest": business_interest,
        "family_income": family_income, "internet_access": float(has_internet)
    }

    if not features:
        return pd.DataFrame([base_row])

    row = {col: base_row.get(col, 0.0) for col in features}
    return pd.DataFrame([row], columns=features)
"""

content = re.sub(
    r'def quiz_to_features.*?return pd\.DataFrame\(\[row\], columns=expected_cols\)',
    quiz_func.strip(),
    content,
    flags=re.DOTALL
)

# 3. Fix page_quiz function call
content = content.replace("def page_quiz(model, label_encoder) -> None:", "def page_quiz(model, label_encoder, features) -> None:")
content = content.replace("feat = quiz_to_features(st.session_state.answers, model)", "feat = quiz_to_features(st.session_state.answers, model, features)")

# 4. Fix main() function call
new_main = """
    # Load career_model_7.pkl once (cached)
    model, err = load_model()
    features, feat_err = load_features()
    label_encoder, le_err = load_label_encoder()
    
    if err and st.session_state.page != "home":
        st.warning(f"⚠️ {err}  —  Running in demo mode; predictions are illustrative.")
    elif le_err and st.session_state.page != "home":
        st.warning(f"⚠️ {le_err}.")

    # Normalize older session values that may still contain encoded class IDs.
    if st.session_state.get("career"):
        st.session_state.career = _decode_career_label(st.session_state.career, label_encoder)

    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "quiz":
        page_quiz(model, label_encoder, features)
    elif st.session_state.page == "results": page_results()
    elif st.session_state.page == "resume":  page_resume()
    elif st.session_state.page == "chat":    page_chat()
"""

# app.py still has `elif page == "results": page_results()`
# we will replace the block that uses `page = st.session_state.page`
content = re.sub(
    r'# Load career_model.pkl once \(cached\).*?page_quiz\(model, label_encoder\)\n    elif page == "results": page_results\(\)\n    elif page == "resume":  page_resume\(\)\n    elif page == "chat":    page_chat\(\)',
    new_main.strip(),
    content,
    flags=re.DOTALL
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated app2.py successfully (logic only).")
