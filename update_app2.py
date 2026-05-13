import re

with open('app2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update CSS
new_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg-deep:      #050505;
  --bg-space:     #0A0A0A;
  --glass-base:   rgba(20, 20, 20, 0.7);
  --glass-border: rgba(255, 255, 255, 0.05);
  
  --accent:       #6366F1;
  --accent-hov:   #818CF8;
  --accent-sub:   rgba(99, 102, 241, 0.1);
  
  --cyan:         #38BDF8;
  --green:        #34D399;
  --purple:       #A78BFA;
  --rose:         #FB7185;
  --amber:        #FBBF24;
  
  --snow:         #F8FAFC;
  --muted:        #94A3B8;
  --dim:          #475569;
  
  --grad-neon:    linear-gradient(135deg, var(--accent), var(--cyan));
  
  --display:      'Outfit', sans-serif;
  --sans:         'Inter', sans-serif;
  --mono:         'Inter', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-deep) !important;
  font-family: var(--sans);
  color: var(--snow);
}

[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08), transparent 50%), var(--bg-deep) !important;
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1000px !important; }
h1, h2, h3 { font-family: var(--display) !important; font-weight: 600 !important; color: var(--snow) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg-space) !important;
  border-right: 1px solid var(--glass-border) !important;
}

.nav-brand { padding: 1.8rem 1.2rem 1rem; border-bottom: 1px solid var(--glass-border); margin-bottom: 0.8rem; }
.nav-logo { display: flex; align-items: center; gap: 12px; font-family: var(--display); font-size: 1.25rem; font-weight: 600; color: var(--snow); margin-bottom: 4px; }
.nav-logo-icon { width: 32px; height: 32px; border-radius: 8px; background: var(--grad-neon); display: flex; align-items: center; justify-content: center; font-size: 16px; box-shadow: 0 4px 12px rgba(99,102,241,0.3); }
.nav-tagline  { font-size: 12px; color: var(--muted); padding-left: 44px; line-height: 1.5; }
.nav-slabel   { font-family: var(--sans); font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--dim); padding: 0 1.2rem; margin: 1.2rem 0 8px; letter-spacing: 1px; }
.nav-div      { height: 1px; background: var(--glass-border); margin: 1rem 0.8rem; }
.nav-status   { margin: 0.8rem; padding: 14px; background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); border-radius: 12px; }
.nav-s-career { font-family: var(--display); font-size: 14px; font-weight: 600; color: var(--accent); margin-bottom: 4px; }
.nav-s-label  { font-family: var(--sans); font-size: 11px; color: var(--muted); }
.nav-footer   { margin-top: 2rem; padding-bottom: 1.2rem; text-align: center; font-size: 11px; color: var(--dim); }

/* Premium Cards */
.card {
  background: var(--glass-base);
  border: 1px solid var(--glass-border);
  border-radius: 20px; 
  padding: 2.2rem; 
  margin-bottom: 1.5rem;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.card:hover {
  transform: translateY(-4px);
  border-color: rgba(255,255,255,0.1);
  box-shadow: 0 12px 24px rgba(0,0,0,0.2), 0 0 0 1px rgba(255,255,255,0.05);
}

/* Modifiers */
.card-pu { border-top: 2px solid var(--purple) !important; }
.card-bl { border-top: 2px solid var(--cyan) !important; }
.card-cy { border-top: 2px solid var(--green) !important; }
.card-gr { border-top: 2px solid var(--amber) !important; }

/* Hero */
.hero { text-align: center; padding: 5rem 1rem 4rem; position: relative; }
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--sans); font-size: 12px; font-weight: 500;
  color: var(--accent); background: var(--accent-sub); 
  border-radius: 100px; padding: 6px 16px; margin-bottom: 2rem;
}
.hero-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.hero-title { font-family: var(--display); font-size: clamp(2.5rem, 6vw, 4.5rem); font-weight: 700; line-height: 1.1; color: var(--snow); margin-bottom: 1.5rem; letter-spacing: -1px; }
.hero-grad-text { background: var(--grad-neon); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: 1.1rem; color: var(--muted); font-weight: 400; max-width: 600px; margin: 0 auto 3rem; line-height: 1.6; }
.hero-pills { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
.hero-pill  { display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.03); border: 1px solid var(--glass-border); border-radius: 100px; padding: 8px 18px; font-size: 13px; color: var(--snow); font-weight: 500; transition: all 0.2s; }
.hero-pill:hover { background: rgba(255,255,255,0.08); transform: translateY(-2px); }

/* Section labels */
.s-eye   { font-family: var(--sans); font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: .5rem; letter-spacing: 1px; }
.s-title { font-family: var(--display); font-size: 2rem; font-weight: 600; color: var(--snow); margin-bottom: .6rem; letter-spacing: -0.5px; }
.s-sub   { font-size: 15px; color: var(--muted); margin-bottom: 2rem; line-height: 1.6; }

/* Quiz */
.quiz-hdr { display: flex; align-items: center; justify-content: space-between; padding: 1.2rem 1.8rem; background: var(--glass-base); border: 1px solid var(--glass-border); border-radius: 16px; margin-bottom: 1.5rem; }
.quiz-prog-label { font-size: 11px; font-weight: 500; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
.quiz-prog-bg    { width: 260px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.quiz-prog-fill  { height: 100%; background: var(--accent); border-radius: 100px; transition: width 0.4s ease; }
.quiz-step-badge { font-size: 12px; font-weight: 600; color: var(--accent); background: var(--accent-sub); border-radius: 100px; padding: 6px 16px; }
.quiz-q-num  { font-size: 12px; font-weight: 500; color: var(--accent); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px; }
.quiz-q-text { font-family: var(--display); font-size: 1.5rem; font-weight: 500; color: var(--snow); margin-bottom: 2rem; line-height: 1.4; }

/* Buttons */
.stButton > button {
  font-family: var(--sans) !important; font-weight: 500 !important;
  font-size: 15px !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important;
  background: rgba(255,255,255,0.03) !important; color: var(--snow) !important;
  padding: 0.8rem 1.5rem !important; transition: all 0.2s ease !important;
}
.stButton > button:hover { background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.2) !important; transform: translateY(-2px) !important; }
div[data-testid="stVerticalBlock"] > div > div > div > div > button:first-child { background: var(--accent) !important; border-color: var(--accent) !important; }
div[data-testid="stVerticalBlock"] > div > div > div > div > button:first-child:hover { background: var(--accent-hov) !important; border-color: var(--accent-hov) !important; }

/* Result */
.result-badge { display: inline-flex; align-items: center; font-size: 12px; font-weight: 600; color: var(--accent); background: var(--accent-sub); border-radius: 100px; padding: 6px 16px; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1px; }
.result-icon   { font-size: 5rem; margin-bottom: 1rem; }
.result-career { font-family: var(--display); font-size: clamp(2rem,5vw,3.5rem); font-weight: 600; color: var(--snow); margin-bottom: .5rem; }
.result-conf   { font-size: 15px; font-weight: 500; color: var(--cyan); margin-bottom: 1.2rem; }
.result-expl   { font-size: 15px; color: var(--muted); max-width: 520px; margin: 0 auto; line-height: 1.7; }

/* Top 3 */
.t3-row  { background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1.2rem 1.5rem; display: flex; align-items: center; margin-bottom: 1rem; transition: all 0.2s ease; }
.t3-row:hover { background: rgba(255,255,255,0.05); transform: translateX(6px); border-color: rgba(255,255,255,0.1); }
.t3-rank { font-size: 14px; font-weight: 600; color: var(--muted); width: 28px; }
.t3-name { font-family: var(--display); font-size: 16px; font-weight: 500; color: var(--snow); flex: 1; margin-left: 12px; }
.t3-bar-bg   { width: 100px; height: 6px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.t3-bar-fill { height: 100%; border-radius: 100px; background: var(--accent); }
.t3-pct  { font-size: 14px; font-weight: 600; color: var(--snow); margin-left: 15px; min-width: 45px; text-align: right; }
.t3-icon { font-size: 20px; margin-left: 12px; }

/* Roadmap */
.rm-step   { display: flex; gap: 1.5rem; margin-bottom: 2rem; }
.rm-left   { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.rm-circle { width: 44px; height: 44px; border-radius: 50%; background: rgba(255,255,255,0.03); display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; color: var(--muted); border: 1px solid var(--glass-border); transition: all 0.3s; }
.rm-step:hover .rm-circle { background: var(--accent); color: white; border-color: var(--accent); }
.rm-line   { width: 2px; flex: 1; margin: 8px 0; min-height: 30px; background: rgba(255,255,255,0.05); }
.rm-step:hover .rm-line { background: var(--accent); opacity: 0.3; }
.rm-body   { padding-top: 6px; flex: 1; }
.rm-tag    { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; letter-spacing: 1px; }
.rm-title  { font-family: var(--display); font-size: 1.25rem; font-weight: 600; color: var(--snow); margin-bottom: 8px; }
.rm-desc   { font-size: 14px; color: var(--muted); line-height: 1.6; margin-bottom: 1rem; }
.sk-tag    { display: inline-block; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 4px 12px; font-size: 12px; color: var(--muted); margin: 3px 4px 3px 0; }

/* Chat */
.chat-b    { max-width: 85%; margin-bottom: 1.5rem; }
.chat-b.u  { margin-left: auto; }
.chat-meta { font-size: 11px; color: var(--muted); margin-bottom: 6px; padding: 0 4px; }
.chat-b.u .chat-meta { text-align: right; }
.chat-txt  { padding: 1.2rem 1.5rem; border-radius: 16px; font-size: 15px; line-height: 1.6; }
.chat-b.u .chat-txt { background: rgba(255,255,255,0.05); color: var(--snow); border-bottom-right-radius: 4px; border: 1px solid rgba(255,255,255,0.05); }
.chat-b.bot .chat-txt { background: var(--glass-base); color: var(--snow); border-bottom-left-radius: 4px; border: 1px solid var(--glass-border); }

/* Inputs */
.stTextArea textarea, .stTextInput input {
  background: rgba(255,255,255,0.02) !important; border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important; color: var(--snow) !important; font-size: 15px !important; padding: 1rem !important; transition: all 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--accent) !important; background: rgba(255,255,255,0.04) !important; outline: none !important; box-shadow: 0 0 0 2px var(--accent-sub) !important; }

/* Chips */
.p-chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 100px; font-size: 13px; font-weight: 500; margin: 4px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); color: var(--snow); }
.p-chip:hover { background: rgba(255,255,255,0.08); }

/* Conf Ring */
.conf-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem 0; }
.conf-outer { width: 140px; height: 140px; border-radius: 50%; display: flex; align-items: center; justify-content: center; position: relative; }
.conf-svg   { position: absolute; inset: 0; transform: rotate(-90deg); transition: stroke-dasharray 1s ease-out; }
.conf-inner { display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }
.conf-num   { font-family: var(--display); font-size: 3rem; font-weight: 600; color: var(--snow); line-height: 1; }
.conf-unit  { font-size: 12px; font-weight: 500; color: var(--muted); margin-top: 4px; }

/* Misc */
.hdiv { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 3rem 0; }
[data-testid="stAlert"] { border-radius: 12px !important; background: rgba(255,255,255,0.02) !important; border: 1px solid var(--glass-border) !important; color: var(--snow) !important; }

</style>
"""

content = re.sub(r'<style>.*?</style>', new_css, content, flags=re.DOTALL)


# 2. Update model and feature loading
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

# 3. Update quiz_to_features
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

# 4. Fix page_quiz function call
content = content.replace("def page_quiz(model, label_encoder) -> None:", "def page_quiz(model, label_encoder, features) -> None:")
content = content.replace("feat = quiz_to_features(st.session_state.answers, model)", "feat = quiz_to_features(st.session_state.answers, model, features)")

# 5. Fix main() function call
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
"""

content = re.sub(
    r'# Load career_model.pkl once \(cached\).*?page_quiz\(model, label_encoder\)',
    new_main.strip(),
    content,
    flags=re.DOTALL
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated app2.py successfully.")
