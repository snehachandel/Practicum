"""
╔══════════════════════════════════════════════════════════════════════╗
║   NextStep AI — Your Smart Career Guide to the Next Step            ║
║   Quiz → ML Prediction → Roadmap → Mentor → Resume Analyzer         ║
║   Built with Streamlit · scikit-learn · Zero External APIs           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, time, pickle
import numpy as np
import streamlit as st

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextStep AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Dark navy/purple glassmorphism theme
# ══════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Satoshi:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

/* ── Reset & tokens ── */
*, *::before, *::after { box-sizing: border-box; }
:root {
  --navy:      #05070F;
  --navy2:     #080C18;
  --navy3:     #0E1428;
  --navy4:     #141D35;
  --glass:     rgba(255,255,255,0.04);
  --glass-hi:  rgba(255,255,255,0.08);
  --border:    rgba(255,255,255,0.06);
  --border-hi: rgba(139,92,246,0.40);
  --border-b:  rgba(59,130,246,0.30);
  --purple:    #8B5CF6;
  --purple2:   #A78BFA;
  --blue:      #3B82F6;
  --blue2:     #60A5FA;
  --cyan:      #22D3EE;
  --green:     #10B981;
  --rose:      #F43F5E;
  --amber:     #F59E0B;
  --snow:      #F0F4FF;
  --muted:     #6B7A99;
  --dim:       #2D3A55;
  --serif:     'DM Serif Display', Georgia, serif;
  --sans:      'Plus Jakarta Sans', sans-serif;
  --mono:      'Fira Code', monospace;
  --grad1:     linear-gradient(135deg,#8B5CF6,#3B82F6);
  --grad2:     linear-gradient(135deg,#3B82F6,#22D3EE);
  --grad3:     linear-gradient(135deg,#8B5CF6 0%,#EC4899 100%);
}

/* ── Base ── */
html,body,[data-testid="stAppViewContainer"] {
  background: var(--navy) !important;
  font-family: var(--sans);
  color: var(--snow);
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 100% 60% at 50% -10%, rgba(139,92,246,0.18) 0%, transparent 55%),
    radial-gradient(ellipse 60%  50% at 95%  80%, rgba(59,130,246,0.10) 0%, transparent 50%),
    radial-gradient(ellipse 50%  40% at 5%   90%, rgba(34,211,238,0.06) 0%, transparent 45%),
    var(--navy) !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 980px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--navy2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── NAV BRAND ── */
.nav-brand {
  padding: 1.8rem 1.2rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.8rem;
}
.nav-logo {
  display: flex; align-items: center; gap: 10px;
  font-family: var(--sans); font-size: 1.15rem; font-weight: 800;
  letter-spacing: -0.3px; color: var(--snow); margin-bottom: 4px;
}
.nav-logo-icon {
  width: 32px; height: 32px; border-radius: 10px;
  background: var(--grad1);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
}
.nav-tagline {
  font-size: 11px; color: var(--muted); padding-left: 42px;
  font-weight: 400; letter-spacing: 0.2px;
}
.nav-section-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--dim);
  padding: 0 1.2rem; margin: 1rem 0 6px;
}
.nav-divider { height: 1px; background: var(--border); margin: 0.8rem 0.8rem; }
.nav-status {
  margin: 0.8rem; padding: 12px 14px;
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 14px;
}
.nav-status-career {
  font-size: 13px; font-weight: 600; color: var(--snow); margin-bottom: 2px;
}
.nav-status-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--muted);
}
.nav-footer {
  position: absolute; bottom: 1.2rem; left: 0; right: 0;
  text-align: center; font-family: var(--mono); font-size: 9px;
  letter-spacing: 2px; color: var(--dim); text-transform: uppercase;
}

/* ── Button styles ── */
.stButton > button {
  font-family: var(--sans) !important; font-weight: 600 !important;
  font-size: 14px !important; border-radius: 12px !important;
  border: none !important; cursor: pointer !important;
  transition: all 0.22s cubic-bezier(.175,.885,.32,1.275) !important;
  letter-spacing: 0.3px !important;
  background: var(--grad1) !important;
  color: #fff !important;
  box-shadow: 0 4px 20px rgba(139,92,246,0.30) !important;
  padding: 0.75rem 1.5rem !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(139,92,246,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Glass cards ── */
.card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.8rem 2rem;
  margin-bottom: 1.4rem;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(12px);
  transition: border-color 0.3s, transform 0.2s;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
}
.card:hover { border-color: rgba(139,92,246,0.25); }
.card-purple { border-color: rgba(139,92,246,0.28) !important; background: linear-gradient(135deg,rgba(139,92,246,0.08),var(--glass)); }
.card-blue   { border-color: rgba(59,130,246,0.28)  !important; background: linear-gradient(135deg,rgba(59,130,246,0.08),var(--glass)); }
.card-cyan   { border-color: rgba(34,211,238,0.28)  !important; background: linear-gradient(135deg,rgba(34,211,238,0.06),var(--glass)); }
.card-green  { border-color: rgba(16,185,129,0.28)  !important; background: linear-gradient(135deg,rgba(16,185,129,0.06),var(--glass)); }
.card-rose   { border-color: rgba(244,63,94,0.28)   !important; background: linear-gradient(135deg,rgba(244,63,94,0.06),var(--glass)); }

/* ── HERO ── */
.hero {
  text-align: center;
  padding: 5.5rem 1rem 3.5rem;
  position: relative;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%238B5CF6' fill-opacity='0.025'%3E%3Cpath d='M50 50c0-5.523 4.477-10 10-10s10 4.477 10 10-4.477 10-10 10c0 5.523-4.477 10-10 10s-10-4.477-10-10 4.477-10 10-10zM10 10c0-5.523 4.477-10 10-10s10 4.477 10 10-4.477 10-10 10c0 5.523-4.477 10-10 10S0 25.523 0 20s4.477-10 10-10z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 3px; text-transform: uppercase;
  color: var(--purple2);
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.28);
  border-radius: 100px; padding: 6px 16px;
  margin-bottom: 1.6rem;
  animation: fadeDown 0.55s ease both;
}
.hero-eyebrow .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--purple); animation: pulse 2s infinite;
}
.hero-title {
  font-family: var(--serif);
  font-size: clamp(3rem, 7vw, 5rem);
  font-weight: 400; line-height: 1.06;
  letter-spacing: -1.5px; color: var(--snow);
  margin-bottom: 1.2rem;
  animation: fadeDown 0.55s 0.1s ease both;
}
.hero-title .gradient-text {
  background: var(--grad1);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-title em { font-style: italic; color: var(--purple2); }
.hero-sub {
  font-size: 1.1rem; color: var(--muted); font-weight: 400;
  max-width: 500px; margin: 0 auto 2.8rem; line-height: 1.78;
  animation: fadeDown 0.55s 0.2s ease both;
}
.hero-cta {
  display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
  animation: fadeDown 0.55s 0.3s ease both; margin-bottom: 3.5rem;
}
.btn-primary {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--grad1); color: #fff;
  padding: 14px 28px; border-radius: 14px; font-size: 15px;
  font-weight: 700; font-family: var(--sans);
  box-shadow: 0 4px 24px rgba(139,92,246,0.4);
  transition: all 0.22s; cursor: pointer; border: none;
  letter-spacing: 0.2px; text-decoration: none;
}
.btn-secondary {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--glass); color: var(--snow);
  padding: 14px 28px; border-radius: 14px; font-size: 15px;
  font-weight: 600; font-family: var(--sans);
  border: 1px solid var(--border); cursor: pointer;
  transition: all 0.22s; text-decoration: none;
}
.hero-features {
  display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;
  animation: fadeDown 0.55s 0.35s ease both;
}
.hero-feat {
  display: flex; align-items: center; gap: 8px;
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 100px; padding: 8px 18px;
  font-size: 12px; color: var(--muted); font-weight: 500;
}
.hero-feat .feat-icon { font-size: 14px; }

/* ── Section labels ── */
.section-eyebrow {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 3px; text-transform: uppercase;
  color: var(--purple2); margin-bottom: 0.5rem;
}
.section-title {
  font-family: var(--serif); font-size: 2.2rem;
  color: var(--snow); margin-bottom: 0.6rem; line-height: 1.15;
}
.section-sub {
  font-size: 14px; color: var(--muted);
  margin-bottom: 2rem; line-height: 1.7;
}

/* ── Feature grid ── */
.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.4rem; }
.feat-item {
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.4rem 1.5rem;
  transition: border-color 0.25s;
}
.feat-item:hover { border-color: rgba(139,92,246,0.3); }
.feat-item-icon { font-size: 1.8rem; margin-bottom: 0.8rem; }
.feat-item-title { font-size: 14px; font-weight: 700; color: var(--snow); margin-bottom: 4px; }
.feat-item-desc  { font-size: 12px; color: var(--muted); line-height: 1.6; }

/* ── Quiz ── */
.quiz-header {
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 16px; padding: 1rem 1.5rem;
  margin-bottom: 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
}
.quiz-header-left { display: flex; flex-direction: column; }
.quiz-prog-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.quiz-prog-bar-bg {
  width: 240px; height: 3px; background: var(--navy3);
  border-radius: 100px; overflow: hidden;
}
.quiz-prog-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple), var(--blue));
  border-radius: 100px; transition: width 0.5s ease;
}
.quiz-step-badge {
  font-family: var(--mono); font-size: 12px; font-weight: 500;
  color: var(--purple2); background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.25);
  border-radius: 100px; padding: 4px 14px;
}
.quiz-q-num {
  font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--purple2); margin-bottom: 0.6rem;
}
.quiz-q-text {
  font-family: var(--serif); font-size: 1.6rem;
  color: var(--snow); margin-bottom: 2rem; line-height: 1.35;
}
.quiz-opt-key {
  width: 26px; height: 26px; border-radius: 8px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10px; color: var(--dim);
  flex-shrink: 0; transition: all 0.2s;
}

/* ── Results ── */
.result-hero {
  text-align: center; padding: 2.8rem 1rem 2rem;
  position: relative;
}
.result-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--amber);
  background: rgba(245,158,11,0.10);
  border: 1px solid rgba(245,158,11,0.28);
  border-radius: 100px; padding: 6px 16px; margin-bottom: 1.2rem;
}
.result-icon   { font-size: 5.5rem; line-height: 1; margin-bottom: 0.5rem; animation: bounceIn 0.6s ease both; }
.result-career { font-family: var(--serif); font-size: clamp(2.2rem,5vw,3.5rem); color: var(--snow); margin-bottom: 0.5rem; letter-spacing: -0.5px; }
.result-conf   { font-family: var(--mono); font-size: 13px; color: var(--green); margin-bottom: 1rem; }
.result-expl   { font-size: 15px; color: var(--muted); max-width: 480px; margin: 0 auto; line-height: 1.75; }

/* ── Confidence ring ── */
.conf-ring-wrap {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 1.5rem 0;
}
.conf-ring-outer {
  width: 130px; height: 130px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  position: relative;
}
.conf-ring-svg { position: absolute; inset: 0; transform: rotate(-90deg); }
.conf-ring-inner {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center;
}
.conf-num   { font-family: var(--serif); font-size: 2.4rem; color: var(--snow); line-height: 1; }
.conf-unit  { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; }

/* ── Personality chips ── */
.personality-chip {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 16px; border-radius: 100px;
  font-family: var(--mono); font-size: 11px; letter-spacing: 1px;
  font-weight: 500; margin: 3px;
}
.chip-analytical { background: rgba(59,130,246,0.12);  border: 1px solid rgba(59,130,246,0.3);  color: var(--blue2); }
.chip-creative   { background: rgba(245,158,11,0.12);  border: 1px solid rgba(245,158,11,0.3);  color: var(--amber); }
.chip-introvert  { background: rgba(139,92,246,0.12);  border: 1px solid rgba(139,92,246,0.3);  color: var(--purple2); }
.chip-extrovert  { background: rgba(244,63,94,0.12);   border: 1px solid rgba(244,63,94,0.3);   color: var(--rose); }
.chip-leader     { background: rgba(16,185,129,0.12);  border: 1px solid rgba(16,185,129,0.3);  color: var(--green); }

/* ── Top 3 rows ── */
.top3-row {
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 14px; padding: 1rem 1.4rem;
  display: flex; align-items: center; margin-bottom: 0.8rem;
  transition: border-color 0.2s, transform 0.15s;
  animation: fadeUp 0.4s ease both;
}
.top3-row:hover { border-color: rgba(139,92,246,0.3); transform: translateX(4px); }
.top3-rank  { font-family: var(--mono); font-size: 11px; color: var(--dim); width: 26px; flex-shrink: 0; }
.top3-name  { font-size: 14px; font-weight: 600; color: var(--snow); flex: 1; margin-left: 12px; }
.top3-bar-bg { width: 80px; height: 3px; background: var(--navy3); border-radius: 100px; overflow: hidden; }
.top3-bar-fill { height: 100%; border-radius: 100px; background: var(--grad1); }
.top3-pct   { font-family: var(--mono); font-size: 12px; color: var(--purple2); margin-left: 12px; min-width: 42px; text-align: right; }
.top3-icon  { font-size: 18px; margin-left: 10px; }

/* ── Roadmap — vertical timeline ── */
.roadmap-timeline { position: relative; }
.roadmap-step {
  display: flex; gap: 1.4rem; margin-bottom: 1.8rem;
  animation: fadeUp 0.45s ease both;
}
.roadmap-step-left {
  display: flex; flex-direction: column; align-items: center; flex-shrink: 0;
}
.roadmap-step-circle {
  width: 42px; height: 42px; border-radius: 50%;
  background: var(--grad1);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 13px; color: #fff;
  font-weight: 500; flex-shrink: 0;
  box-shadow: 0 0 18px rgba(139,92,246,0.45);
}
.roadmap-step-line {
  width: 2px; flex: 1; margin: 6px 0; min-height: 24px;
  background: linear-gradient(to bottom, rgba(139,92,246,0.35), transparent);
}
.roadmap-step-body { padding-top: 8px; flex: 1; }
.roadmap-step-tag {
  font-family: var(--mono); font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--purple2); margin-bottom: 4px;
}
.roadmap-step-title { font-size: 16px; font-weight: 700; color: var(--snow); margin-bottom: 5px; }
.roadmap-step-desc  { font-size: 13px; color: var(--muted); line-height: 1.65; margin-bottom: 0.7rem; }
.skill-tag {
  display: inline-block;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  border-radius: 8px; padding: 3px 10px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  margin: 2px 3px 2px 0;
}
.skill-tag.purple { background: rgba(139,92,246,0.12); border-color: rgba(139,92,246,0.3); color: var(--purple2); }
.skill-tag.blue   { background: rgba(59,130,246,0.12);  border-color: rgba(59,130,246,0.3);  color: var(--blue2); }
.skill-tag.green  { background: rgba(16,185,129,0.12);  border-color: rgba(16,185,129,0.3);  color: var(--green); }

/* ── Chat ── */
.chat-bubble { max-width: 80%; margin-bottom: 1.1rem; animation: fadeUp 0.3s ease both; }
.chat-bubble.user { margin-left: auto; }
.chat-meta {
  font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--dim);
  margin-bottom: 5px; padding: 0 4px;
}
.chat-bubble.user .chat-meta { text-align: right; }
.chat-text {
  padding: 0.9rem 1.25rem; border-radius: 18px;
  font-size: 14px; line-height: 1.68;
}
.chat-bubble.user .chat-text {
  background: linear-gradient(135deg,rgba(139,92,246,0.20),rgba(59,130,246,0.15));
  border: 1px solid rgba(139,92,246,0.3); color: var(--snow);
  border-bottom-right-radius: 4px;
}
.chat-bubble.bot .chat-text {
  background: var(--glass); border: 1px solid var(--border);
  color: var(--muted); border-bottom-left-radius: 4px;
}
.chat-suggestions {
  display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1.4rem;
}
.chat-sug {
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 100px; padding: 7px 16px;
  font-size: 12px; color: var(--muted); cursor: pointer;
  transition: all 0.2s; font-weight: 500;
}
.chat-sug:hover { border-color: rgba(139,92,246,0.4); color: var(--purple2); }

/* ── Resume ── */
.resume-score-card {
  text-align: center; padding: 2rem;
}
.resume-grade {
  font-family: var(--serif); font-size: 5.5rem;
  line-height: 1; margin-bottom: 0.2rem;
}
.resume-score-sub {
  font-family: var(--mono); font-size: 11px;
  letter-spacing: 2px; color: var(--muted); text-transform: uppercase;
}
.grade-badge {
  display: inline-block; margin-top: 1rem;
  padding: 5px 18px; border-radius: 100px;
  font-family: var(--mono); font-size: 12px; font-weight: 500;
}
.kw-chip {
  display: inline-block;
  padding: 3px 10px; border-radius: 8px;
  font-family: var(--mono); font-size: 11px;
  margin: 2px 3px 2px 0;
}
.kw-found   { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: var(--green); }
.kw-missing { background: rgba(244,63,94,0.10);  border: 1px solid rgba(244,63,94,0.25); color: var(--rose); }
.suggestion-row {
  display: flex; gap: 12px; align-items: flex-start;
  padding: 0.8rem 0; border-bottom: 1px solid var(--border);
  font-size: 13px; color: var(--muted); line-height: 1.65;
}
.suggestion-row:last-child { border-bottom: none; }
.sug-icon { flex-shrink: 0; margin-top: 1px; font-size: 15px; }

/* ── Inputs ── */
.stTextArea textarea {
  background: var(--navy3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--snow) !important;
  font-family: var(--mono) !important; font-size: 13px !important;
}
.stTextArea textarea:focus {
  border-color: rgba(139,92,246,0.45) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.10) !important;
}
.stTextInput input {
  background: var(--navy3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--snow) !important;
  font-family: var(--sans) !important;
}
.stTextInput input:focus {
  border-color: rgba(139,92,246,0.45) !important;
}

/* ── Dividers ── */
.hdivider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(139,92,246,0.15), transparent);
  margin: 2.5rem 0;
}

/* ── Footer ── */
.footer {
  text-align: center; padding: 3rem 0 1.5rem;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 2px; color: var(--dim); text-transform: uppercase; line-height: 2.4;
}

/* ── Animations ── */
@keyframes fadeDown  { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:none} }
@keyframes fadeUp    { from{opacity:0;transform:translateY(12px)}  to{opacity:1;transform:none} }
@keyframes bounceIn  { 0%{opacity:0;transform:scale(0.4)} 70%{transform:scale(1.12)} 100%{opacity:1;transform:scale(1)} }
@keyframes pulse     { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Streamlit overrides ── */
[data-testid="stAlert"] { border-radius: 12px !important; font-family: var(--sans) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# QUIZ DATA — 15 behavioural MCQs
# ══════════════════════════════════════════════════════════════════════
QUESTIONS = [
    {
        "text": "When you face a complex problem, how do you usually approach it?",
        "options": [
            "Break it into logical steps and analyse data carefully",
            "Brainstorm creatively and experiment with novel ideas",
            "Discuss with teammates and gather different viewpoints",
            "Follow an established process or best-practice framework",
        ],
    },
    {
        "text": "Which weekend activity sounds most exciting to you?",
        "options": [
            "Building a personal coding project or automating something",
            "Designing, drawing, writing, or making something visual",
            "Organising a community event or helping people around you",
            "Reading research, learning a new concept, or doing experiments",
        ],
    },
    {
        "text": "In a group project, which role do you naturally gravitate towards?",
        "options": [
            "The builder or developer who writes the code",
            "The designer or storyteller who shapes the experience",
            "The leader or coordinator who keeps everyone aligned",
            "The researcher or analyst who digs into the data",
        ],
    },
    {
        "text": "How would you rate your comfort with Maths and Statistics?",
        "options": [
            "Very comfortable — I genuinely enjoy numbers and equations",
            "Comfortable with fundamentals, find advanced topics challenging",
            "I strongly prefer language, ideas, and creativity over numbers",
            "I manage when I need to, but I dislike it overall",
        ],
    },
    {
        "text": "Which academic subject did you most enjoy or excel in?",
        "options": [
            "Computer Science, Programming, or Engineering",
            "Arts, Design, Media, or Creative Writing",
            "Business, Management, Economics, or Commerce",
            "Physics, Biology, Research, or Pure Mathematics",
        ],
    },
    {
        "text": "How do you feel about speaking or presenting in front of an audience?",
        "options": [
            "I love it — audiences energise and motivate me",
            "I am fine with it, though I prefer smaller groups",
            "I would much rather write a report or send a document",
            "I actively avoid public speaking whenever possible",
        ],
    },
    {
        "text": "How do you typically respond to tight deadlines and high pressure?",
        "options": [
            "I thrive — pressure sharpens my concentration",
            "I manage well as long as I have planned ahead",
            "I struggle but usually push through somehow",
            "I find it very stressful and it affects my output",
        ],
    },
    {
        "text": "Which work environment fits your personality best?",
        "options": [
            "Deep-focus, independent work — remote or alone",
            "Creative studio with freedom, collaboration, and variety",
            "Fast-paced office with lots of people and interaction",
            "Structured environment with clear processes and stability",
        ],
    },
    {
        "text": "How would you describe your overall academic performance?",
        "options": [
            "High achiever — consistently 80 percent or above",
            "Good — solid 65 to 79 percent across most subjects",
            "Average — around 50 to 64 percent, stronger in certain areas",
            "Below average academically but strong in practical skills",
        ],
    },
    {
        "text": "How often do you independently explore new tools or technologies?",
        "options": [
            "Constantly — I am a self-driven, curious learner",
            "Often — especially when a project needs it",
            "Sometimes — mainly when external pressure pushes me",
            "Rarely — I stick to structured classes and formal learning",
        ],
    },
    {
        "text": "Have you taken part in hackathons, competitions, or open-source work?",
        "options": [
            "Yes — multiple times and I genuinely enjoyed every experience",
            "Once or twice — it was a decent and worthwhile experience",
            "Tried once but found it was not really for me",
            "No — I have never participated in any such activity",
        ],
    },
    {
        "text": "What type of product would you find most satisfying to build?",
        "options": [
            "A robust backend system, AI model, or data pipeline",
            "A beautiful app, website, or immersive visual experience",
            "A business strategy, pitch deck, or product launch plan",
            "A scientific research tool, dataset, or analytical model",
        ],
    },
    {
        "text": "How do you prefer to communicate complex ideas to others?",
        "options": [
            "Through data, structured reports, and precise charts",
            "Through visuals, illustrations, mockups, or storytelling",
            "Through live conversation, demos, and direct discussion",
            "Through detailed written documentation and long-form writing",
        ],
    },
    {
        "text": "What motivates you most when choosing a career direction?",
        "options": [
            "Solving hard technical or engineering challenges every day",
            "Creating things that look, feel, or sound truly remarkable",
            "Leading people, driving strategy, and making big decisions",
            "Discovering new knowledge, doing research, and publishing findings",
        ],
    },
    {
        "text": "Where do you picture yourself professionally in five years?",
        "options": [
            "Senior engineer, ML specialist, or technical architect",
            "Creative director, UX lead, or senior product designer",
            "Product manager, startup founder, or management consultant",
            "Research scientist, data expert, or academic professional",
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════
# CAREER ROADMAPS
# ══════════════════════════════════════════════════════════════════════
ROADMAPS = {
    "default": [
        {"tag":"Phase 01 · Foundations","title":"Core Fundamentals","desc":"Master the fundamental concepts of your chosen domain. One language, one tool — go deep before going wide.","skills":["Python / Java","Problem Solving","Git & Version Control"],"type":"purple"},
        {"tag":"Phase 02 · Specialisation","title":"Domain Skills","desc":"Dive into the specialised tools and frameworks your career demands. Build small projects alongside learning.","skills":["Domain Tools","Certifications","Structured Practice"],"type":"blue"},
        {"tag":"Phase 03 · Portfolio","title":"Real-World Projects","desc":"Build 2–3 portfolio projects that demonstrate your abilities to employers. Deploy and document every one.","skills":["GitHub Portfolio","Documentation","Deployment"],"type":"green"},
        {"tag":"Phase 04 · Network","title":"Community & Visibility","desc":"Engage with professional communities, attend meetups, contribute to open-source, and write about your work.","skills":["LinkedIn","Open Source","Tech Blog"],"type":"purple"},
        {"tag":"Phase 05 · Career","title":"Land Your First Role","desc":"Apply strategically. Tailor your resume to each role and prepare thoroughly for domain-specific interviews.","skills":["Resume Polish","Mock Interviews","LeetCode / DSA"],"type":"blue"},
    ],
    "software developer": [
        {"tag":"Phase 01 · Foundations","title":"Programming Fundamentals","desc":"Learn Python or JavaScript deeply. Understand data structures, algorithms, and OOP from first principles.","skills":["Python","JavaScript","DSA","Git"],"type":"purple"},
        {"tag":"Phase 02 · Full-Stack","title":"Web & API Development","desc":"Build end-to-end applications with modern frameworks. Master REST APIs, authentication, and databases.","skills":["React / Vue","Node.js / Django","SQL","REST APIs"],"type":"blue"},
        {"tag":"Phase 03 · DevOps","title":"Deployment & Infrastructure","desc":"Learn CI/CD pipelines, containerisation, and cloud platforms to ship production-grade software.","skills":["Docker","GitHub Actions","AWS / GCP","Linux"],"type":"green"},
        {"tag":"Phase 04 · Portfolio","title":"Three Showcase Projects","desc":"Build: a CRUD app with auth, a real-time app with WebSockets, and a complex API-heavy service. Deploy all three.","skills":["Full Stack Apps","Open Source","GitHub"],"type":"purple"},
        {"tag":"Phase 05 · Interviews","title":"Interview Mastery","desc":"Practice DSA daily, study system design patterns, and conduct mock interviews with peers.","skills":["LeetCode","System Design","Mock Interviews"],"type":"blue"},
    ],
    "data scientist": [
        {"tag":"Phase 01 · Foundations","title":"Python & Statistical Thinking","desc":"Master Python for data work and build strong statistical reasoning around probability and inference.","skills":["Python","NumPy","Pandas","Statistics"],"type":"purple"},
        {"tag":"Phase 02 · Machine Learning","title":"Core ML Algorithms","desc":"Learn supervised and unsupervised learning, model evaluation, cross-validation, and feature engineering.","skills":["scikit-learn","XGBoost","Feature Engineering","CV"],"type":"blue"},
        {"tag":"Phase 03 · Deep Learning","title":"Neural Networks & NLP","desc":"Explore neural networks, language models, and computer vision with PyTorch or TensorFlow.","skills":["TensorFlow","PyTorch","CNNs","Transformers"],"type":"green"},
        {"tag":"Phase 04 · Communication","title":"Data Storytelling","desc":"Translate complex findings into clear, compelling visuals and dashboards that drive decisions.","skills":["Matplotlib","Seaborn","Power BI","Streamlit"],"type":"purple"},
        {"tag":"Phase 05 · Projects","title":"End-to-End ML Projects","desc":"Compete on Kaggle, deploy models as APIs, and publish insights publicly on Medium or LinkedIn.","skills":["Kaggle","MLflow","FastAPI","Hugging Face"],"type":"blue"},
    ],
    "machine learning engineer": [
        {"tag":"Phase 01 · Foundations","title":"Math & Python Mastery","desc":"Linear algebra, calculus, probability theory, and clean Python are the non-negotiable bedrock.","skills":["Linear Algebra","Calculus","Probability","Python"],"type":"purple"},
        {"tag":"Phase 02 · Frameworks","title":"Deep Learning Tools","desc":"Build and train neural networks with PyTorch and TensorFlow. Understand modern architectures deeply.","skills":["PyTorch","TensorFlow","Keras","JAX"],"type":"blue"},
        {"tag":"Phase 03 · MLOps","title":"Production ML Systems","desc":"Deploy, monitor, retrain, and scale ML models. Learn the full lifecycle from experiment to production.","skills":["Docker","Kubernetes","MLflow","Airflow"],"type":"green"},
        {"tag":"Phase 04 · Specialisation","title":"Choose Your Domain","desc":"Pick NLP, Computer Vision, Recommender Systems, or Reinforcement Learning — then go extremely deep.","skills":["Transformers","YOLO","RecSys","RL"],"type":"purple"},
        {"tag":"Phase 05 · Research","title":"Contribute & Publish","desc":"Read arXiv papers, reproduce results, contribute to Hugging Face, and build an open research presence.","skills":["arXiv","Papers With Code","Hugging Face","GitHub"],"type":"blue"},
    ],
    "web developer": [
        {"tag":"Phase 01 · Foundations","title":"The Web Triad","desc":"Master HTML, CSS, and JavaScript from scratch. Build responsive, accessible layouts without frameworks.","skills":["HTML5","CSS3","JavaScript ES6+","Flexbox/Grid"],"type":"purple"},
        {"tag":"Phase 02 · Frontend","title":"React & Modern Tooling","desc":"Learn React deeply — state, hooks, performance. Add TypeScript, Webpack/Vite, and testing to your stack.","skills":["React","Next.js","TypeScript","Redux"],"type":"blue"},
        {"tag":"Phase 03 · Backend","title":"Servers & Databases","desc":"Build REST APIs and connect to databases. Understand auth, ORMs, and scalability basics.","skills":["Node.js","Express","PostgreSQL","GraphQL"],"type":"green"},
        {"tag":"Phase 04 · Quality","title":"Performance & Testing","desc":"Optimise Core Web Vitals, write automated tests, set up CI/CD, and improve developer experience.","skills":["Jest / Cypress","Lighthouse","GitHub Actions","Docker"],"type":"purple"},
        {"tag":"Phase 05 · Ship","title":"Portfolio & Launch","desc":"Deploy real projects to Vercel or Netlify and build a personal portfolio that stands out.","skills":["Vercel / Netlify","GitHub","Portfolio Site","Open Source"],"type":"blue"},
    ],
    "ux designer": [
        {"tag":"Phase 01 · Foundations","title":"Design Principles","desc":"Study Gestalt theory, typography, colour, layout, and accessibility standards for digital products.","skills":["Typography","Colour Theory","Gestalt","WCAG"],"type":"purple"},
        {"tag":"Phase 02 · Research","title":"UX Research Methods","desc":"Master user interviews, usability testing, affinity mapping, and persona creation workflows.","skills":["User Interviews","Usability Testing","Personas","Journey Maps"],"type":"blue"},
        {"tag":"Phase 03 · Tools","title":"Figma & Prototyping","desc":"Become proficient in Figma — auto layout, design systems, interactive prototypes, and handoff.","skills":["Figma","Prototyping","Design Systems","Auto Layout"],"type":"green"},
        {"tag":"Phase 04 · Portfolio","title":"Case Study Portfolio","desc":"Document three redesign or original projects with clear problem statements, process, and outcomes.","skills":["Case Studies","Behance","Dribbble","Portfolio Site"],"type":"purple"},
        {"tag":"Phase 05 · Collaboration","title":"Developer Partnership","desc":"Work in cross-functional teams, communicate decisions clearly, and deliver precise developer handoffs.","skills":["Dev Handoff","Zeplin","Agile/Scrum","A/B Testing"],"type":"blue"},
    ],
    "product manager": [
        {"tag":"Phase 01 · Foundations","title":"Product Thinking","desc":"Understand user needs, define product vision, and learn market positioning and competitive analysis.","skills":["User Research","Jobs-to-be-Done","Competitive Analysis"],"type":"purple"},
        {"tag":"Phase 02 · Strategy","title":"Roadmaps & Prioritisation","desc":"Learn OKRs, RICE, MoSCoW, and write clear, actionable product requirements documents.","skills":["OKRs","RICE","Roadmaps","PRDs"],"type":"blue"},
        {"tag":"Phase 03 · Data","title":"Analytics & Experimentation","desc":"Drive decisions with product analytics. Master funnels, retention, cohort analysis, and A/B testing.","skills":["Mixpanel","SQL","A/B Testing","Cohort Analysis"],"type":"green"},
        {"tag":"Phase 04 · Execution","title":"Agile Leadership","desc":"Lead sprints, manage engineering relationships, and ship products iteratively and on schedule.","skills":["Jira","Agile","Scrum","Stakeholder Mgmt"],"type":"purple"},
        {"tag":"Phase 05 · Career","title":"PM Portfolio","desc":"Build side projects, document case studies, and target Associate PM programmes at top companies.","skills":["Case Studies","PM Certs","Networking","APM Roles"],"type":"blue"},
    ],
    "data analyst": [
        {"tag":"Phase 01 · Foundations","title":"SQL & Excel","desc":"Data manipulation in Excel, then SQL for querying large structured datasets efficiently.","skills":["Excel","SQL","Pivot Tables","Joins & Aggregations"],"type":"purple"},
        {"tag":"Phase 02 · Python","title":"Python for Analysis","desc":"Automate analysis workflows with Python. Master Pandas, NumPy, and visualisation libraries.","skills":["Python","Pandas","Matplotlib","Seaborn"],"type":"blue"},
        {"tag":"Phase 03 · Visualisation","title":"Dashboards & BI","desc":"Build executive-ready dashboards and business intelligence reports with Tableau or Power BI.","skills":["Tableau","Power BI","Looker","Streamlit"],"type":"green"},
        {"tag":"Phase 04 · Statistics","title":"Statistical Analysis","desc":"Apply hypothesis testing, regression, and A/B experiment design to answer business questions.","skills":["Statistics","Hypothesis Testing","Regression","A/B Tests"],"type":"purple"},
        {"tag":"Phase 05 · Portfolio","title":"Public Work","desc":"Share analyses on GitHub, write explanatory articles on Medium, and solve Kaggle datasets publicly.","skills":["Kaggle","GitHub","Medium Blog","Public APIs"],"type":"blue"},
    ],
    "cybersecurity analyst": [
        {"tag":"Phase 01 · Foundations","title":"Networking & Linux","desc":"Master TCP/IP, DNS, firewalls, and Linux command line. Analyse network traffic with Wireshark.","skills":["TCP/IP","DNS","Wireshark","Linux CLI"],"type":"purple"},
        {"tag":"Phase 02 · Security Core","title":"Attack Vectors & Defence","desc":"Learn cryptography, authentication protocols, OWASP Top 10, and common vulnerability patterns.","skills":["Cryptography","OWASP Top 10","PKI","Auth Protocols"],"type":"blue"},
        {"tag":"Phase 03 · Offensive","title":"Ethical Hacking","desc":"Practice through CTF challenges and pentesting labs. Build hands-on experience safely.","skills":["Kali Linux","Metasploit","Burp Suite","CTFs"],"type":"green"},
        {"tag":"Phase 04 · Certifications","title":"Industry Credentials","desc":"Earn recognised certifications to validate your expertise to potential employers.","skills":["CompTIA Security+","CEH","OSCP","CISSP"],"type":"purple"},
        {"tag":"Phase 05 · SOC","title":"Incident Response","desc":"Detect, respond to, contain, and recover from real-world security incidents inside a SOC.","skills":["SIEM","Splunk","Incident Response","Threat Intel"],"type":"blue"},
    ],
}

# ══════════════════════════════════════════════════════════════════════
# CHATBOT KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════
CHATBOT_KB = {
    "_general": {
        "salary":        "Salaries vary widely by location and experience. In India, entry-level tech roles range from 4 to 12 LPA; senior roles reach 25–60 LPA. In the US, junior engineers earn 70–110K USD; seniors often exceed 150K USD.",
        "internship":    "To land internships: build 2–3 projects on GitHub, connect actively with founders and HRs on LinkedIn, and apply via Internshala, Unstop, AngelList, and LinkedIn. Thoughtful cold emails work surprisingly well.",
        "resume":        "A strong resume is one page, ATS-friendly, and highlights measurable impact using numbers. Use the STAR format for experience bullets. Use the Resume Analyzer tab for personalised feedback.",
        "certifications":"Certifications add real credibility. Prioritise: Google, AWS, Microsoft, Coursera, or edX certs that align directly with your target career path.",
        "college":       "Your college name helps open doors but is far from the whole story. A strong GitHub portfolio, live projects, and clear communication often outweigh institution prestige at most companies.",
    },
    "software developer": {
        "skills":    "Master: Python or JavaScript, DSA, REST APIs, React (frontend), Django or Node.js (backend), Docker, and Git. TypeScript is increasingly essential.",
        "start":     "Start with Python fundamentals, build a CRUD app, learn React, connect frontend and backend, then deploy. Aim for three live deployed projects within six months.",
        "next":      "After the basics: learn TypeScript, system design patterns, contribute to open-source, and solve 100+ LeetCode medium problems consistently.",
        "projects":  "Build: 1) A task manager with authentication, 2) A real-time chat app, 3) A well-documented REST API. All three should be live, deployed, and on GitHub.",
        "interview": "Practice DSA on LeetCode daily, study system design via Grokking the System Design Interview, and do mock interviews on Pramp or interviewing.io.",
    },
    "data scientist": {
        "skills":    "Master: Python, Pandas, NumPy, scikit-learn, SQL, Statistics, Matplotlib or Seaborn, and at least one deep learning framework such as PyTorch or TensorFlow.",
        "start":     "Python first, then Statistics, then Pandas, then ML with scikit-learn, then your first Kaggle notebook. Introduce deep learning incrementally.",
        "next":      "Specialise in NLP with Hugging Face, Computer Vision with CNNs, or MLOps with MLflow and Docker. Go deep in one area rather than shallow across many.",
        "projects":  "Build: 1) A full EDA and prediction notebook on Kaggle, 2) A tweet sentiment analyser, 3) A deployed ML model as a Streamlit or FastAPI app.",
        "interview": "Revise bias-variance tradeoff, regularisation, and ensemble methods. Practice SQL. Prepare a strong end-to-end project walkthrough.",
    },
    "machine learning engineer": {
        "skills":    "Core: Python, Linear Algebra, PyTorch or TensorFlow, MLOps tools (Docker, Kubernetes, MLflow), FastAPI for model serving, and cloud platforms like AWS SageMaker.",
        "start":     "Math foundations, then Python, then ML theory, then PyTorch, then train your first models, then MLOps, then deploy a model to production.",
        "next":      "Specialise in NLP, Computer Vision, or Reinforcement Learning. Contribute to Hugging Face or Papers With Code. Target MLE roles at AI-first companies.",
        "projects":  "Build: 1) A fine-tuned LLM application, 2) A computer vision classifier deployed as an API, 3) A recommender system with offline and online evaluation.",
        "interview": "Prepare ML system design questions, Python exercises, model debugging scenarios, and a detailed walkthrough of an end-to-end ML project.",
    },
    "web developer": {
        "skills":    "HTML, CSS, JavaScript (deeply), React, Node.js or Express, SQL or MongoDB, REST APIs, TypeScript, Docker, and CI/CD pipelines.",
        "start":     "Clone a simple website, build your portfolio, add a backend, connect a database, then deploy. Ship something real within three months.",
        "next":      "TypeScript, Jest and Cypress for testing, Core Web Vitals optimisation, and cloud fundamentals with AWS or Vercel.",
        "projects":  "Build: 1) Personal portfolio site, 2) A SaaS landing page with auth, 3) A real-time collaboration tool with WebSockets.",
        "interview": "Focus on JavaScript fundamentals, React internals, REST API design, and one full-stack project you can discuss in full technical depth.",
    },
    "ux designer": {
        "skills":    "Figma, UX research methods, information architecture, prototyping, design systems, WCAG accessibility standards, and basic HTML and CSS awareness.",
        "start":     "Learn design principles, use Figma daily, redesign three popular apps, and document each process as a detailed case study.",
        "next":      "Go deep on UX research through user interviews and usability testing. Explore motion design with Lottie or Framer. Collaborate on real product teams.",
        "projects":  "Build: 1) A redesign case study with before/after, 2) A new app concept with a full Figma prototype, 3) A reusable design system component library.",
        "interview": "Present case studies clearly. Explain every design decision and its rationale. Walk through your complete research and ideation process step by step.",
    },
    "product manager": {
        "skills":    "Product thinking, roadmapping frameworks, SQL, user research, A/B testing, stakeholder communication, and Agile or Scrum methodology.",
        "start":     "Read Inspired by Marty Cagan, shadow a PM, build a side product, document it as a case study, then apply for Associate PM programmes.",
        "next":      "Develop a specialisation: growth PM, platform PM, or B2B enterprise PM. Get SQL-proficient. Learn to interpret analytics independently.",
        "projects":  "Document: 1) A product teardown of an app you use daily, 2) A full PRD for a new feature, 3) A go-to-market strategy and launch plan.",
        "interview": "Prepare product design, metrics, market estimation, and strategy questions. Use resources like Exponent for structured PM interview practice.",
    },
    "data analyst": {
        "skills":    "SQL is essential. Then Excel, Python with Pandas, Tableau or Power BI for dashboards, Statistics for interpretation, and storytelling with data.",
        "start":     "SQL, then Excel, then Python basics, then Pandas, then build your first interactive dashboard in Tableau or Power BI.",
        "next":      "A/B testing, statistical modelling, Python automation, and telling data stories that directly influence business strategy.",
        "projects":  "Build: 1) A sales dashboard in Tableau, 2) A full EDA report on a public dataset, 3) An automated Python report delivered via email.",
        "interview": "Practice SQL joins and aggregations, walk through past analyses, and explain how you converted data insights into concrete business actions.",
    },
    "cybersecurity analyst": {
        "skills":    "Networking fundamentals (TCP/IP, DNS), Linux command line, Python scripting, Wireshark, Burp Suite for web app testing, and SIEM tools.",
        "start":     "Networking fundamentals, then Linux, then Python scripting, then OWASP Top 10, then CTF challenges on HackTheBox or TryHackMe.",
        "next":      "Specialise in penetration testing, threat intelligence, SOC analysis, or cloud security. Pursue CompTIA Security+ as your first major certification.",
        "projects":  "Complete: 1) The TryHackMe beginner path, 2) A Kali Linux home lab setup, 3) A written vulnerability assessment report for a sample target.",
        "interview": "Expect: networking protocol questions, SQLi and XSS attack mechanics, incident response procedures, and deep dives into your lab work.",
    },
}

INTENT_MAP = {
    "skills":        ["skill","learn","what should i","which tech","language","tool","know","technology","need to study"],
    "start":         ["start","begin","new to","how to get","roadmap","first step","where do i","getting started"],
    "next":          ["next","after","advanced","already know","improve","level up","what else","progression"],
    "projects":      ["project","portfolio","build","practice","example","idea","make","create"],
    "interview":     ["interview","prepare","crack","question","job","hiring","placement","selection"],
    "salary":        ["salary","pay","earn","money","package","lpa","ctc","compensation","income"],
    "internship":    ["intern","internship","experience","fresher","entry level","placement"],
    "resume":        ["resume","cv","profile","ats","curriculum vitae"],
    "certifications":["cert","certification","course","udemy","coursera","mooc","badge","credential"],
    "college":       ["college","degree","tier","nit","iit","university","campus","institution"],
}

def detect_intent(msg: str) -> str:
    """Identify the user's intent from a chat message."""
    msg_l = msg.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(k in msg_l for k in keywords):
            return intent
    return "unknown"

def chatbot_response(user_msg: str, career: str) -> str:
    """
    Rule-based chatbot response engine.
    Tries career-specific answers first, falls back to general KB.
    """
    career_l   = career.lower().strip()
    intent     = detect_intent(user_msg)
    kb_career  = CHATBOT_KB.get(career_l, {})
    kb_general = CHATBOT_KB["_general"]

    if intent != "unknown" and intent in kb_career:
        return kb_career[intent]
    if intent != "unknown" and intent in kb_general:
        return kb_general[intent]

    # Keyword scan fallback
    msg_l = user_msg.lower()
    for key, val in kb_career.items():
        if key in msg_l:
            return val

    return (
        f"That is a great question for someone exploring {career.title()}! "
        f"I can help you with: skills to learn, how to get started, project ideas, "
        f"interview preparation, salary expectations, or internship advice. "
        f"What would you like to know more about?"
    )

# ══════════════════════════════════════════════════════════════════════
# RESUME ANALYZER
# ══════════════════════════════════════════════════════════════════════
RESUME_KEYWORDS = {
    "default":              ["Python","project","team","experience","skills","education","GitHub","communication"],
    "software developer":   ["Python","JavaScript","React","Node","API","SQL","Docker","Git","testing","deployment","agile","CI/CD"],
    "data scientist":       ["Python","machine learning","TensorFlow","PyTorch","pandas","scikit","SQL","statistics","model","Kaggle","deep learning","EDA"],
    "machine learning engineer": ["PyTorch","TensorFlow","MLOps","model deployment","Docker","Kubernetes","pipeline","NLP","computer vision","MLflow","API"],
    "web developer":        ["HTML","CSS","JavaScript","React","Node","REST","API","responsive","TypeScript","deployment","database","frontend","backend"],
    "ux designer":          ["Figma","user research","prototype","wireframe","usability","design system","accessibility","case study","UI","UX","user testing"],
    "product manager":      ["roadmap","stakeholder","OKR","user research","A/B test","agile","sprint","metrics","strategy","product","KPI","launch"],
    "data analyst":         ["SQL","Excel","Tableau","Power BI","Python","dashboard","KPI","report","analysis","statistics","visualisation"],
    "cybersecurity analyst":["penetration","SIEM","firewall","vulnerability","Kali","network","incident response","cryptography","security","compliance","threat"],
}

def analyze_resume(text: str, career: str) -> dict:
    """
    Score a resume against career-specific keywords and structural signals.
    Returns score (0–100), grade (A–D), matched/missing keywords, and suggestions.
    """
    career_l   = career.lower().strip()
    keywords   = RESUME_KEYWORDS.get(career_l, RESUME_KEYWORDS["default"])
    text_lower = text.lower()

    matched = [k for k in keywords if k.lower() in text_lower]
    missing = [k for k in keywords if k.lower() not in text_lower]

    # Scoring components
    kw_score   = (len(matched) / max(len(keywords), 1)) * 48
    length_ok  = 180 < len(text.split()) < 750
    has_github = "github" in text_lower
    has_nums   = bool(re.search(r"\d+\s*%|\d+x|\$[\d,]+|₹[\d,]+|\d+\s*(users|clients|projects)", text, re.I))
    has_action = any(w in text_lower for w in ["built","developed","designed","led","improved",
                     "reduced","increased","deployed","created","launched","optimised","automated","delivered"])
    has_edu    = any(w in text_lower for w in ["b.tech","b.e","bsc","msc","bachelor","master","degree","cgpa","gpa"])
    has_contact = any(w in text_lower for w in ["email","phone","linkedin","@","mobile"])

    bonus = sum([
        has_github  * 9,
        has_nums    * 11,
        has_action  * 8,
        has_edu     * 5,
        has_contact * 4,
        (14 if length_ok else 3),
    ])
    score = max(0, min(100, int(kw_score + bonus)))
    grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))

    # Suggestions
    suggestions = []
    if len(matched) < len(keywords) * 0.5:
        top_missing = ", ".join(missing[:4])
        suggestions.append(("warning", f"Add key {career.title()} terms: {top_missing}"))
    if not has_nums:
        suggestions.append(("chart",   "Quantify achievements — use numbers, percentages, or scale metrics"))
    if not has_github:
        suggestions.append(("link",    "Add your GitHub profile link to showcase real code and projects"))
    if not has_action:
        suggestions.append(("rocket",  "Start every bullet with an action verb: Built, Deployed, Optimised, Led"))
    if not length_ok:
        msg = ("Your resume appears too short — expand your project and experience sections"
               if len(text.split()) <= 180
               else "Your resume is long — aim for one page (approx. 400–600 words)")
        suggestions.append(("doc", msg))
    if not has_edu:
        suggestions.append(("grad",  "Include your educational qualifications clearly (degree, institution, year, CGPA)"))
    if not has_contact:
        suggestions.append(("phone", "Ensure your contact information (email, phone, LinkedIn) is clearly visible"))
    if not suggestions:
        suggestions.append(("check", "Strong resume! Consider a peer review or running it through an ATS checker."))

    return {
        "score":       score,
        "grade":       grade,
        "matched_kws": matched,
        "missing_kws": missing,
        "suggestions": suggestions,
    }

# ══════════════════════════════════════════════════════════════════════
# CAREER ICON MAPPING
# ══════════════════════════════════════════════════════════════════════
CAREER_ICONS = {
    "software developer":        "💻",
    "data scientist":            "📊",
    "machine learning engineer": "🤖",
    "web developer":             "🌐",
    "ux designer":               "🎨",
    "product manager":           "🎯",
    "data analyst":              "📈",
    "cybersecurity analyst":     "🔐",
    "cloud engineer":            "☁️",
    "devops engineer":           "⚙️",
    "database administrator":    "🗄️",
    "network engineer":          "🔌",
    "mobile developer":          "📱",
    "ai researcher":             "🧬",
    "blockchain developer":      "⛓️",
}

def career_icon(career: str) -> str:
    """Return emoji for a career string."""
    c = career.lower()
    for key, icon in CAREER_ICONS.items():
        if key in c:
            return icon
    return "🚀"

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """Load career_model.pkl safely with error handling."""
    path = os.path.join(BASE_DIR, "career_model.pkl")
    if not os.path.exists(path):
        return None, "career_model.pkl not found — place it in the same folder as app.py"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

# ══════════════════════════════════════════════════════════════════════
# QUIZ → FEATURES MAPPING
# ══════════════════════════════════════════════════════════════════════
def quiz_to_features(answers: list) -> np.ndarray:
    """
    Convert 15 quiz answer indices (0–3) into a 23-feature numerical vector
    that matches the training schema of career_model.pkl.

    Feature order (mirrors original training data):
      [0]  logical quotient rating    (0–10)
      [1]  coding skills rating       (0–10)
      [2]  hackathons                 (0–10)
      [3]  public speaking points     (0–10)
      [4]  self-learning capability   (0/1)
      [5]  extra courses did          (0/1)
      [6]  certifications             (0–5)
      [7]  workshops                  (0–5)
      [8]  reading & writing skills   (0–10)
      [9]  memory capability score    (0–10)
      [10] interested subjects        (encoded 1–7)
      [11] interested career area     (encoded 1–8)
      [12] job / higher studies       (0/1)
      [13] company type preference    (encoded 1–5)
      [14] taken inputs from seniors  (0/1)
      [15] interested book type       (encoded 1–5)
      [16] management or technical    (0/1)
      [17] hard / smart worker        (0/1)
      [18] worked in teams            (0/1)
      [19] introvert                  (0/1)
      [20] academic percentage        (0–100)
      [21] math score                 (0–10)
      [22] soft skills score          (0–10)
    """
    a = answers  # shorthand

    # ── Raw signals derived from answers ──────────────────────────────
    logical_boost  = 3 if a[0] == 0 else (1 if a[0] == 3 else 0)
    creative_boost = 3 if a[0] == 1 else 0
    social_boost   = 3 if a[0] == 2 else 0
    tech_boost     = 3 if a[1] == 0 else 0
    design_boost   = 2 if a[1] == 1 else 0
    math_raw       = [9, 6, 3, 2][a[3]]
    ps_raw         = [9, 6, 3, 1][a[5]]
    pressure_bonus = [2, 1, 0, 0][a[6]]
    introvert_flag = 1 if a[7] == 0 else 0
    academic_pct   = [88, 72, 57, 45][a[8]]
    self_learn     = 1 if a[9] in [0, 1] else 0
    hackathon_cnt  = [4, 2, 1, 0][a[10]]
    mgmt_flag      = 1 if a[13] == 2 else 0
    rw_raw         = [7, 6, 5, 9][a[12]]
    job_flag       = 1 if a[14] in [0, 1] else 0

    # ── Composite scores ──────────────────────────────────────────────
    logical_score  = min(10, 5 + logical_boost + tech_boost + pressure_bonus)
    coding_score   = min(10, 4 + tech_boost + (2 if a[1] == 0 else 0) + (2 if a[4] == 0 else 0))
    soft_score     = min(10, 4 + social_boost + (2 if a[12] == 2 else 0))
    memory_score   = min(10, 5 + logical_boost + (2 if a[9] == 0 else 0))
    cert_count     = min(5, 1 + (2 if a[9] in [0, 1] else 0) + (1 if a[10] in [0, 1] else 0))
    workshop_count = min(5, 1 + (1 if a[10] in [0, 1] else 0) + (1 if a[11] in [0, 1] else 0))

    # ── Encodings for categorical features ────────────────────────────
    subject_enc = {0: 3, 1: 5, 2: 6, 3: 4}      # CS / Arts / Business / Science
    career_enc  = {0: 1, 1: 4, 2: 6, 3: 2}       # Tech / Design / Mgmt / Research
    company_enc = {0: 2, 1: 3, 2: 4, 3: 1}       # Remote / Studio / Corporate / Structured
    book_enc    = {0: 2, 1: 4, 2: 3, 3: 1}       # Data / Visual / Conversational / Written

    vector = [
        logical_score,                        # [0]  logical
        coding_score,                         # [1]  coding
        hackathon_cnt,                        # [2]  hackathons
        ps_raw,                               # [3]  public speaking
        self_learn,                           # [4]  self-learning
        1 if a[9] in [0, 1] else 0,           # [5]  extra courses
        cert_count,                           # [6]  certifications
        workshop_count,                       # [7]  workshops
        rw_raw,                               # [8]  reading & writing
        memory_score,                         # [9]  memory
        subject_enc.get(a[4], 3),             # [10] interested subjects
        career_enc.get(a[11], 1),             # [11] career area
        job_flag,                             # [12] job/higher studies
        company_enc.get(a[7], 2),             # [13] company type
        1 if a[6] in [0, 1] else 0,           # [14] senior input
        book_enc.get(a[12], 2),               # [15] book type
        1 - mgmt_flag,                        # [16] technical (1) vs management (0)
        1 if a[6] == 0 else 0,                # [17] smart worker
        1 if a[2] in [0, 1, 2] else 0,        # [18] team work
        introvert_flag,                       # [19] introvert
        academic_pct,                         # [20] academic %
        math_raw,                             # [21] math
        soft_score,                           # [22] soft skills
    ]
    return np.array([vector], dtype=float)

def predict(model, feature_vector: np.ndarray) -> tuple:
    """
    Run prediction and return (career_str, confidence_pct, top3_list).
    Handles models without predict_proba gracefully.
    """
    raw    = model.predict(feature_vector)
    top_c  = str(raw[0]).strip().lower()
    top3   = []
    conf   = None

    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(feature_vector)[0]
        idx_top = np.argsort(probs)[::-1][:3]
        for idx in idx_top:
            name = str(model.classes_[idx]).strip().lower()
            top3.append((name, round(float(probs[idx]) * 100, 1)))
        if top3:
            top_c = top3[0][0]
            conf  = top3[0][1]

    return top_c, conf, top3

def infer_personality(answers: list) -> list:
    """Infer personality descriptors from quiz answer patterns."""
    tags = []
    # Introvert vs extrovert
    if answers[7] == 0 or answers[5] in [2, 3]:
        tags.append(("Introvert",   "chip-introvert", "🪐"))
    else:
        tags.append(("Extrovert",   "chip-extrovert", "⚡"))
    # Analytical
    if answers[0] == 0 and answers[3] in [0, 1]:
        tags.append(("Analytical",  "chip-analytical","🔬"))
    # Creative
    if answers[0] == 1 or answers[1] == 1:
        tags.append(("Creative",    "chip-creative",  "🎨"))
    # Leader
    if answers[2] == 2 and answers[5] == 0:
        tags.append(("Leader",      "chip-leader",    "🏆"))
    # Researcher
    if answers[0] == 0 and answers[1] == 3 and answers[13] == 3:
        tags.append(("Researcher",  "chip-analytical","📚"))
    return tags if tags else [("Explorer", "chip-analytical", "🌐")]

def generate_explanation(career: str, answers: list) -> str:
    """Generate a personalised explanation of why this career suits the user."""
    strengths = []
    if answers[5] == 0:          strengths.append("natural communication ability")
    if answers[0] == 0:          strengths.append("sharp logical reasoning")
    if answers[3] in [0, 1]:     strengths.append("strong mathematical grounding")
    if answers[9] == 0:          strengths.append("self-driven curiosity")
    if answers[10] in [0, 1]:    strengths.append("competitive hands-on experience")
    if answers[0] == 1:          strengths.append("creative problem-solving instinct")
    if answers[6] == 0:          strengths.append("high-performance mindset")
    if not strengths:
        strengths = ["a versatile skill profile", "intellectual adaptability"]
    s = " and ".join(strengths[:2])
    return f"Your {s} position you well for a career in {career.title()}. The role aligns naturally with how you think, work, and grow."

def render_roadmap(career: str):
    """Render the vertical timeline roadmap for a given career."""
    key   = career.lower().strip()
    steps = ROADMAPS.get(key, ROADMAPS["default"])
    for i, step in enumerate(steps):
        is_last   = (i == len(steps) - 1)
        tags_html = "".join(
            f'<span class="skill-tag {step["type"]}">{s}</span>'
            for s in step["skills"]
        )
        line_html = "" if is_last else '<div class="roadmap-step-line"></div>'
        st.markdown(f"""
<div class="roadmap-step" style="animation-delay:{i*0.08}s">
  <div class="roadmap-step-left">
    <div class="roadmap-step-circle">{str(i+1).zfill(2)}</div>
    {line_html}
  </div>
  <div class="roadmap-step-body">
    <div class="roadmap-step-tag">{step["tag"]}</div>
    <div class="roadmap-step-title">{step["title"]}</div>
    <div class="roadmap-step-desc">{step["desc"]}</div>
    <div>{tags_html}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("home",    "🏠", "Home"),
    ("quiz",    "📝", "Neural Assessment"),
    ("results", "🎯", "AI Projections"),
    ("resume",  "📄", "Resume Analyzer"),
    ("chat",    "💬", "AI Mentor"),
]

def render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
<div class="nav-brand">
  <div class="nav-logo">
    <div class="nav-logo-icon">◈</div>
    NextStep AI
  </div>
  <div class="nav-tagline">Your Smart Career Guide to the Next Step</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

        current = st.session_state.get("page", "home")
        for key, icon, label in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-section-label">Your Profile</div>', unsafe_allow_html=True)

        career = st.session_state.get("career", "")
        conf   = st.session_state.get("confidence", None)
        if career:
            conf_str = f"  ·  {conf}% confidence" if conf else ""
            st.markdown(f"""
<div class="nav-status">
  <div class="nav-status-career">{career_icon(career)}  {career.title()}</div>
  <div class="nav-status-label">Predicted Career{conf_str}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="nav-status">
  <div style="font-size:12px;color:var(--dim)">Complete the quiz to see your prediction.</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="nav-footer">NextStep AI · v3.0 · Zero APIs</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <span class="dot"></span>
    AI-Powered · ML Prediction · Zero External APIs
  </div>
  <div class="hero-title">
    Discover Your<br><em>Perfect Career</em><br>
    <span class="gradient-text">with NextStep AI</span>
  </div>
  <div class="hero-sub">
    Unlock your professional destiny using advanced machine learning
    and deep personality mapping. Your smart career guide to the next step.
  </div>
  <div class="hero-features">
    <div class="hero-feat"><span class="feat-icon">📝</span> 15-Question Assessment</div>
    <div class="hero-feat"><span class="feat-icon">🤖</span> ML Career Prediction</div>
    <div class="hero-feat"><span class="feat-icon">🗺️</span> Step-by-Step Roadmap</div>
    <div class="hero-feat"><span class="feat-icon">💬</span> AI Career Mentor</div>
    <div class="hero-feat"><span class="feat-icon">📄</span> Resume Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div class="section-eyebrow">Cognitive Toolkit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The Neural Advantage</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Everything you need to find and pursue the right career path — all in one intelligent platform.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    features = [
        ("🤖","AI Prediction","Real-time ML analysis maps your profile to the best-fit career with confidence scoring.","card-purple"),
        ("🧠","Personality Mapping","Deep behavioural analysis reveals your Introvert / Extrovert / Analytical / Creative profile.","card-blue"),
        ("🗺️","Roadmap Generator","Phase-by-phase learning paths with tools, skills, and projects for your exact career.","card-cyan"),
        ("💬","Mentor Chatbot","24/7 AI-powered career coaching with no API keys — fully local and private.","card-green"),
        ("📄","Resume Analyzer","Keyword scoring and improvement suggestions tailored to your predicted career.","card-purple"),
        ("📊","Top 3 Projections","See the top three careers with probability bars and confidence percentages.","card-blue"),
    ]
    for i, (em, title, desc, cls) in enumerate(features):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
<div class="card {cls}">
  <div class="feat-item-icon">{em}</div>
  <div class="feat-item-title">{title}</div>
  <div class="feat-item-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
    b1, _, _ = st.columns([1, 2, 2])
    with b1:
        if st.button("🚀  Start Neural Assessment", use_container_width=True):
            st.session_state.page    = "quiz"
            st.session_state.q_index = 0
            st.session_state.answers = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ══════════════════════════════════════════════════════════════════════
def page_quiz(model):
    if "q_index" not in st.session_state: st.session_state.q_index = 0
    if "answers" not in st.session_state: st.session_state.answers  = []

    total = len(QUESTIONS)
    qi    = st.session_state.q_index

    # Header with progress
    pct = int((qi / total) * 100)
    st.markdown(f"""
<div class="quiz-header">
  <div class="quiz-header-left">
    <div class="quiz-prog-label">Neural Assessment Pulse · {pct}% Complete</div>
    <div class="quiz-prog-bar-bg">
      <div class="quiz-prog-bar-fill" style="width:{pct}%"></div>
    </div>
  </div>
  <div class="quiz-step-badge">Question {qi+1} of {total}</div>
</div>""", unsafe_allow_html=True)

    # ── All answered → predict ──
    if qi >= total:
        st.markdown("""
<div class="card card-purple" style="text-align:center;padding:3rem">
  <div style="font-size:3.5rem;margin-bottom:1rem">🧠</div>
  <div style="font-family:var(--serif);font-size:1.8rem;color:var(--snow);margin-bottom:.5rem">Assessment Complete</div>
  <div style="font-size:14px;color:var(--muted)">Analysing your cognitive profile with NextStep AI…</div>
</div>""", unsafe_allow_html=True)

        with st.spinner("Running ML prediction on your profile…"):
            time.sleep(0.9)
            feat = quiz_to_features(st.session_state.answers)
            if model:
                career, conf, top3 = predict(model, feat)
            else:
                # Demo fallback — model.pkl missing
                career, conf = "software developer", 73.6
                top3 = [("software developer", 73.6), ("data scientist", 15.8), ("web developer", 7.2)]

        st.session_state.career      = career
        st.session_state.confidence  = conf
        st.session_state.top3        = top3
        st.session_state.personality = infer_personality(st.session_state.answers)
        st.session_state.explanation = generate_explanation(career, st.session_state.answers)

        st.success("✅  Your AI projection is ready! Go to **AI Projections** in the sidebar.")
        if st.button("View My Results →"):
            st.session_state.page = "results"
            st.rerun()
        return

    # ── Current question ──
    q = QUESTIONS[qi]
    st.markdown(f"""
<div class="card">
  <div class="quiz-q-num">Assessment Pulse · Question {qi+1}</div>
  <div class="quiz-q-text">{q["text"]}</div>
</div>""", unsafe_allow_html=True)

    labels = ["A", "B", "C", "D"]
    for oi, opt in enumerate(q["options"]):
        lc, bc = st.columns([0.055, 0.945])
        with lc:
            st.markdown(f"""
<div style="margin-top:9px;font-family:var(--mono);font-size:11px;
            color:var(--purple2);padding:6px 0;font-weight:600">{labels[oi]}</div>""",
                        unsafe_allow_html=True)
        with bc:
            if st.button(opt, key=f"q{qi}_opt{oi}", use_container_width=True):
                st.session_state.answers.append(oi)
                st.session_state.q_index += 1
                st.rerun()

    if qi > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", key="back_btn"):
            st.session_state.q_index = max(0, qi - 1)
            if st.session_state.answers:
                st.session_state.answers.pop()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════
def page_results():
    if not st.session_state.get("career"):
        st.markdown("""
<div class="card" style="text-align:center;padding:3.5rem">
  <div style="font-size:3.5rem;margin-bottom:1.2rem">📝</div>
  <div style="font-family:var(--serif);font-size:1.6rem;color:var(--snow);margin-bottom:.5rem">No Results Yet</div>
  <div style="font-size:14px;color:var(--muted)">Complete the Neural Assessment to see your AI career projection.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Take the Assessment →"):
            st.session_state.page = "quiz"
            st.rerun()
        return

    career  = st.session_state.career
    conf    = st.session_state.confidence
    top3    = st.session_state.get("top3", [])
    persona = st.session_state.get("personality", [])
    expl    = st.session_state.get("explanation", "")
    icon    = career_icon(career)
    conf_pct = conf if conf else 0

    # ── Primary result ──
    st.markdown(f"""
<div class="card card-purple" style="text-align:center;padding:3.5rem 2rem">
  <div class="result-badge">✦ AI Projection · Top Match</div>
  <div class="result-icon">{icon}</div>
  <div class="result-career">{career.title()}</div>
  <div class="result-conf">● Neural Sync: {conf_pct}% confidence</div>
  <div class="result-expl">{expl}</div>
</div>""", unsafe_allow_html=True)

    # ── Confidence ring + Personality ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Neural Sync Score</div>', unsafe_allow_html=True)

        # SVG confidence ring
        r   = 52
        circ = 2 * 3.14159 * r
        dash = (conf_pct / 100) * circ if conf_pct else 0
        ring_color = "#10B981" if conf_pct >= 70 else ("#F59E0B" if conf_pct >= 45 else "#F43F5E")
        st.markdown(f"""
<div class="conf-ring-wrap">
  <div class="conf-ring-outer" style="width:130px;height:130px;position:relative">
    <svg class="conf-ring-svg" viewBox="0 0 120 120" style="position:absolute;inset:0;transform:rotate(-90deg)">
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8"/>
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="{ring_color}" stroke-width="8"
              stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"/>
    </svg>
    <div class="conf-ring-inner">
      <span class="conf-num">{int(conf_pct) if conf_pct else "–"}</span>
      <span class="conf-unit">% sync</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card card-purple">', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Personality Profile</div>', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="personality-chip {cls}">{em} {lbl}</span>'
            for lbl, cls, em in persona
        )
        st.markdown(f'<div style="margin-top:.8rem;line-height:2.2">{chips}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top 3 ──
    if top3:
        st.markdown('<div class="section-eyebrow" style="margin-top:0.5rem">AI Projections</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your Top 3 Matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Ranked by your neural profile match score based on current trajectory.</div>', unsafe_allow_html=True)

        max_p = top3[0][1] if top3 else 1
        labels = ["TOP MATCH", "ALTERNATIVE", "ALTERNATIVE"]
        for rank, (name, pct) in enumerate(top3):
            bw   = int((pct / max(max_p, 1)) * 100)
            lbl  = labels[min(rank, 2)]
            icn  = career_icon(name)
            st.markdown(f"""
<div class="top3-row" style="animation-delay:{rank*0.12}s">
  <span class="top3-rank">#{rank+1}</span>
  <span class="top3-icon">{icn}</span>
  <div style="flex:1;margin-left:12px">
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:3px">{lbl}</div>
    <span class="top3-name">{name.title()}</span>
  </div>
  <div class="top3-bar-bg"><div class="top3-bar-fill" style="width:{bw}%"></div></div>
  <span class="top3-pct">{pct}</span>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    # ── Roadmap ──
    st.markdown('<div class="section-eyebrow">The Path Forward</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Your {career.title()} Roadmap</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">A structured, phase-by-phase guide built for your neural profile and career trajectory.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_roadmap(career)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("💬  AI Mentor", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
    with r2:
        if st.button("📄  Resume Analyzer", use_container_width=True):
            st.session_state.page = "resume"; st.rerun()
    with r3:
        if st.button("🔄  Retake Assessment", use_container_width=True):
            for k in ["career","confidence","top3","personality","explanation","answers","q_index"]:
                st.session_state.pop(k, None)
            st.session_state.page = "quiz"; st.rerun()

# ══════════════════════════════════════════════════════════════════════
# PAGE: RESUME ANALYZER
# ══════════════════════════════════════════════════════════════════════
def page_resume():
    st.markdown('<div class="section-eyebrow">Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Score Your Resume</div>', unsafe_allow_html=True)
    career = st.session_state.get("career", "default")
    st.markdown(f'<div class="section-sub">Keyword analysis optimised for <strong style="color:var(--purple2)">{career.title()}</strong>. Paste plain text below — not PDF.</div>', unsafe_allow_html=True)

    resume_text = st.text_area(
        "Resume",
        height=260,
        placeholder="Paste your full resume text here…",
        label_visibility="collapsed",
    )

    ca, cb, _ = st.columns([1, 1, 2])
    with ca: do_analyze = st.button("🔍  Analyse Resume", use_container_width=True)
    with cb: do_clear   = st.button("🗑️  Clear",          use_container_width=True)
    if do_clear:
        st.rerun()

    if do_analyze:
        if len(resume_text.strip()) < 50:
            st.error("⚠️ Please paste at least a paragraph of your resume text.")
            return

        with st.spinner("Analysing your resume…"):
            time.sleep(0.6)
            result = analyze_resume(resume_text, career)

        score = result["score"]
        grade = result["grade"]
        grade_colors = {"A":"var(--green)","B":"var(--blue2)","C":"var(--amber)","D":"var(--rose)"}
        gc = grade_colors.get(grade, "var(--purple2)")
        grade_bg = {
            "A":"rgba(16,185,129,0.12)","B":"rgba(59,130,246,0.12)",
            "C":"rgba(245,158,11,0.12)","D":"rgba(244,63,94,0.12)"
        }.get(grade, "rgba(139,92,246,0.12)")
        grade_border = {
            "A":"rgba(16,185,129,0.3)","B":"rgba(59,130,246,0.3)",
            "C":"rgba(245,158,11,0.3)","D":"rgba(244,63,94,0.3)"
        }.get(grade, "rgba(139,92,246,0.3)")

        cs, ck = st.columns([1, 2])
        with cs:
            st.markdown(f"""
<div class="card resume-score-card">
  <div style="font-family:var(--mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:1rem">Neural Resume Score</div>
  <div class="resume-grade" style="color:{gc}">{score}</div>
  <div class="resume-score-sub">/ 100</div>
  <div class="grade-badge" style="background:{grade_bg};border:1px solid {grade_border};color:{gc}">Grade {grade}</div>
</div>""", unsafe_allow_html=True)

        with ck:
            matched = result["matched_kws"]
            missing = result["missing_kws"]
            mh = "".join(f'<span class="kw-chip kw-found">{k}</span>' for k in matched)
            nh = "".join(f'<span class="kw-chip kw-missing">{k}</span>' for k in missing[:7])
            st.markdown(f"""
<div class="card">
  <div style="margin-bottom:1.2rem">
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--green);margin-bottom:.7rem">✓ Keywords Found ({len(matched)})</div>
    <div>{mh if mh else '<span style="color:var(--dim);font-size:13px">None detected</span>'}</div>
  </div>
  <div>
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--rose);margin-bottom:.7rem">✗ Keywords Missing ({len(missing)})</div>
    <div>{nh if nh else '<span style="color:var(--dim);font-size:13px">All key terms present!</span>'}</div>
  </div>
</div>""", unsafe_allow_html=True)

        icon_map = {
            "warning":"⚠️","chart":"📊","link":"🔗",
            "rocket":"🚀","doc":"📄","grad":"🎓",
            "check":"✅","phone":"📞",
        }
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow" style="margin-bottom:1rem">Improvement Suggestions</div>', unsafe_allow_html=True)
        for key, tip in result["suggestions"]:
            ic = icon_map.get(key, "💡")
            st.markdown(f'<div class="suggestion-row"><span class="sug-icon">{ic}</span><span>{tip}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ══════════════════════════════════════════════════════════════════════
def page_chat():
    career = st.session_state.get("career", "")

    st.markdown('<div class="section-eyebrow">AI Mentor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">NextStep AI Career Mentor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Your personal AI mentor, calibrated to <strong style="color:var(--purple2)">'
        f'{career.title() if career else "general career guidance"}'
        f'</strong>. Ask anything — skills, roadmap, interviews, salary.</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        greeting = (
            f"Hello! I am your NextStep AI Career Mentor, and I can see your predicted career is "
            f"**{career.title()}**. I am here to help you navigate your journey. "
            f"Ask me anything — skills to learn, how to get started, project ideas, interview tips, "
            f"or salary expectations. What would you like to explore first?"
            if career else
            "Hello! I am your NextStep AI Career Mentor. "
            "Complete the Neural Assessment to get personalised career advice, "
            "or feel free to ask me a general career guidance question right now!"
        )
        st.session_state.chat_history.append(("bot", greeting))

    # Suggested prompts
    st.markdown('<div class="chat-suggestions">', unsafe_allow_html=True)
    suggestions = [
        "What skills do I need?",
        "How do I get started?",
        "Suggest 3 projects",
        "Interview tips",
        "What salary can I expect?",
    ]
    scols = st.columns(len(suggestions))
    for i, (col, sug) in enumerate(zip(scols, suggestions)):
        with col:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append(("user", sug))
                st.session_state.chat_history.append(("bot", chatbot_response(sug, career or "default")))
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Conversation history
    for role, msg in st.session_state.chat_history:
        meta = "You" if role == "user" else "◈ NextStep AI Mentor"
        st.markdown(f"""
<div class="chat-bubble {role}">
  <div class="chat-meta">{meta}</div>
  <div class="chat-text">{msg}</div>
</div>""", unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        ic, sc = st.columns([5, 1])
        with ic:
            user_input = st.text_input(
                "Message",
                placeholder="Ask your career mentor anything…",
                label_visibility="collapsed",
            )
        with sc:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append(("user", user_input.strip()))
        reply = chatbot_response(user_input.strip(), career or "default")
        st.session_state.chat_history.append(("bot", reply))
        st.rerun()

    if st.button("🗑️  Clear Conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # Initialise session state
    defaults = [("page","home"),("career",""),("answers",[]),("q_index",0)]
    for key, val in defaults:
        if key not in st.session_state:
            st.session_state[key] = val

    # Load model (career_model.pkl)
    model, err = load_model()
    if err and st.session_state.page not in ["home"]:
        st.warning(f"⚠️ {err}  Running in demo mode — predictions are illustrative.")

    render_sidebar()

    page = st.session_state.page
    if   page == "home":    page_home()
    elif page == "quiz":    page_quiz(model)
    elif page == "results": page_results()
    elif page == "resume":  page_resume()
    elif page == "chat":    page_chat()

    st.markdown("""
<div class="footer">
  NextStep AI · Your Smart Career Guide to the Next Step<br>
  <span style="opacity:0.45">Built with ❤️ using Machine Learning &amp; Streamlit · Zero External APIs</span>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()