"""
╔══════════════════════════════════════════════════════════════════╗
║           AI CAREER PREDICTOR — Production ML Web App           ║
║           Built with Streamlit + scikit-learn RandomForest       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import pickle
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All 23 features the model was trained on (in exact order)
ALL_FEATURES = [
    "Logical quotient rating",
    "coding skills rating",
    "hackathons",
    "public speaking points",
    "self-learning capability?",
    "Extra-courses did",
    "certifications",
    "workshops",
    "reading and writing skills",
    "memory capability score",
    "Interested subjects",
    "interested career area",
    "Job/Higher Studies?",
    "Type of company want to settle in?",
    "Taken inputs from seniors or elders",
    "Interested Type of Books",
    "Management or Technical",
    "hard/smart worker",
    "worked in teams ever?",
    "Introvert",
    "academic percentage",
    "Math",
    "Soft skills",
]

# Sensible defaults for features not surfaced in the UI
FEATURE_DEFAULTS = {
    "Logical quotient rating": 7,
    "coding skills rating": 6,
    "hackathons": 2,
    "public speaking points": 6,
    "self-learning capability?": 1,
    "Extra-courses did": 1,
    "certifications": 3,
    "workshops": 2,
    "reading and writing skills": 7,
    "memory capability score": 7,
    "Interested subjects": 5,
    "interested career area": 3,
    "Job/Higher Studies?": 1,
    "Type of company want to settle in?": 2,
    "Taken inputs from seniors or elders": 1,
    "Interested Type of Books": 3,
    "Management or Technical": 1,
    "hard/smart worker": 1,
    "worked in teams ever?": 1,
    "Introvert": 0,
    "academic percentage": 75,
    "Math": 7,
    "Soft skills": 7,
}

# Career-to-emoji mapping for richer results display
CAREER_ICONS = {
    "software developer": "💻",
    "data scientist": "📊",
    "web developer": "🌐",
    "machine learning engineer": "🤖",
    "database administrator": "🗄️",
    "network engineer": "🔌",
    "cybersecurity analyst": "🔐",
    "cloud engineer": "☁️",
    "devops engineer": "⚙️",
    "ai engineer": "🧠",
    "business analyst": "📈",
    "product manager": "🎯",
    "ux designer": "🎨",
    "mobile developer": "📱",
    "blockchain developer": "⛓️",
    "default": "🚀",
}

SAMPLE_INPUTS = {
    "academic_pct": 85,
    "math": 9,
    "logical": 8,
    "coding": 9,
    "soft_skills": 7,
    "hackathons": 3,
    "public_speaking": 6,
    "self_learning": 1,
    "extra_courses": 1,
    "certifications": 4,
    "workshops": 3,
    "memory": 8,
    "reading_writing": 7,
    "hard_smart": 1,
    "team_work": 1,
    "introvert": 0,
    "management_technical": 1,
    "job_higher": 1,
}

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; }

    :root {
        --bg-deep:      #030712;
        --bg-card:      rgba(17, 24, 39, 0.65);
        --bg-raised:    rgba(31, 41, 55, 0.7);
        --border:       rgba(255, 255, 255, 0.08);
        --border-hover: rgba(56, 189, 248, 0.4);
        --accent:       #0ea5e9;
        --accent-glow:  rgba(14, 165, 233, 0.5);
        --accent-warm:  #f59e0b;
        --accent-green: #10b981;
        --text-primary: #f8fafc;
        --text-muted:   #94a3b8;
        --text-dim:     #64748b;
        --mono:         'DM Mono', monospace;
        --serif:        'DM Serif Display', serif;
        --sans:         'Outfit', 'Inter', sans-serif;
    }

    /* ── Dynamic Animated Background ── */
    html, body {
        background-color: var(--bg-deep) !important;
        font-family: var(--sans) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 15% 50%, rgba(14, 165, 233, 0.12), transparent 45%),
                    radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.12), transparent 45%),
                    radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.08), transparent 50%),
                    var(--bg-deep) !important;
        background-attachment: fixed !important;
        animation: bgDrift 20s ease-in-out infinite alternate !important;
    }

    @keyframes bgDrift {
        0%   { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }

    [data-testid="stHeader"]  { background: transparent !important; }
    [data-testid="stToolbar"] { display: none !important; }
    section.main > div        { padding-top: 0 !important; }

    /* ── Layout cap ── */
    .block-container { max-width: 1140px !important; padding: 2.5rem 2rem 5rem !important; margin: auto !important; }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 6rem 1rem 4rem;
        position: relative;
        z-index: 10;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: -20% -10%;
        background: radial-gradient(circle, rgba(14,165,233,0.05) 0%, transparent 60%);
        z-index: -1;
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-family: var(--mono);
        font-size: 12px;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 100px;
        padding: 8px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.2);
        animation: fadeDown 0.7s ease-out forwards;
    }
    .hero-eyebrow .dot {
        width: 8px; height: 8px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
        animation: pulseDot 2s infinite;
    }
    .hero-title {
        font-family: var(--serif);
        font-size: clamp(3.5rem, 8vw, 6rem);
        font-weight: 500;
        line-height: 1.1;
        letter-spacing: -2px;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        animation: fadeDown 0.7s 0.15s ease-out both;
        text-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-title span { 
        background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-style: italic; 
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-muted);
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto 3.5rem;
        line-height: 1.8;
        animation: fadeDown 0.7s 0.3s ease-out both;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3.5rem;
        flex-wrap: wrap;
        animation: fadeDown 0.7s 0.45s ease-out both;
    }
    .stat { 
        text-align: center;
        padding: 1.2rem 2rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stat:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3), 0 0 15px rgba(56, 189, 248, 0.1);
    }
    .stat-number { font-family: var(--serif); font-size: 2.5rem; color: var(--text-primary); line-height: 1; margin-bottom: 0.5rem; }
    .stat-label  { font-family: var(--mono); font-size: 11px; letter-spacing: 2px; color: var(--text-dim); text-transform: uppercase; }

    /* ── Section labels ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
    }
    .section-icon {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(139,92,246,0.15));
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .section-title {
        font-family: var(--sans);
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
    }
    .section-sub {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 400;
        margin-top: 2px;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent);
    }

    /* ── Advanced Glassmorphism Cards ── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2.5rem 2.5rem;
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), rgba(139,92,246,0.3), transparent);
        opacity: 0.5;
        transition: opacity 0.4s ease;
    }
    .glass-card:hover { 
        border-color: rgba(56,189,248,0.3);
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.1);
    }
    .glass-card:hover::before { opacity: 1; }

    /* ── Form Inputs ── */
    [data-testid="stSlider"] label,
    [data-testid="stSelectbox"] label {
        font-family: var(--sans) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
        padding: 0.2rem 0.5rem !important;
    }
    [data-testid="stSelectbox"] > div > div:hover,
    [data-testid="stSelectbox"] > div > div:focus-within {
        border-color: rgba(56,189,248,0.5) !important;
        background: rgba(255,255,255,0.07) !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] > div > div > div > div {
        background: rgba(255,255,255,0.1) !important;
        height: 6px !important;
        border-radius: 10px !important;
    }
    [data-testid="stSlider"] > div > div > div > div > div { /* Fill bar */
        background: linear-gradient(90deg, #38bdf8, #8b5cf6) !important;
    }
    [data-testid="stSlider"] > div > div > div > div > div[role="slider"] { /* Thumb */
        background: #fff !important;
        box-shadow: 0 0 15px rgba(56,189,248,0.8), 0 0 5px rgba(255,255,255,0.5) !important;
        width: 22px !important; height: 22px !important;
        top: -8px !important;
        border: 2px solid #38bdf8 !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="stSlider"] > div > div > div > div > div[role="slider"]:hover {
        transform: scale(1.2) !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--text-dim) !important;
        font-family: var(--mono) !important;
        font-size: 12px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: var(--sans) !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border-radius: 16px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        position: relative;
        overflow: hidden;
    }
    /* Primary predict button */
    div[data-testid="column"]:first-child .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: #fff !important;
        box-shadow: 0 8px 30px rgba(14,165,233,0.4) !important;
        padding: 1.2rem 2.5rem !important;
        width: 100% !important;
    }
    div[data-testid="column"]:first-child .stButton > button::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transform: skewX(-20deg);
        transition: 0.5s;
    }
    div[data-testid="column"]:first-child .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(14,165,233,0.6) !important;
    }
    div[data-testid="column"]:first-child .stButton > button:hover::after {
        left: 150%;
    }
    
    /* Secondary buttons */
    div[data-testid="column"]:not(:first-child) .stButton > button {
        background: rgba(255,255,255,0.03) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 1.2rem 2rem !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-testid="column"]:not(:first-child) .stButton > button:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.2) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2) !important;
    }

    /* ── Result card ── */
    .result-wrap {
        background: linear-gradient(145deg, rgba(14,165,233,0.08), rgba(99,102,241,0.08));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 28px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin-top: 2rem;
        animation: blastIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 0 40px rgba(14,165,233,0.1);
    }
    .result-wrap::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(14,165,233,0.1) 0%, transparent 60%);
        animation: spinBg 20s linear infinite;
        pointer-events: none;
    }
    .result-wrap::after {
        content: '';
        position: absolute;
        inset: 0;
        background: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M2 0h2v2H2V0zm0 4h2v2H2V4zm0 4h2v2H2V8zm0 4h2v2H2v-2zm0 4h2v2H2v-2zM6 0h2v2H6V0zm0 4h2v2H6V4zm0 4h2v2H6V8zm0 4h2v2H6v-2zm0 4h2v2H6v-2zM10 0h2v2h-2V0zm0 4h2v2h-2V4zm0 4h2v2h-2V8zm0 4h2v2h-2v-2zm0 4h2v2h-2v-2z' fill='rgba(255,255,255,0.015)' fill-rule='evenodd'/%3E%3C/svg%3E");
        opacity: 0.8;
        pointer-events: none;
    }
    .result-rank-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: var(--mono);
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #fff;
        background: linear-gradient(135deg, #10b981, #059669);
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 100px;
        padding: 6px 18px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
        position: relative;
        z-index: 2;
    }
    .result-icon { 
        font-size: 5.5rem; 
        margin-bottom: 1rem; 
        line-height: 1;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.3));
        position: relative;
        z-index: 2;
        animation: floatIcon 3s ease-in-out infinite;
    }
    .result-career-name {
        font-family: var(--serif);
        font-size: clamp(2.5rem, 5vw, 3.5rem);
        font-weight: 500;
        color: #fff;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        position: relative;
        z-index: 2;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .result-confidence {
        font-family: var(--mono);
        font-size: 15px;
        color: #34d399;
        margin-bottom: 1.2rem;
        background: rgba(16,185,129,0.1);
        padding: 4px 16px;
        border-radius: 100px;
        display: inline-block;
        position: relative;
        z-index: 2;
    }
    .result-explanation {
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.8;
        position: relative;
        z-index: 2;
    }

    /* ── Top-3 alt careers ── */
    .alt-career {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.3s ease;
        animation: slideUpFade 0.5s ease both;
    }
    .alt-career:hover { 
        border-color: rgba(56,189,248,0.3); 
        transform: translateX(8px) translateY(-2px);
        background: rgba(255,255,255,0.04);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .alt-career-left { display: flex; align-items: center; gap: 16px; }
    .alt-rank { 
        font-family: var(--mono); 
        font-size: 13px; 
        color: var(--text-dim); 
        width: 28px;
        height: 28px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
    }
    .alt-name { font-size: 16px; font-weight: 600; color: #f1f5f9; }
    .alt-bar-wrap { flex: 1; margin: 0 1.5rem; }
    .alt-bar {
        height: 6px;
        background: linear-gradient(90deg, #38bdf8, #8b5cf6);
        border-radius: 100px;
        transition: width 1s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 0 10px rgba(56,189,248,0.3);
    }
    .alt-pct { 
        font-family: var(--mono); 
        font-size: 14px; 
        font-weight: 600;
        color: #38bdf8; 
    }

    /* ── Feature importance ── */
    .feat-bar-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .feat-label { font-size: 13px; font-weight: 500; color: #cbd5e1; width: 180px; flex-shrink: 0; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
    .feat-bar-bg { flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
    .feat-bar-fill { 
        height: 100%; 
        background: linear-gradient(90deg, #f59e0b, #ef4444); 
        border-radius: 100px; 
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
    }
    .feat-value { font-family: var(--mono); font-size: 12px; color: #94a3b8; width: 45px; text-align: right; }

    /* ── Divider ── */
    .hdivider { 
        height: 2px; 
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent); 
        margin: 3rem 0; 
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text-dim);
        letter-spacing: 2px;
        line-height: 2.2;
    }
    .footer a { color: #38bdf8; text-decoration: none; transition: color 0.2s; }
    .footer a:hover { color: #fff; text-shadow: 0 0 8px #38bdf8; }

    /* ── Streamlit alert overrides ── */
    [data-testid="stAlert"] {
        border-radius: 16px !important;
        font-family: var(--sans) !important;
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* ── Animations ── */
    @keyframes fadeDown  { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: none; } }
    @keyframes slideUpFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: none; } }
    @keyframes blastIn   { 0% { opacity: 0; transform: scale(0.8) translateY(30px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
    @keyframes pulseDot  { 0% { box-shadow: 0 0 0 0 rgba(56,189,248, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(56,189,248, 0); } 100% { box-shadow: 0 0 0 0 rgba(56,189,248, 0); } }
    @keyframes floatIcon { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
    @keyframes spinBg    { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and encoder from disk with graceful error handling."""
    model_path   = os.path.join(BASE_DIR, "career_model.pkl")
    encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

    if not os.path.exists(model_path):
        return None, None, "career_model.pkl not found in the app directory."

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    encoder = None
    if os.path.exists(encoder_path):
        try:
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
        except Exception:
            pass  # encoder optional

    return model, encoder, None


def preprocess_inputs(ui_values: dict) -> np.ndarray:
    """
    Align UI values to the exact 23-feature vector the model expects.
    Missing features are filled with sensible defaults.

    Args:
        ui_values: dict mapping feature name → value from UI

    Returns:
        numpy array of shape (1, 23)
    """
    vector = []
    for feat in ALL_FEATURES:
        val = ui_values.get(feat, FEATURE_DEFAULTS.get(feat, 0))
        vector.append(float(val))
    return np.array([vector])


def predict(model, encoder, feature_vector: np.ndarray):
    """
    Run prediction and return (top_career, confidence_pct, top3_list).
    top3_list → list of (career_name, confidence_pct) tuples.
    """
    prediction = model.predict(feature_vector)

    top_career = str(prediction[0])
    if encoder:
        try:
            top_career = encoder.inverse_transform(prediction)[0]
        except Exception:
            pass

    # Probability scores for top-3
    top3 = []
    if hasattr(model, "predict_proba"):
        probs     = model.predict_proba(feature_vector)[0]
        top3_idx  = np.argsort(probs)[::-1][:3]
        classes   = model.classes_

        for idx in top3_idx:
            name = str(classes[idx])
            if encoder:
                try:
                    name = encoder.inverse_transform([classes[idx]])[0]
                except Exception:
                    pass
            top3.append((name, round(float(probs[idx]) * 100, 1)))

        confidence = top3[0][1] if top3 else None
    else:
        confidence = None

    return top_career, confidence, top3


def get_career_icon(career_name: str) -> str:
    """Return emoji for a given career string."""
    lower = career_name.lower()
    for key, icon in CAREER_ICONS.items():
        if key in lower:
            return icon
    return CAREER_ICONS["default"]


def build_explanation(career: str, ui_values: dict) -> str:
    """Generate a human-readable explanation sentence."""
    logical  = ui_values.get("Logical quotient rating", 5)
    coding   = ui_values.get("coding skills rating", 5)
    soft     = ui_values.get("Soft skills", 5)
    academic = ui_values.get("academic percentage", 60)

    strengths = []
    if logical  >= 7: strengths.append("strong logical reasoning")
    if coding   >= 7: strengths.append("solid coding proficiency")
    if soft     >= 7: strengths.append("excellent communication")
    if academic >= 75: strengths.append("high academic performance")

    if strengths:
        return f"Based on your {', '.join(strengths[:2])}, this career aligns closely with your profile."
    return "This career aligns well with your overall skill and academic profile."


def get_feature_importances(model, top_n: int = 8):
    """Return top-N (feature_name, importance) pairs if model supports it."""
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    pairs = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


# ──────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────

def section_header(icon: str, title: str, sub: str = ""):
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div>
            <div class="section-title">{title}</div>
            {sub_html}
        </div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)


def validate_inputs(ui_values: dict) -> list[str]:
    """Return list of error messages (empty = valid)."""
    errors = []
    rating_fields = [
        "Logical quotient rating", "coding skills rating",
        "public speaking points", "reading and writing skills",
        "memory capability score", "Math", "Soft skills",
    ]
    for f in rating_fields:
        v = ui_values.get(f, 0)
        if not (0 <= v <= 10):
            errors.append(f"'{f}' must be between 0 and 10.")
    if not (0 <= ui_values.get("academic percentage", 0) <= 100):
        errors.append("Academic percentage must be between 0 and 100.")
    if not (0 <= ui_values.get("hackathons", 0) <= 20):
        errors.append("Hackathons count seems unusually high (max 20).")
    return errors


# ──────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Load model once ──
    model, encoder, load_error = load_model()

    # ── Session state for reset / sample ──
    if "reset_flag" not in st.session_state:
        st.session_state.reset_flag = False
    if "sample_flag" not in st.session_state:
        st.session_state.sample_flag = False

    defaults = SAMPLE_INPUTS if st.session_state.sample_flag else {
        "academic_pct": 70, "math": 6, "logical": 6, "coding": 6,
        "soft_skills": 6, "hackathons": 1, "public_speaking": 5,
        "self_learning": 0, "extra_courses": 0, "certifications": 2,
        "workshops": 1, "memory": 6, "reading_writing": 6,
        "hard_smart": 0, "team_work": 1, "introvert": 0,
        "management_technical": 0, "job_higher": 0,
    }

    # ──────────────────────────── HERO ────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">
            <span class="dot"></span>
            Machine Learning · Random Forest · 23 Features
        </div>
        <div class="hero-title">AI <span>Career</span> Predictor</div>
        <div class="hero-subtitle">
            Smart career recommendations powered by Machine Learning —
            fill in your profile below and discover the path built for you.
        </div>
        <div class="hero-stats">
            <div class="stat"><div class="stat-number">23</div><div class="stat-label">Input Features</div></div>
            <div class="stat"><div class="stat-number">40+</div><div class="stat-label">Career Paths</div></div>
            <div class="stat"><div class="stat-number">RF</div><div class="stat-label">Algorithm</div></div>
            <div class="stat"><div class="stat-number">Top 3</div><div class="stat-label">Predictions</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Load warning ──
    if load_error:
        st.error(f"⚠️ {load_error}  \nPlace your `.pkl` files in the same folder as `app.py` and restart.")

    # ────────────────────── SECTION 1 — ACADEMICS ─────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("🎓", "Academic Performance", "Your educational background & scores")

    col1, col2, col3 = st.columns(3)
    with col1:
        academic_pct = st.slider("Academic Percentage (%)", 0, 100,
                                  defaults["academic_pct"], step=1,
                                  help="Overall academic score across all subjects")
    with col2:
        math_score = st.slider("Mathematics Score (0–10)", 0, 10,
                                defaults["math"],
                                help="Proficiency in Mathematics")
    with col3:
        reading_writing = st.slider("Reading & Writing (0–10)", 0, 10,
                                     defaults["reading_writing"],
                                     help="Language skills")
    st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────── SECTION 2 — SKILLS ────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("⚡", "Technical & Soft Skills", "Rate yourself honestly — this drives the prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        logical    = st.slider("Logical Reasoning (0–10)", 0, 10, defaults["logical"])
        coding     = st.slider("Coding Skills (0–10)",     0, 10, defaults["coding"])
    with col2:
        soft       = st.slider("Soft Skills (0–10)",       0, 10, defaults["soft_skills"])
        memory     = st.slider("Memory Capability (0–10)", 0, 10, defaults["memory"])
    with col3:
        public_sp  = st.slider("Public Speaking (0–10)",   0, 10, defaults["public_speaking"])
        hackathons = st.slider("Hackathons Attended",       0, 20, defaults["hackathons"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────── SECTION 3 — PERSONALITY ───────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("🧠", "Personality & Preferences", "Helps refine the match to your working style")

    col1, col2, col3 = st.columns(3)
    with col1:
        self_learning  = st.selectbox("Self-Learning Capability",
                                       ["Yes", "No"],
                                       index=0 if defaults["self_learning"] else 1)
        extra_courses  = st.selectbox("Completed Extra Courses",
                                       ["Yes", "No"],
                                       index=0 if defaults["extra_courses"] else 1)
        certifications = st.slider("Certifications Earned", 0, 10, defaults["certifications"])
    with col2:
        workshops      = st.slider("Workshops Attended", 0, 10, defaults["workshops"])
        team_work      = st.selectbox("Worked in Teams",
                                       ["Yes", "No"],
                                       index=0 if defaults["team_work"] else 1)
        introvert      = st.selectbox("Personality Type",
                                       ["Extrovert", "Introvert"],
                                       index=1 if defaults["introvert"] else 0)
    with col3:
        hard_smart     = st.selectbox("Work Style",
                                       ["Smart Worker", "Hard Worker"],
                                       index=0 if defaults["hard_smart"] else 1)
        mgmt_tech      = st.selectbox("Preference",
                                       ["Technical", "Management"],
                                       index=0 if defaults["management_technical"] else 1)
        job_higher     = st.selectbox("After Graduation",
                                       ["Job", "Higher Studies"],
                                       index=0 if defaults["job_higher"] else 1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────── ACTION BUTTONS ────────────────────────
    divider()
    btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])

    with btn_col1:
        predict_btn = st.button("🎯  Predict My Career Path", use_container_width=True)
    with btn_col2:
        sample_btn  = st.button("✦  Sample Input",            use_container_width=True)
    with btn_col3:
        reset_btn   = st.button("↺  Reset",                   use_container_width=True)

    # Handle sample / reset toggles
    if sample_btn:
        st.session_state.sample_flag = True
        st.session_state.reset_flag  = False
        st.rerun()

    if reset_btn:
        st.session_state.sample_flag = False
        st.session_state.reset_flag  = True
        st.rerun()

    # ────────────────────── PREDICTION LOGIC ──────────────────────
    if predict_btn:
        # Build ui_values dict
        ui_values = {
            "Logical quotient rating":    float(logical),
            "coding skills rating":       float(coding),
            "hackathons":                 float(hackathons),
            "public speaking points":     float(public_sp),
            "self-learning capability?":  1.0 if self_learning == "Yes" else 0.0,
            "Extra-courses did":          1.0 if extra_courses  == "Yes" else 0.0,
            "certifications":             float(certifications),
            "workshops":                  float(workshops),
            "reading and writing skills": float(reading_writing),
            "memory capability score":    float(memory),
            "Management or Technical":    1.0 if mgmt_tech == "Technical" else 0.0,
            "hard/smart worker":          1.0 if hard_smart == "Smart Worker" else 0.0,
            "worked in teams ever?":      1.0 if team_work == "Yes" else 0.0,
            "Introvert":                  1.0 if introvert == "Introvert" else 0.0,
            "academic percentage":        float(academic_pct),
            "Math":                       float(math_score),
            "Soft skills":                float(soft),
            "Job/Higher Studies?":        1.0 if job_higher == "Job" else 0.0,
        }

        # Validate
        errors = validate_inputs(ui_values)
        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
            return

        if not model:
            st.error("⚠️ Model not loaded. Check that `career_model.pkl` is present.")
            return

        # Predict with spinner
        with st.spinner("Analysing your profile…"):
            time.sleep(0.6)  # subtle UX delay for loading feel
            feature_vector = preprocess_inputs(ui_values)
            top_career, confidence, top3 = predict(model, encoder, feature_vector)

        explanation = build_explanation(top_career, ui_values)
        icon        = get_career_icon(top_career)
        conf_str    = f"{confidence}% confidence" if confidence else "Prediction complete"

        # ── Primary result card ──
        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-rank-badge">✦ Top Match</div>
            <div class="result-icon">{icon}</div>
            <div class="result-career-name">{top_career}</div>
            <div class="result-confidence">● {conf_str}</div>
            <div class="result-explanation">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top 3 alternatives ──
        if top3 and len(top3) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("📊", "Top 3 Career Matches", "Ranked by prediction probability")

            for rank, (name, pct) in enumerate(top3, 1):
                alt_icon  = get_career_icon(name)
                bar_width = int(pct)
                st.markdown(f"""
                <div class="alt-career" style="animation-delay:{rank*0.1}s">
                    <div class="alt-career-left">
                        <span class="alt-rank">#{rank}</span>
                        <span style="font-size:20px">{alt_icon}</span>
                        <span class="alt-name">{name}</span>
                    </div>
                    <div class="alt-bar-wrap">
                        <div class="alt-bar" style="width:{bar_width}%"></div>
                    </div>
                    <span class="alt-pct">{pct}%</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Feature importance ──
        importances = get_feature_importances(model)
        if importances:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🔍", "Feature Importance", "What the model valued most in your prediction")

            max_imp = importances[0][1]
            for feat_name, imp in importances:
                bar_pct = int((imp / max_imp) * 100)
                st.markdown(f"""
                <div class="feat-bar-wrap">
                    <span class="feat-label">{feat_name}</span>
                    <div class="feat-bar-bg">
                        <div class="feat-bar-fill" style="width:{bar_pct}%"></div>
                    </div>
                    <span class="feat-value">{imp:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────── FOOTER ──────────────────────────
    divider()
    st.markdown("""
    <div class="footer">
        BUILT WITH ❤️ USING MACHINE LEARNING &amp; STREAMLIT<br>
        <span style="opacity:0.4">RandomForest · scikit-learn · Python</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()