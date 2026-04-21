"""
╔══════════════════════════════════════════════════════════════════════╗
║  NextStep AI — Your Smart Career Guide to the Next Step              ║
║  Features: Quiz · ML Prediction · Roadmap · Mentor · Resume Scan     ║
║  Model   : career_model.pkl  |  Proprietary ML Engine                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, time, pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextStep AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ── Design Tokens: 3D Immersive & Neon Glassmorphism ── */
:root {
  /* Backgrounds */
  --bg-deep:      #05050A;
  --bg-space:     #0A0A12;
  
  /* Glass & Borders */
  --glass-base:   rgba(20, 25, 40, 0.45);
  --glass-hi:     rgba(255, 255, 255, 0.08);
  --glass-border: rgba(255, 255, 255, 0.08);
  --border-pu:    rgba(138, 43, 226, 0.4);
  --border-cy:    rgba(0, 240, 255, 0.4);
  
  /* Neon Accents */
  --cyan:         #00F0FF;
  --purple:       #8A2BE2;
  --purple2:      #B47CFF;
  --green:        #00FF66;
  --rose:         #FF0055;
  --amber:        #FFB000;
  
  /* Text */
  --snow:         #F0F4FF;
  --muted:        #8B9BB4;
  --dim:          #4B5A77;
  
  /* Gradients */
  --grad-neon:    linear-gradient(90deg, var(--cyan), var(--purple));
  --grad-fade:    linear-gradient(135deg, rgba(138,43,226,0.2), rgba(0,240,255,0.1));
  
  /* Typography */
  --display:      'Space Grotesk', sans-serif;
  --sans:         'Inter', sans-serif;
  --mono:         'Fira Code', monospace;
}

/* ── Reset & Animated Background ── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-deep) !important;
  font-family: var(--sans);
  color: var(--snow);
}

@keyframes slowPan {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

[data-testid="stAppViewContainer"] {
  background: 
    radial-gradient(circle at 15% 50%, rgba(138,43,226,0.12), transparent 30%),
    radial-gradient(circle at 85% 30%, rgba(0,240,255,0.08), transparent 30%),
    linear-gradient(-45deg, #05050A, #0A0B15, #10081A, #05050A) !important;
  background-size: 400% 400% !important;
  animation: slowPan 25s ease infinite !important;
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1000px !important; }
h1, h2, h3 { font-family: var(--display) !important; font-weight: 700 !important; color: var(--snow) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(5, 5, 10, 0.75) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
  border-right: 1px solid var(--glass-border) !important;
}

/* ── Nav elements ── */
.nav-brand {
  padding: 1.8rem 1.2rem 1rem;
  border-bottom: 1px solid var(--glass-border);
  margin-bottom: 0.8rem;
}
.nav-logo {
  display: flex; align-items: center; gap: 12px;
  font-family: var(--display); font-size: 1.25rem; font-weight: 700;
  letter-spacing: -0.3px; color: var(--snow); margin-bottom: 4px;
}
.nav-logo-icon {
  width: 32px; height: 32px; border-radius: 8px;
  background: var(--grad-neon);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
  box-shadow: 0 0 12px var(--border-pu);
}
.nav-tagline  { font-size: 11px; color: var(--muted); padding-left: 44px; line-height: 1.5; font-family: var(--sans); }
.nav-slabel   { font-family: var(--mono); font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); padding: 0 1.2rem; margin: 1.2rem 0 8px; }
.nav-div      { height: 1px; background: var(--glass-border); margin: 1rem 0.8rem; }
.nav-status   { margin: 0.8rem; padding: 14px; background: var(--glass-base); border: 1px solid var(--glass-border); border-radius: 12px; box-shadow: inset 0 0 20px rgba(0,0,0,0.5); }
.nav-s-career { font-family: var(--display); font-size: 14px; font-weight: 600; color: var(--cyan); margin-bottom: 4px; text-shadow: 0 0 10px var(--border-cy); }
.nav-s-label  { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }
.nav-footer   { margin-top: 2rem; padding-bottom: 1.2rem; text-align: center; font-family: var(--mono); font-size: 9px; letter-spacing: 2px; color: var(--dim); text-transform: uppercase; }

/* ── 3D Glass Cards (The Parallax Effect) ── */
.card {
  background: var(--glass-base);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px; 
  padding: 2rem; 
  margin-bottom: 1.5rem;
  position: relative; 
  overflow: hidden; 
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  transform-style: preserve-3d;
}

/* Diagonal glass shine effect */
.card::before {
  content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
  transform: skewX(-20deg); transition: 0.6s; z-index: 1; pointer-events: none;
}

/* Hover Physics */
.card:hover {
  transform: translateY(-6px) perspective(1000px) rotateX(2deg) rotateY(-1deg);
  border-color: rgba(138, 43, 226, 0.4); 
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 15px var(--border-pu);
}
.card:hover::before { left: 150%; }

/* Card Modifiers */
.card-pu { border-color: rgba(138,43,226,0.3) !important; background: linear-gradient(135deg, rgba(138,43,226,0.05), var(--glass-base)); }
.card-bl { border-color: rgba(0,240,255,0.3) !important; background: linear-gradient(135deg, rgba(0,240,255,0.05), var(--glass-base)); }
.card-cy { border-color: rgba(0,255,102,0.2) !important; background: linear-gradient(135deg, rgba(0,255,102,0.05), var(--glass-base)); }
.card-gr { border-color: rgba(255,176,0,0.2) !important; background: linear-gradient(135deg, rgba(255,176,0,0.05), var(--glass-base)); }
.card-ro { border-color: rgba(255,0,85,0.2) !important; background: linear-gradient(135deg, rgba(255,0,85,0.05), var(--glass-base)); }
.card-am { border-color: rgba(255,176,0,0.2) !important; background: linear-gradient(135deg, rgba(255,176,0,0.05), var(--glass-base)); }

/* ── Hero ── */
@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-12px); }
  100% { transform: translateY(0px); }
}

.hero { text-align: center; padding: 6rem 1rem 4rem; position: relative; animation: float 6s ease-in-out infinite; }

.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 11px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--purple2);
  background: rgba(138, 43, 226, 0.1); border: 1px solid var(--border-pu);
  border-radius: 100px; padding: 8px 20px; margin-bottom: 2rem;
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.15);
}

@keyframes pulse-dot { 0%,100%{opacity:1; box-shadow: 0 0 8px var(--cyan);} 50%{opacity:.4; box-shadow: none;} }
.hero-eyebrow .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); animation: pulse-dot 2s infinite; }

.hero-title {
  font-family: var(--display); font-size: clamp(3.2rem, 7vw, 5.5rem);
  font-weight: 700; line-height: 1.1; letter-spacing: -2px;
  color: var(--snow); margin-bottom: 1.5rem;
}
.hero-title em { font-style: normal; color: var(--cyan); text-shadow: 0 0 20px var(--border-cy); }
.hero-grad-text {
  background: var(--grad-neon); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; background-clip: text;
  text-shadow: 0 0 30px rgba(0, 240, 255, 0.3);
}

.hero-sub { font-size: 1.15rem; color: var(--muted); font-weight: 400; max-width: 580px; margin: 0 auto 3rem; line-height: 1.7; }
.hero-pills { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
.hero-pill  { 
  display: flex; align-items: center; gap: 8px; background: var(--glass-base); 
  border: 1px solid var(--glass-border); border-radius: 100px; padding: 8px 18px; 
  font-size: 13px; color: var(--snow); font-weight: 500;
  transition: all 0.3s;
}
.hero-pill:hover {
  transform: translateY(-2px); border-color: var(--cyan);
  box-shadow: 0 5px 15px rgba(0,240,255,0.2); color: var(--cyan);
}

/* ── Section labels ── */
.s-eye   { font-family: var(--mono); font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--cyan); margin-bottom: .5rem; text-shadow: 0 0 8px rgba(0,240,255,0.4); }
.s-title { font-family: var(--display); font-size: 2.2rem; font-weight: 700; color: var(--snow); margin-bottom: .6rem; line-height: 1.15; letter-spacing: -0.5px; text-shadow: 0 0 15px rgba(255,255,255,0.1); }
.s-sub   { font-size: 15px; color: var(--muted); margin-bottom: 2rem; line-height: 1.7; font-weight: 300; }

/* ── Quiz ── */
.quiz-hdr {
  background: var(--glass-base); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border); border-radius: 16px;
  padding: 1.2rem 1.8rem; margin-bottom: 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.quiz-prog-label { font-family: var(--mono); font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.quiz-prog-bg    { width: 260px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 100px; overflow: hidden; }
.quiz-prog-fill  { height: 100%; background: var(--grad-neon); border-radius: 100px; transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 10px var(--cyan); }
.quiz-step-badge { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--cyan); background: rgba(0,240,255,0.1); border: 1px solid rgba(0,240,255,0.3); border-radius: 100px; padding: 6px 16px; box-shadow: 0 0 10px rgba(0,240,255,0.15); }
.quiz-q-num  { font-family: var(--mono); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--purple2); margin-bottom: .8rem; }
.quiz-q-text { font-family: var(--display); font-size: 1.7rem; font-weight: 600; color: var(--snow); margin-bottom: 2rem; line-height: 1.4; }

/* ── Result ── */
.result-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 11px; letter-spacing: 3px; text-transform: uppercase; font-weight: 600;
  color: var(--cyan); background: rgba(0,240,255,0.1);
  border: 1px solid rgba(0,240,255,0.3); border-radius: 100px; padding: 8px 20px; margin-bottom: 1.5rem;
  box-shadow: 0 0 15px rgba(0,240,255,0.2);
}
.result-icon   { font-size: 6rem; line-height: 1; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(255,255,255,0.2)); transform: translateZ(30px); }
.result-career { font-family: var(--display); font-size: clamp(2.5rem,6vw,4rem); font-weight: 700; color: var(--snow); margin-bottom: .5rem; text-shadow: 0 0 20px rgba(138,43,226,0.4); transform: translateZ(20px); }
.result-conf   { font-family: var(--mono); font-size: 14px; font-weight: 500; color: var(--cyan); margin-bottom: 1.2rem; text-shadow: 0 0 10px rgba(0,240,255,0.3); }
.result-expl   { font-size: 16px; color: var(--muted); max-width: 520px; margin: 0 auto; line-height: 1.8; transform: translateZ(10px); }

/* ── Personality chips ── */
.p-chip { display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 100px; font-family: var(--mono); font-size: 12px; letter-spacing: 1px; font-weight: 600; margin: 4px; transition: all 0.3s; }
.p-chip:hover { transform: translateY(-2px) scale(1.05); }
.pc-an  { background: rgba(0,240,255,0.1);  border: 1px solid rgba(0,240,255,0.3);  color: var(--cyan); box-shadow: 0 0 10px rgba(0,240,255,0.1); }
.pc-cr  { background: rgba(255,176,0,0.1);  border: 1px solid rgba(255,176,0,0.3);  color: var(--amber); box-shadow: 0 0 10px rgba(255,176,0,0.1); }
.pc-in  { background: rgba(138,43,226,0.1); border: 1px solid rgba(138,43,226,0.3); color: var(--purple2); box-shadow: 0 0 10px rgba(138,43,226,0.1); }
.pc-ex  { background: rgba(255,0,85,0.1);   border: 1px solid rgba(255,0,85,0.3);   color: var(--rose); box-shadow: 0 0 10px rgba(255,0,85,0.1); }
.pc-ld  { background: rgba(0,255,102,0.1);  border: 1px solid rgba(0,255,102,0.3);  color: var(--green); box-shadow: 0 0 10px rgba(0,255,102,0.1); }

/* ── Confidence ring ── */
.conf-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem 0; }
.conf-outer { width: 140px; height: 140px; border-radius: 50%; display: flex; align-items: center; justify-content: center; position: relative; filter: drop-shadow(0 0 15px rgba(0,240,255,0.2)); }
.conf-svg   { position: absolute; inset: 0; transform: rotate(-90deg); transition: stroke-dasharray 1s ease-out; }
.conf-inner { display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }
.conf-num   { font-family: var(--display); font-size: 3rem; font-weight: 700; color: var(--snow); line-height: 1; text-shadow: 0 0 15px var(--cyan); }
.conf-unit  { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--cyan); text-transform: uppercase; margin-top: 4px; }

/* ── Top-3 rows ── */
.t3-row  { background: var(--glass-base); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1.2rem 1.5rem; display: flex; align-items: center; margin-bottom: 1rem; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
.t3-row:hover { border-color: var(--purple); transform: translateX(8px) scale(1.02); box-shadow: -5px 5px 20px rgba(138,43,226,0.2); background: rgba(138,43,226,0.08); }
.t3-rank { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--purple2); width: 28px; flex-shrink: 0; }
.t3-name { font-family: var(--display); font-size: 15px; font-weight: 600; color: var(--snow); flex: 1; margin-left: 12px; letter-spacing: 0.5px; }
.t3-bar-bg   { width: 90px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 100px; overflow: hidden; }
.t3-bar-fill { height: 100%; border-radius: 100px; background: var(--grad-neon); box-shadow: 0 0 8px var(--cyan); }
.t3-pct  { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--cyan); margin-left: 15px; min-width: 45px; text-align: right; text-shadow: 0 0 8px rgba(0,240,255,0.3); }
.t3-icon { font-size: 20px; margin-left: 12px; filter: drop-shadow(0 0 8px rgba(255,255,255,0.2)); }

/* ── Roadmap ── */
.rm-step   { display: flex; gap: 1.5rem; margin-bottom: 2rem; }
.rm-left   { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.rm-circle {
  width: 48px; height: 48px; border-radius: 50%; background: var(--bg-space);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--display); font-size: 15px; color: var(--cyan); font-weight: 700;
  flex-shrink: 0; border: 2px solid var(--cyan); box-shadow: 0 0 15px rgba(0,240,255,0.3);
  z-index: 2; transition: all 0.3s;
}
.rm-step:hover .rm-circle { transform: scale(1.1); box-shadow: 0 0 25px rgba(0,240,255,0.6); background: rgba(0,240,255,0.1); }
.rm-line   { width: 2px; flex: 1; margin: 8px 0; min-height: 30px; background: linear-gradient(to bottom, var(--cyan), transparent); opacity: 0.5; }
.rm-body   { padding-top: 8px; flex: 1; }
.rm-tag    { font-family: var(--mono); font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--purple2); margin-bottom: 6px; }
.rm-title  { font-family: var(--display); font-size: 1.2rem; font-weight: 700; color: var(--snow); margin-bottom: 6px; }
.rm-desc   { font-size: 14px; color: var(--muted); line-height: 1.7; margin-bottom: 1rem; }
.sk-tag    { display: inline-block; background: rgba(255,255,255,0.03); border: 1px solid var(--glass-border); border-radius: 6px; padding: 4px 12px; font-family: var(--mono); font-size: 11px; color: var(--snow); margin: 3px 4px 3px 0; transition: all 0.2s; }
.sk-tag:hover { transform: translateY(-2px); border-color: var(--cyan); color: var(--cyan); box-shadow: 0 0 10px rgba(0,240,255,0.2); }
.sk-pu     { border-color: rgba(138,43,226,0.3); color: var(--purple2); }
.sk-bl     { border-color: rgba(0,240,255,0.3); color: var(--cyan); }
.sk-gr     { border-color: rgba(0,255,102,0.3); color: var(--green); }

/* ── Chat ── */
.chat-b    { max-width: 85%; margin-bottom: 1.5rem; animation: fadeUp 0.4s ease-out forwards; }
.chat-b.u  { margin-left: auto; }
.chat-meta { font-family: var(--mono); font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; padding: 0 4px; }
.chat-b.u .chat-meta { text-align: right; color: var(--cyan); }
.chat-txt  { padding: 1rem 1.4rem; border-radius: 18px; font-size: 15px; line-height: 1.7; backdrop-filter: blur(10px); }
.chat-b.u .chat-txt { background: rgba(0,240,255,0.08); border: 1px solid rgba(0,240,255,0.3); color: var(--snow); border-bottom-right-radius: 4px; box-shadow: 0 5px 15px rgba(0,240,255,0.1); }
.chat-b.bot .chat-txt { background: var(--glass-base); border: 1px solid var(--glass-border); color: var(--snow); border-bottom-left-radius: 4px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); }

/* ── Resume ── */
.sug-row  { display: flex; gap: 14px; align-items: flex-start; padding: 1rem 0; border-bottom: 1px solid var(--glass-border); font-size: 14px; color: var(--muted); line-height: 1.7; transition: background 0.3s; }
.sug-row:hover { background: rgba(255,255,255,0.02); border-radius: 8px; padding: 1rem; margin-left: -1rem; margin-right: -1rem; border-bottom-color: transparent; }
.sug-row:last-child { border-bottom: none; }
.kw-found   { display: inline-block; padding: 4px 12px; border-radius: 6px; font-family: var(--mono); font-size: 11px; font-weight: 600; margin: 2px 3px; background: rgba(0,255,102,0.1); border: 1px solid rgba(0,255,102,0.3); color: var(--green); box-shadow: 0 0 8px rgba(0,255,102,0.15); }
.kw-missing { display: inline-block; padding: 4px 12px; border-radius: 6px; font-family: var(--mono); font-size: 11px; font-weight: 600; margin: 2px 3px; background: rgba(255,0,85,0.1); border: 1px solid rgba(255,0,85,0.3); color: var(--rose); box-shadow: 0 0 8px rgba(255,0,85,0.15); }

/* ── Buttons ── */
.stButton > button {
  font-family: var(--display) !important; font-weight: 600 !important;
  font-size: 15px !important; border-radius: 12px !important; border: none !important;
  cursor: pointer !important; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
  letter-spacing: 0.5px !important; text-transform: uppercase !important;
  background: var(--grad-neon) !important; color: #fff !important;
  box-shadow: 0 6px 20px rgba(138,43,226,0.3) !important; padding: 0.8rem 1.8rem !important;
}
.stButton > button:hover { transform: translateY(-3px) scale(1.02) !important; box-shadow: 0 10px 25px rgba(0,240,255,0.5), 0 0 15px rgba(138,43,226,0.4) !important; }

/* ── Text inputs ── */
.stTextArea textarea {
  background: rgba(10,15,30,0.6) !important; border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important; color: var(--snow) !important;
  font-family: var(--mono) !important; font-size: 14px !important; transition: all 0.3s !important;
}
.stTextArea textarea:focus { border-color: var(--cyan) !important; box-shadow: 0 0 15px rgba(0,240,255,0.2) !important; outline: none !important; }
.stTextInput input {
  background: rgba(10,15,30,0.6) !important; border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important; color: var(--snow) !important;
  font-family: var(--sans) !important; transition: all 0.3s !important;
}
.stTextInput input:focus { border-color: var(--cyan) !important; box-shadow: 0 0 15px rgba(0,240,255,0.2) !important; outline: none !important; }

/* ── Misc ── */
.hdiv { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,240,255,0.2), rgba(138,43,226,0.2), transparent); margin: 3rem 0; box-shadow: 0 0 10px rgba(0,240,255,0.2); }
.footer { text-align: center; padding: 3rem 0 1.5rem; font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--dim); text-transform: uppercase; line-height: 2.4; }
[data-testid="stAlert"] { border-radius: 12px !important; font-family: var(--sans) !important; background: var(--glass-base) !important; border: 1px solid var(--glass-border) !important; backdrop-filter: blur(10px) !important; color: var(--snow) !important;}

/* ── Animations ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes fadeUp { from{opacity:0; transform:translateY(15px) scale(0.98);} to{opacity:1; transform:none;} }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# QUIZ — 15 behavioural MCQs
# ═════════════════════════════════════════════════════════════
QUESTIONS: list[dict] = [
    {
        "text": "When you face a complex problem, how do you usually approach it?",
        "options": [
            "Break it into logical steps and analyse the data carefully",
            "Brainstorm creatively and experiment with novel solutions",
            "Discuss with teammates to gather different viewpoints",
            "Follow an established process or proven best-practice framework",
        ],
    },
    {
        "text": "Which weekend activity excites you the most?",
        "options": [
            "Building a coding project or automating something tedious",
            "Designing, drawing, writing, or creating something visual",
            "Organising a community event or helping people around you",
            "Reading research, studying a new concept, or running experiments",
        ],
    },
    {
        "text": "In a group project, which role do you naturally take on?",
        "options": [
            "The builder or developer who writes the code",
            "The designer or storyteller who shapes the experience",
            "The leader or coordinator who keeps everyone aligned",
            "The researcher or analyst who digs deep into the data",
        ],
    },
    {
        "text": "How comfortable are you with Mathematics or Statistics?",
        "options": [
            "Very comfortable — I genuinely enjoy numbers and equations",
            "Comfortable with basics; advanced topics feel challenging",
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
            "I would much rather write a report than speak publicly",
            "I actively avoid public speaking whenever possible",
        ],
    },
    {
        "text": "How do you typically respond to tight deadlines and high pressure?",
        "options": [
            "I thrive — pressure sharpens my concentration",
            "I manage well as long as I have planned ahead",
            "I struggle but usually push through",
            "I find it very stressful and it hurts my output",
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
        "text": "How often do you independently learn new tools or technologies?",
        "options": [
            "Constantly — I am a self-driven, curious learner",
            "Often — especially when a project demands it",
            "Sometimes — mainly when external pressure pushes me",
            "Rarely — I stick to structured classes and formal learning only",
        ],
    },
    {
        "text": "Have you taken part in hackathons, competitions, or open-source work?",
        "options": [
            "Yes — multiple times and I genuinely enjoyed every experience",
            "Once or twice — decent and worthwhile overall",
            "Tried once but found it was not really for me",
            "No — I have never participated in any such activity",
        ],
    },
    {
        "text": "Which type of product would you find most satisfying to build?",
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
        "text": "What motivates you most when thinking about your career?",
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


# ═════════════════════════════════════════════════════════════
# CAREER ROADMAPS  (9 careers + default fallback)
# ═════════════════════════════════════════════════════════════
ROADMAPS: dict[str, list[dict]] = {
    "default": [
        {"tag": "Phase 01 · Foundations",    "title": "Core Fundamentals",       "desc": "Master the fundamental concepts of your domain. One language, one tool — go deep before going wide.",                                          "skills": ["Python / Java", "Problem Solving", "Git"],          "type": "pu"},
        {"tag": "Phase 02 · Specialisation", "title": "Domain Skills",            "desc": "Dive into the tools and frameworks your career demands. Build small projects alongside every course.",                                            "skills": ["Domain Tools", "Certifications", "Practice"],       "type": "bl"},
        {"tag": "Phase 03 · Portfolio",      "title": "Real-World Projects",      "desc": "Build 2–3 portfolio projects that demonstrate your abilities. Deploy and document every one of them.",                                            "skills": ["GitHub", "Documentation", "Deployment"],            "type": "gr"},
        {"tag": "Phase 04 · Network",        "title": "Community & Visibility",   "desc": "Engage with professionals, attend meetups, contribute to open-source, and write about your work.",                                               "skills": ["LinkedIn", "Open Source", "Blog"],                  "type": "pu"},
        {"tag": "Phase 05 · Career",         "title": "Land Your First Role",     "desc": "Apply strategically. Tailor your resume to each role and prepare deeply for domain-specific interviews.",                                        "skills": ["Resume Polish", "Mock Interviews", "LeetCode"],     "type": "bl"},
    ],
    "software developer": [
        {"tag": "Phase 01 · Foundations",    "title": "Programming Fundamentals", "desc": "Learn Python or JavaScript deeply. Understand data structures, algorithms, and OOP from first principles.",                                      "skills": ["Python", "JavaScript", "DSA", "Git"],               "type": "pu"},
        {"tag": "Phase 02 · Full-Stack",     "title": "Web & API Development",   "desc": "Build end-to-end applications with modern frameworks. Master REST APIs, authentication, and databases.",                                          "skills": ["React / Vue", "Node.js / Django", "SQL", "REST"],   "type": "bl"},
        {"tag": "Phase 03 · DevOps",         "title": "Deployment & Cloud",       "desc": "Learn CI/CD, containerisation, and cloud platforms to ship production-grade software confidently.",                                              "skills": ["Docker", "GitHub Actions", "AWS / GCP", "Linux"],   "type": "gr"},
        {"tag": "Phase 04 · Portfolio",      "title": "Three Showcase Projects",  "desc": "Build: a CRUD app with auth, a real-time app, and a complex API-heavy service. Deploy all three.",                                              "skills": ["Full Stack Apps", "Open Source", "GitHub"],         "type": "pu"},
        {"tag": "Phase 05 · Interviews",     "title": "Interview Mastery",        "desc": "Practice DSA daily, study system design patterns, and conduct mock interviews with peers regularly.",                                            "skills": ["LeetCode", "System Design", "Mock Interviews"],     "type": "bl"},
    ],
    "data scientist": [
        {"tag": "Phase 01 · Foundations",    "title": "Python & Statistics",      "desc": "Master Python for data work and build strong statistical reasoning around probability and inference.",                                           "skills": ["Python", "NumPy", "Pandas", "Statistics"],          "type": "pu"},
        {"tag": "Phase 02 · ML Core",        "title": "Machine Learning",         "desc": "Learn supervised and unsupervised algorithms, model evaluation, cross-validation, and feature engineering.",                                     "skills": ["scikit-learn", "XGBoost", "Feature Eng.", "CV"],    "type": "bl"},
        {"tag": "Phase 03 · Deep Learning",  "title": "Neural Networks & NLP",   "desc": "Explore neural networks, language models, and computer vision with PyTorch or TensorFlow.",                                                     "skills": ["TensorFlow", "PyTorch", "CNNs", "Transformers"],    "type": "gr"},
        {"tag": "Phase 04 · Communication",  "title": "Data Storytelling",        "desc": "Translate complex findings into clear, compelling visuals and dashboards that drive real decisions.",                                             "skills": ["Matplotlib", "Seaborn", "Power BI", "Streamlit"],   "type": "pu"},
        {"tag": "Phase 05 · Projects",       "title": "End-to-End ML Projects",   "desc": "Compete on Kaggle, deploy models as APIs, and publish insights publicly on Medium or LinkedIn.",                                                "skills": ["Kaggle", "MLflow", "FastAPI", "Hugging Face"],      "type": "bl"},
    ],
    "machine learning engineer": [
        {"tag": "Phase 01 · Foundations",    "title": "Math & Python Mastery",    "desc": "Linear algebra, calculus, probability theory, and clean Python are the non-negotiable bedrock.",                                               "skills": ["Linear Algebra", "Calculus", "Probability", "Py"],  "type": "pu"},
        {"tag": "Phase 02 · Frameworks",     "title": "Deep Learning Tools",      "desc": "Build and train neural networks with PyTorch and TensorFlow. Understand modern architectures deeply.",                                          "skills": ["PyTorch", "TensorFlow", "Keras", "JAX"],            "type": "bl"},
        {"tag": "Phase 03 · MLOps",          "title": "Production ML Systems",    "desc": "Deploy, monitor, retrain, and scale ML models. Own the full lifecycle from experiment to production.",                                          "skills": ["Docker", "Kubernetes", "MLflow", "Airflow"],        "type": "gr"},
        {"tag": "Phase 04 · Specialisation", "title": "Choose Your Domain",       "desc": "Pick NLP, Computer Vision, Recommender Systems, or Reinforcement Learning — then go extremely deep.",                                          "skills": ["Transformers", "YOLO", "RecSys", "RL"],             "type": "pu"},
        {"tag": "Phase 05 · Research",       "title": "Contribute & Publish",     "desc": "Read arXiv papers, reproduce results, contribute to Hugging Face, and build an open research presence.",                                        "skills": ["arXiv", "Papers With Code", "HF", "GitHub"],        "type": "bl"},
    ],
    "web developer": [
        {"tag": "Phase 01 · Foundations",    "title": "The Web Triad",            "desc": "Master HTML, CSS, and JavaScript from scratch. Build responsive, accessible layouts without frameworks.",                                       "skills": ["HTML5", "CSS3", "JavaScript ES6+", "Flex/Grid"],    "type": "pu"},
        {"tag": "Phase 02 · Frontend",       "title": "React & Modern Tooling",   "desc": "Learn React deeply — state, hooks, performance. Add TypeScript and Vite to your workflow.",                                                     "skills": ["React", "Next.js", "TypeScript", "Redux"],          "type": "bl"},
        {"tag": "Phase 03 · Backend",        "title": "Servers & Databases",      "desc": "Build REST APIs and connect to databases. Understand auth, ORMs, and basic scalability concepts.",                                              "skills": ["Node.js", "Express", "PostgreSQL", "GraphQL"],      "type": "gr"},
        {"tag": "Phase 04 · Quality",        "title": "Performance & Testing",    "desc": "Optimise Core Web Vitals, write automated tests, set up CI/CD, and sharpen developer experience.",                                              "skills": ["Jest / Cypress", "Lighthouse", "Docker", "CI/CD"],  "type": "pu"},
        {"tag": "Phase 05 · Ship",           "title": "Portfolio & Launch",       "desc": "Deploy real projects to Vercel or Netlify and build a personal portfolio that genuinely stands out.",                                           "skills": ["Vercel / Netlify", "GitHub", "Portfolio", "OSS"],   "type": "bl"},
    ],
    "ux designer": [
        {"tag": "Phase 01 · Foundations",    "title": "Design Principles",        "desc": "Study Gestalt theory, typography, colour, layout, and accessibility standards for digital products.",                                           "skills": ["Typography", "Colour Theory", "Gestalt", "WCAG"],   "type": "pu"},
        {"tag": "Phase 02 · Research",       "title": "UX Research Methods",      "desc": "Master user interviews, usability testing, affinity mapping, and persona creation workflows.",                                                  "skills": ["Interviews", "Usability Tests", "Personas", "JMs"], "type": "bl"},
        {"tag": "Phase 03 · Tools",          "title": "Figma & Prototyping",      "desc": "Become proficient in Figma — auto layout, design systems, interactive prototypes, and dev handoff.",                                           "skills": ["Figma", "Prototyping", "Design Systems", "Auto"],   "type": "gr"},
        {"tag": "Phase 04 · Portfolio",      "title": "Case Study Portfolio",     "desc": "Document three redesign or original projects with clear problem statements, process, and outcomes.",                                             "skills": ["Case Studies", "Behance", "Dribbble", "Portfolio"], "type": "pu"},
        {"tag": "Phase 05 · Collaboration",  "title": "Developer Partnership",    "desc": "Work in cross-functional teams, communicate decisions clearly, and deliver precise developer handoffs.",                                        "skills": ["Dev Handoff", "Zeplin", "Agile/Scrum", "A/B"],      "type": "bl"},
    ],
    "product manager": [
        {"tag": "Phase 01 · Foundations",    "title": "Product Thinking",         "desc": "Understand user needs, define product vision, and learn market positioning and competitive analysis.",                                          "skills": ["User Research", "JTBD", "Competitive Analysis"],    "type": "pu"},
        {"tag": "Phase 02 · Strategy",       "title": "Roadmaps & Prioritisation","desc": "Learn OKRs, RICE, MoSCoW, and write clear, actionable product requirements documents.",                                                        "skills": ["OKRs", "RICE", "Roadmaps", "PRDs"],                 "type": "bl"},
        {"tag": "Phase 03 · Data",           "title": "Analytics & Experiments",  "desc": "Drive decisions with product analytics. Master funnels, retention, cohort analysis, and A/B testing.",                                         "skills": ["Mixpanel", "SQL", "A/B Testing", "Cohorts"],        "type": "gr"},
        {"tag": "Phase 04 · Execution",      "title": "Agile Leadership",         "desc": "Lead sprints, manage engineering relationships, and ship products iteratively and on schedule.",                                                "skills": ["Jira", "Agile", "Scrum", "Stakeholder Mgmt"],       "type": "pu"},
        {"tag": "Phase 05 · Career",         "title": "PM Portfolio",             "desc": "Build side projects, document case studies, and target Associate PM programmes at top tech companies.",                                         "skills": ["Case Studies", "PM Certs", "Networking", "APM"],    "type": "bl"},
    ],
    "data analyst": [
        {"tag": "Phase 01 · Foundations",    "title": "SQL & Excel",              "desc": "Data manipulation in Excel, then SQL for querying large structured datasets efficiently and accurately.",                                       "skills": ["Excel", "SQL", "Pivot Tables", "Joins & Aggs"],     "type": "pu"},
        {"tag": "Phase 02 · Python",         "title": "Python for Analysis",      "desc": "Automate analysis workflows with Python. Master Pandas, NumPy, and key visualisation libraries.",                                              "skills": ["Python", "Pandas", "Matplotlib", "Seaborn"],        "type": "bl"},
        {"tag": "Phase 03 · Visualisation",  "title": "Dashboards & BI",          "desc": "Build executive-ready dashboards and business intelligence reports with Tableau or Power BI.",                                                 "skills": ["Tableau", "Power BI", "Looker", "Streamlit"],       "type": "gr"},
        {"tag": "Phase 04 · Statistics",     "title": "Statistical Analysis",     "desc": "Apply hypothesis testing, regression, and A/B experiment design to answer real business questions.",                                            "skills": ["Statistics", "Hypothesis Testing", "Regression"],   "type": "pu"},
        {"tag": "Phase 05 · Portfolio",      "title": "Public Work",              "desc": "Share analyses on GitHub, write articles on Medium, and solve Kaggle datasets to build credibility.",                                          "skills": ["Kaggle", "GitHub", "Medium Blog", "Public APIs"],   "type": "bl"},
    ],
    "cybersecurity analyst": [
        {"tag": "Phase 01 · Foundations",    "title": "Networking & Linux",       "desc": "Master TCP/IP, DNS, firewalls, and Linux CLI. Analyse network traffic with Wireshark confidently.",                                            "skills": ["TCP/IP", "DNS", "Wireshark", "Linux CLI"],          "type": "pu"},
        {"tag": "Phase 02 · Security Core",  "title": "Attack Vectors & Defence", "desc": "Learn cryptography, authentication protocols, OWASP Top 10, and common vulnerability patterns.",                                              "skills": ["Cryptography", "OWASP Top 10", "PKI", "Auth"],      "type": "bl"},
        {"tag": "Phase 03 · Offensive",      "title": "Ethical Hacking",          "desc": "Practice through CTF challenges and pentesting labs. Build genuine hands-on experience safely.",                                               "skills": ["Kali Linux", "Metasploit", "Burp Suite", "CTFs"],   "type": "gr"},
        {"tag": "Phase 04 · Certifications", "title": "Industry Credentials",     "desc": "Earn recognised certifications that validate your expertise and open doors with employers.",                                                   "skills": ["CompTIA Sec+", "CEH", "OSCP", "CISSP"],             "type": "pu"},
        {"tag": "Phase 05 · SOC",            "title": "Incident Response",        "desc": "Detect, respond to, contain, and recover from real-world security incidents inside a SOC environment.",                                        "skills": ["SIEM", "Splunk", "Incident Response", "Threat Intel"],"type": "bl"},
    ],
}


# ═════════════════════════════════════════════════════════════
# CHATBOT KNOWLEDGE BASE
# ═════════════════════════════════════════════════════════════
CHATBOT_KB: dict[str, dict[str, str]] = {
    "_general": {
        "salary":        "Salaries vary widely based on skill and location. In India, entry-level tech roles range 4–12 LPA; senior roles reach 25–60+ LPA. In the US, junior engineers earn $70K–$110K; senior roles often exceed $160K+.",
        "internship":    "To land internships: build 2–3 live projects on GitHub, connect with founders and recruiters on LinkedIn, and apply via Internshala, Unstop, Wellfound (formerly AngelList), and LinkedIn. Thoughtful, personalized cold emails to startup founders work surprisingly well.",
        "resume":        "A strong resume is strictly one page, ATS-friendly, and highlights measurable impact using numbers (e.g., 'Reduced load time by 20%'). Use the STAR format for experience bullets. Try the Resume Analyzer tab above for personalised feedback.",
        "certifications":"Certifications add real credibility if they match your stack. Prioritise official Google Cloud, AWS, Microsoft Azure, Coursera, or edX certs that align directly with your target career path.",
        "college":       "Your college name helps open doors, but it is far from the whole story. A strong GitHub portfolio, live deployed projects, and clear communication during interviews often outweigh institution prestige at modern tech companies.",
    },
    "software developer": {
        "skills":    "Master: Python or JavaScript, Data Structures & Algorithms (DSA), REST APIs, React (frontend), Django or Node.js (backend), Docker, and Git. TypeScript is increasingly essential at top companies.",
        "start":     "Start with Python fundamentals, build a CRUD app, learn React, connect the frontend and backend, then deploy it. Aim for three live deployed projects within your first six months.",
        "next":      "After the basics: learn TypeScript, understand microservices and system design patterns, contribute to open-source, and consistently solve LeetCode Medium problems.",
        "projects":  "Build: 1) A task manager with secure authentication, 2) A real-time chat app using WebSockets, 3) A well-documented RESTful API. All three should be live, deployed, and linked on your GitHub.",
        "interview": "Practice DSA on LeetCode daily, study system design via 'Grokking the System Design Interview', and conduct mock technical interviews on platforms like Pramp or interviewing.io.",
    },
    "data scientist": {
        "skills":    "Master: Python, Pandas, NumPy, scikit-learn, SQL, Applied Statistics, Matplotlib/Seaborn, and at least one deep learning framework (PyTorch or TensorFlow). Familiarity with Prompt Engineering and GenAI is highly recommended.",
        "start":     "Python first, then Statistics, then Pandas, then ML with scikit-learn, and finally your first Kaggle notebook. Introduce deep learning and neural networks incrementally afterward.",
        "next":      "Specialise in NLP (using Hugging Face), Computer Vision (CNNs/ViTs), or MLOps (MLflow and Docker). Go deep in one specific area rather than being shallow across many.",
        "projects":  "Build: 1) A full EDA and prediction notebook on Kaggle, 2) A sentiment analyser or RAG (Retrieval-Augmented Generation) pipeline, 3) A deployed ML model as a Streamlit or FastAPI web app.",
        "interview": "Revise the bias-variance tradeoff, regularisation, ensemble methods, and SQL window functions. Prepare a strong, end-to-end project walkthrough explaining your data cleaning decisions.",
    },
    "machine learning engineer": {
        "skills":    "Core: Python, Linear Algebra, PyTorch or TensorFlow, MLOps tools (Docker, Kubernetes, GitHub Actions), FastAPI for model serving, and cloud AI platforms (AWS SageMaker).",
        "start":     "Math foundations → Python → ML theory → PyTorch → train your first models → MLOps → deploy a model to a live production endpoint.",
        "next":      "Master integrating Large Language Models (LLMs) via APIs, fine-tuning open-source models (Llama/Mistral), and building RAG systems. Contribute to Hugging Face or Papers With Code.",
        "projects":  "Build: 1) A fine-tuned LLM application, 2) A computer vision classifier deployed as a scalable API, 3) A recommender system featuring both offline and online evaluation.",
        "interview": "Prepare for ML system design questions (e.g., 'How do you scale a recommendation engine?'), Python coding exercises, model debugging scenarios, and MLOps deployment strategies.",
    },
    "web developer": {
        "skills":    "HTML, CSS, JavaScript deeply, then React, Node.js or Express, SQL or MongoDB, REST APIs, TypeScript, Docker, and CI/CD pipelines.",
        "start":     "Clone a simple website layout, build your own portfolio, add a backend, connect a database, then deploy. Ship something real and functional within three months.",
        "next":      "Transition to full-stack frameworks like Next.js. Master TypeScript, write tests with Jest/Cypress, optimise Core Web Vitals, and learn cloud hosting fundamentals (Vercel/AWS).",
        "projects":  "Build: 1) A highly polished personal portfolio site, 2) A SaaS landing page with Stripe payment integration and auth, 3) A real-time collaboration tool.",
        "interview": "Focus on JavaScript closures/promises, React lifecycle/hooks under the hood, REST API design principles, and have one full-stack project you can discuss in extreme technical depth.",
    },
    "ux designer": {
        "skills":    "Figma, UX research methodologies, information architecture, rapid prototyping, design systems, WCAG accessibility standards, and basic HTML/CSS awareness.",
        "start":     "Learn core design principles (Gestalt, typography, spacing), use Figma daily, redesign three popular apps, and document each process as a detailed case study.",
        "next":      "Go deep on UX research through live user interviews and usability testing. Explore motion design with Lottie or Framer. Learn how to execute perfect developer handoffs.",
        "projects":  "Build: 1) A redesign case study showing the 'before/after' metrics, 2) A new app concept with a fully interactive Figma prototype, 3) A reusable design system/component library.",
        "interview": "Present your case studies as compelling stories. Explain the 'why' behind every design decision. Be prepared to walk through your complete user research and ideation process.",
    },
    "product manager": {
        "skills":    "Product sense, roadmapping frameworks (RICE, MoSCoW), SQL for data analysis, user research, A/B testing methodology, stakeholder communication, and Agile/Scrum.",
        "start":     "Read 'Inspired' by Marty Cagan, shadow an existing PM if possible, build a side product, document it as a case study, then apply for Associate PM (APM) programmes.",
        "next":      "Develop a clear specialisation: Growth PM, Platform PM, or AI/Data PM. Get highly proficient in SQL. Learn to interpret Mixpanel/Amplitude analytics independently.",
        "projects":  "Document: 1) A critical product teardown of an app you use daily, 2) A comprehensive PRD (Product Requirements Document) for a new feature, 3) A go-to-market strategy.",
        "interview": "Prepare for product design ('Design an elevator for the blind'), metrics/execution, market estimation (fermi problems), and strategy questions. Use resources like Exponent.",
    },
    "data analyst": {
        "skills":    "SQL is absolutely essential. Following that: Advanced Excel, Python (Pandas), Tableau or Power BI for dashboards, Applied Statistics, and business storytelling.",
        "start":     "SQL syntax → Excel pivot tables → Python basics → Pandas manipulation → build your first interactive dashboard in Tableau or Power BI.",
        "next":      "Learn A/B testing experiment design, basic statistical modelling, Python task automation, and how to translate raw data into actionable business strategy.",
        "projects":  "Build: 1) An interactive sales dashboard in Tableau/Power BI, 2) A full Exploratory Data Analysis (EDA) report on a public dataset, 3) An automated Python report delivered via email.",
        "interview": "Practice complex SQL joins and aggregations daily. Walk through past analyses and, most importantly, explain how your insights drove concrete business actions.",
    },
    "cybersecurity analyst": {
        "skills":    "Networking fundamentals (TCP/IP, DNS, OSI model), Linux CLI, Python scripting, Wireshark, Burp Suite for web app testing, and SIEM tools (Splunk).",
        "start":     "Networking fundamentals → Linux administration → Python scripting → OWASP Top 10 vulnerabilities → complete CTF (Capture The Flag) challenges on TryHackMe or HackTheBox.",
        "next":      "Specialise in penetration testing, threat intelligence, SOC (Security Operations Center) analysis, or cloud security. Pursue the CompTIA Security+ as your foundational certification.",
        "projects":  "Complete: 1) The TryHackMe beginner/defense paths, 2) Set up a Kali Linux home lab with vulnerable VMs, 3) Write a professional vulnerability assessment report for a sample target.",
        "interview": "Expect deep networking protocol questions, explanations of SQLi and XSS attack mechanics, incident response frameworks, and technical deep dives into your home lab setup.",
    },
}

INTENT_MAP: dict[str, list[str]] = {
    "skills":        ["skill", "learn", "what should i", "which tech", "language", "tool", "know", "technology", "need to study"],
    "start":         ["start", "begin", "new to", "how to get", "roadmap", "first step", "where do i", "getting started"],
    "next":          ["next", "after", "advanced", "already know", "improve", "level up", "what else", "progression"],
    "projects":      ["project", "portfolio", "build", "practice", "example", "idea", "make", "create"],
    "interview":     ["interview", "prepare", "crack", "question", "job", "hiring", "placement"],
    "salary":        ["salary", "pay", "earn", "money", "package", "lpa", "ctc", "compensation", "income"],
    "internship":    ["intern", "internship", "experience", "fresher", "entry level", "placement"],
    "resume":        ["resume", "cv", "profile", "ats", "curriculum vitae"],
    "certifications":["cert", "certification", "course", "udemy", "coursera", "mooc", "badge", "credential"],
    "college":       ["college", "degree", "tier", "nit", "iit", "university", "campus", "institution"],
}


def detect_intent(msg: str) -> str:
    """Identify user intent from a chat message."""
    msg_l = msg.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(k in msg_l for k in keywords):
            return intent
    return "unknown"


def chatbot_response(user_msg: str, career: str) -> str:
    """
    Rule-based chatbot.  Career-specific answers first, general KB fallback.
    """
    career_l   = career.lower().strip()
    intent     = detect_intent(user_msg)
    kb_career  = CHATBOT_KB.get(career_l, {})
    kb_general = CHATBOT_KB["_general"]

    if intent != "unknown" and intent in kb_career:
        return kb_career[intent]
    if intent != "unknown" and intent in kb_general:
        return kb_general[intent]

    msg_l = user_msg.lower()
    for key, val in kb_career.items():
        if key in msg_l:
            return val

    return (
        f"Great question! For a {career.title()} career, I can help with: "
        f"skills to learn, how to get started, project ideas, interview tips, "
        f"salary expectations, or internship advice. What would you like to explore?"
    )

# ═════════════════════════════════════════════════════════════
# RESUME ANALYZER
# ═════════════════════════════════════════════════════════════
RESUME_KEYWORDS: dict[str, list[str]] = {
    "default":              ["Python", "project", "team", "experience", "skills", "education", "GitHub", "impact"],
    "software developer":   ["Python", "JavaScript", "React", "Node", "API", "SQL", "Docker", "Git", "testing", "CI/CD", "agile", "TypeScript"],
    "data scientist":       ["Python", "machine learning", "TensorFlow", "PyTorch", "pandas", "scikit", "SQL", "statistics", "model", "GenAI", "deep learning"],
    "machine learning engineer": ["PyTorch", "TensorFlow", "MLOps", "deployment", "Docker", "Kubernetes", "pipeline", "NLP", "LLM", "MLflow"],
    "web developer":        ["HTML", "CSS", "JavaScript", "React", "Node", "REST", "API", "responsive", "TypeScript", "Next.js", "database"],
    "ux designer":          ["Figma", "user research", "prototype", "wireframe", "usability", "design system", "accessibility", "case study", "UI", "UX"],
    "product manager":      ["roadmap", "stakeholder", "OKR", "user research", "A/B test", "agile", "sprint", "metrics", "strategy", "launch"],
    "data analyst":         ["SQL", "Excel", "Tableau", "Power BI", "Python", "dashboard", "KPI", "report", "analysis", "statistics"],
    "cybersecurity analyst":["penetration", "SIEM", "firewall", "vulnerability", "Kali", "network", "incident response", "cryptography", "security", "compliance"],
}


def analyze_resume(text: str, career: str) -> dict:
    """
    Score resume (0–100) against career-specific keywords + structural signals.
    Returns score, grade (A–D), matched/missing keywords, and improvement suggestions.
    """
    career_l   = career.lower().strip()
    keywords   = RESUME_KEYWORDS.get(career_l, RESUME_KEYWORDS["default"])
    text_lower = text.lower()

    matched = [k for k in keywords if k.lower() in text_lower]
    missing = [k for k in keywords if k.lower() not in text_lower]

    # Adjusted to 49 so that max kw_score (49) + max bonus (51) exactly equals 100
    kw_score   = (len(matched) / max(len(keywords), 1)) * 49
    
    length_ok  = 180 < len(text.split()) < 750
    has_github  = "github" in text_lower
    has_nums    = bool(re.search(r"\d+\s*%|\d+x|\$[\d,]+|₹[\d,]+|\d+\s*(users|clients|projects)", text, re.I))
    has_action  = any(w in text_lower for w in ["built", "developed", "designed", "led", "improved",
                      "reduced", "increased", "deployed", "created", "launched", "optimised", "automated"])
    has_edu     = any(w in text_lower for w in ["b.tech", "b.e", "bsc", "msc", "bachelor", "master", "degree", "cgpa", "gpa"])
    has_contact = any(w in text_lower for w in ["email", "phone", "linkedin", "@", "mobile"])

    bonus = sum([
        has_github  * 9,
        has_nums    * 11,
        has_action  * 8,
        has_edu     * 5,
        has_contact * 4,
        14 if length_ok else 3,
    ])
    
    score = max(0, min(100, int(kw_score + bonus)))
    grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))

    suggestions: list[tuple[str, str]] = []
    if len(matched) < len(keywords) * 0.5:
        suggestions.append(("⚠️", f"Add key {career.title()} terms: {', '.join(missing[:4])}"))
    if not has_nums:
        suggestions.append(("📊", "Quantify achievements — add numbers, percentages, or scale metrics"))
    if not has_github:
        suggestions.append(("🔗", "Add your GitHub profile URL to showcase real code and projects"))
    if not has_action:
        suggestions.append(("🚀", "Start every bullet with an action verb: Built, Deployed, Optimised, Led"))
    if not length_ok:
        msg = ("Expand your resume — add more project detail and experience" if len(text.split()) <= 180
               else "Trim to one page — aim for roughly 400–600 words of content")
        suggestions.append(("📄", msg))
    if not has_edu:
        suggestions.append(("🎓", "Include qualifications clearly: degree, institution, year, and CGPA"))
    if not has_contact:
        suggestions.append(("📞", "Ensure your contact info (email, phone, LinkedIn) is clearly visible"))
    if not suggestions:
        suggestions.append(("✅", "Strong resume! Consider a peer review or running it through an ATS checker."))

    return {
        "score":       score,
        "grade":       grade,
        "matched_kws": matched,
        "missing_kws": missing,
        "suggestions": suggestions,
    }

# ═════════════════════════════════════════════════════════════
# CAREER ICON MAP
# ═════════════════════════════════════════════════════════════
_CAREER_ICONS: dict[str, str] = {
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
    c = career.lower()
    for key, icon in _CAREER_ICONS.items():
        if key in c:
            return icon
    return "🚀"


# ═════════════════════════════════════════════════════════════
# MODEL — loads career_model.pkl
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    """
    Safely load career_model.pkl from the same directory as app.py.
    Returns (model, error_string_or_None).
    """
    path = os.path.join(BASE_DIR, "career_model.pkl")
    if not os.path.exists(path):
        return None, "career_model.pkl not found — place it in the same folder as app.py"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


@st.cache_resource(show_spinner=False)
def load_label_encoder() -> tuple:
    """
    Safely load label_encoder.pkl from the same directory as app.py.
    Returns (label_encoder, error_string_or_None).
    """
    path = os.path.join(BASE_DIR, "label_encoder.pkl")
    if not os.path.exists(path):
        return None, "label_encoder.pkl not found — class labels may appear as numbers"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load label encoder: {exc}"


# ═════════════════════════════════════════════════════════════
# QUIZ → 23-FEATURE VECTOR
# ═════════════════════════════════════════════════════════════
def quiz_to_features(answers: list[int], model) -> pd.DataFrame:
    """
    Convert quiz answers into one model-ready row with the same feature names
    used during model training (including dynamic internet_access columns).
    """
    a = answers

    def clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # Core latent signals (0..1) derived from behavioural answers
    tech_lat = (
        (1.0 if a[1] == 0 else 0.0) +
        (1.0 if a[4] == 0 else 0.0) +
        (1.0 if a[11] == 0 else 0.0) +
        (1.0 if a[13] == 0 else 0.0) +
        (1.0 if a[14] == 0 else 0.0)
    ) / 5.0
    
    art_lat = (
        (1.0 if a[1] == 1 else 0.0) +
        (1.0 if a[4] == 1 else 0.0) +
        (1.0 if a[11] == 1 else 0.0) +
        (1.0 if a[13] == 1 else 0.0) +
        (1.0 if a[14] == 1 else 0.0)
    ) / 5.0
    
    biz_lat = (
        (1.0 if a[1] == 2 else 0.0) +
        (1.0 if a[4] == 2 else 0.0) +
        (1.0 if a[11] == 2 else 0.0) +
        (1.0 if a[13] == 2 else 0.0) +
        (1.0 if a[14] == 2 else 0.0)
    ) / 5.0
    
    res_lat = (
        (1.0 if a[1] == 3 else 0.0) +
        (1.0 if a[4] == 3 else 0.0) +
        (1.0 if a[11] == 3 else 0.0) +
        (1.0 if a[13] == 3 else 0.0) +
        (1.0 if a[14] == 3 else 0.0)
    ) / 5.0

    # Base trait mappings from specific questions
    academic_base = [17, 14, 11, 8][a[8]]
    math_score    = [9, 7, 5, 3][a[3]]
    self_learning = [1.0, 0.8, 0.5, 0.2][a[9]]
    deadline_ctrl = [1.0, 0.8, 0.5, 0.2][a[6]]

    # Feature Calculation
    grade1      = clip(round(academic_base + (math_score - 6) * 0.6), 0, 20)
    grade2      = clip(round(academic_base + (deadline_ctrl - 0.6) * 3), 0, 20)
    final_grade = clip(round((grade1 + grade2) / 2 + self_learning * 1.5), 0, 20)

    study_time = int(clip(round(1 + self_learning * 3), 1, 4))
    failures   = int(clip(round((20 - final_grade) / 6), 0, 3))
    absences   = int(clip(round(20 - (academic_base * 0.8) + (3 - a[6]) * 2), 0, 30))

    # Big Five Personality Mapping
    openness          = clip(0.35 + art_lat * 0.35 + res_lat * 0.2, 0, 1)
    conscientiousness = clip(0.3 + self_learning * 0.4 + deadline_ctrl * 0.3, 0, 1)
    extraversion      = clip(0.2 + (3 - a[5]) * 0.18 + (1 if a[2] == 2 else 0) * 0.15, 0, 1)
    agreeableness     = clip(0.35 + (1 if a[2] in [1, 2] else 0) * 0.25 + (1 if a[12] == 2 else 0) * 0.2, 0, 1)
    neuroticism       = clip(0.2 + (a[6] / 3) * 0.6, 0, 1)

    # Skill & Habit Mapping
    coding_skill        = int(clip(round(2 + tech_lat * 7), 0, 9))
    communication_skill = int(clip(round(2 + (extraversion * 3.5) + (1 if a[2] == 2 else 0) * 2), 0, 9))
    analytical_skill    = int(clip(round(2 + (1 if a[0] == 0 else 0) * 3 + (math_score / 10) * 4), 0, 9))

    study_hours   = int(clip(round(3 + self_learning * 6), 1, 10))
    consistency   = clip(0.25 + self_learning * 0.4 + deadline_ctrl * 0.25, 0, 1)
    participation = clip(0.2 + (1 if a[2] in [1, 2] else 0) * 0.25 + (extraversion * 0.35), 0, 1)

    tech_interest     = int(clip(round(tech_lat * 9), 0, 9))
    art_interest      = int(clip(round(art_lat * 9), 0, 9))
    business_interest = int(clip(round(biz_lat * 9), 0, 9))

    # Calibration: the training labels default to Research Scientist unless
    # a domain signal crosses specific thresholds. Push dominant intent above
    # those boundaries so different quiz profiles lead to distinct careers.
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
    else:
        analytical_skill = max(analytical_skill, 8)
        final_grade      = max(final_grade, 13)

    family_income = int(clip(round(2 + (0.5 if a[14] in [0, 1] else -0.2)), 1, 4))
    has_internet  = 1 if (self_learning >= 0.5 or a[10] in [0, 1, 2]) else 0

    base_row = {
        "grade1": grade1,
        "grade2": grade2,
        "final_grade": final_grade,
        "study_time": study_time,
        "failures": failures,
        "absences": absences,
        "openness": openness,
        "conscientiousness": conscientiousness,
        "extraversion": extraversion,
        "agreeableness": agreeableness,
        "neuroticism": neuroticism,
        "coding_skill": coding_skill,
        "communication_skill": communication_skill,
        "analytical_skill": analytical_skill,
        "study_hours": study_hours,
        "consistency": consistency,
        "participation": participation,
        "tech_interest": tech_interest,
        "art_interest": art_interest,
        "business_interest": business_interest,
        "family_income": family_income,
    }

    # Model Alignment Failsafe
    expected_cols = []
    if model is not None and hasattr(model, "feature_names_in_"):
        expected_cols = [str(c) for c in model.feature_names_in_]

    if not expected_cols:
        expected_cols = list(base_row.keys()) + ["internet_access_no", "internet_access_yes"]

    # Construct the final row matching model expectations
    row = {col: 0.0 for col in expected_cols}
    for key, val in base_row.items():
        if key in row:
            row[key] = float(val)

    # Dynamic One-Hot Encoding handling
    if "internet_access_yes" in row:
        row["internet_access_yes"] = float(has_internet)
    if "internet_access_no" in row:
        row["internet_access_no"] = float(1 - has_internet)
    if "internet_access" in row:
        row["internet_access"] = float(has_internet)

    return pd.DataFrame([row], columns=expected_cols)
def _decode_career_label(value, label_encoder=None) -> str:
    """Decode model output class into a readable career string."""
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(value)])[0]
            return str(decoded).strip().lower()
        except Exception:
            pass
        try:
            decoded = label_encoder.inverse_transform([value])[0]
            return str(decoded).strip().lower()
        except Exception:
            pass

    # Fallback map for older model iterations without a label encoder
    fallback_map = {
        0: "data scientist",
        1: "entrepreneur",
        2: "research scientist",
        3: "software developer",
        4: "ui/ux designer",
    }
    try:
        maybe_num = int(value)
        if maybe_num in fallback_map:
            return fallback_map[maybe_num]
    except Exception:
        pass

    return str(value).strip().lower()


def predict(model, feature_vector: np.ndarray, label_encoder=None) -> tuple[str, float | None, list[tuple[str, float]]]:
    """
    Run inference and return (top_career, confidence_pct, top3_list).
    Handles models without predict_proba gracefully — no crashes.
    """
    raw   = model.predict(feature_vector)
    top_c = _decode_career_label(raw[0], label_encoder)
    top3: list[tuple[str, float]] = []
    conf: float | None = None

    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(feature_vector)[0]
        idx_top = np.argsort(probs)[::-1][:3]
        for idx in idx_top:
            name = _decode_career_label(model.classes_[idx], label_encoder)
            top3.append((name, round(float(probs[idx]) * 100, 1)))
        if top3:
            top_c = top3[0][0]
            conf  = top3[0][1]

    return top_c, conf, top3


def infer_personality(answers: list[int]) -> list[tuple[str, str, str]]:
    """Return list of (label, css_class, emoji) personality descriptors."""
    tags: list[tuple[str, str, str]] = []
    if answers[7] == 0 or answers[5] in [2, 3]:
        tags.append(("Introvert",  "pc-in", "🪐"))
    else:
        tags.append(("Extrovert",  "pc-ex", "⚡"))
    if answers[0] == 0 and answers[3] in [0, 1]:
        tags.append(("Analytical", "pc-an", "🔬"))
    if answers[0] == 1 or answers[1] == 1:
        tags.append(("Creative",   "pc-cr", "🎨"))
    if answers[2] == 2 and answers[5] == 0:
        tags.append(("Leader",     "pc-ld", "🏆"))
    if answers[0] == 0 and answers[1] == 3 and answers[13] == 3:
        tags.append(("Researcher", "pc-an", "📚"))
    return tags if tags else [("Explorer", "pc-an", "🌐")]


def generate_explanation(career: str, answers: list[int]) -> str:
    """Return a one-sentence personalised explanation for the predicted career."""
    strengths: list[str] = []
    if answers[5] == 0:          strengths.append("natural communication ability")
    if answers[0] == 0:          strengths.append("sharp logical reasoning")
    if answers[3] in [0, 1]:     strengths.append("strong mathematical grounding")
    if answers[9] == 0:          strengths.append("self-driven curiosity")
    if answers[10] in [0, 1]:    strengths.append("competitive hands-on experience")
    if answers[0] == 1:          strengths.append("creative problem-solving instinct")
    if answers[6] == 0:          strengths.append("high-performance mindset under pressure")
    if not strengths:
        strengths = ["a versatile skill profile", "intellectual adaptability"]
    s = " and ".join(strengths[:2])
    return f"Your {s} position you well for a career in {career.title()} — the role aligns naturally with how you think, work, and grow."

# ═════════════════════════════════════════════════════════════
# RENDER ROADMAP
# ═════════════════════════════════════════════════════════════
def render_roadmap(career: str) -> None:
    """Render the vertical-timeline roadmap for the predicted career."""
    key   = career.lower().strip()
    steps = ROADMAPS.get(key, ROADMAPS["default"])
    for i, step in enumerate(steps):
        is_last   = (i == len(steps) - 1)
        tags_html = "".join(
            f'<span class="sk-tag sk-{step["type"]}">{s}</span>'
            for s in step["skills"]
        )
        line_html = "" if is_last else '<div class="rm-line"></div>'
        st.markdown(f"""
<div class="rm-step">
  <div class="rm-left">
    <div class="rm-circle">{str(i + 1).zfill(2)}</div>
    {line_html}
  </div>
  <div class="rm-body">
    <div class="rm-tag">{step["tag"]}</div>
    <div class="rm-title">{step["title"]}</div>
    <div class="rm-desc">{step["desc"]}</div>
    <div>{tags_html}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════
_NAV = [
    ("home",    "🏠", "Home"),
    ("quiz",    "📝", "Neural Assessment"),
    ("results", "🎯", "AI Projections"),
    ("resume",  "📄", "Resume Analyzer"),
    ("chat",    "💬", "AI Mentor"),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
<div class="nav-brand">
  <div class="nav-logo">
    <div class="nav-logo-icon">◈</div>
    NextStep AI
  </div>
  <div class="nav-tagline">Your Smart Career Guide to the Next Step</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="nav-slabel">Navigation</div>', unsafe_allow_html=True)

        for key, icon, label in _NAV:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown('<div class="nav-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-slabel">Your Profile</div>', unsafe_allow_html=True)

        career = st.session_state.get("career", "")
        conf   = st.session_state.get("confidence", None)
        if career:
            conf_str = f"  ·  {conf}% match" if conf else ""
            st.markdown(f"""
<div class="nav-status">
  <div class="nav-s-career">{career_icon(career)}  {career.title()}</div>
  <div class="nav-s-label">Predicted Career{conf_str}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="nav-status">
  <div style="font-size:12px;color:var(--dim)">Complete the quiz to unlock your prediction.</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="nav-footer">NextStep AI · v4.0 · Proprietary ML Engine</div>',
                    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════
def page_home() -> None:
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <span class="dot"></span>
    Proprietary ML Engine · 100% Local Inference
  </div>
  <div class="hero-title">
    Discover Your<br><em>Perfect Career</em><br>
    <span class="hero-grad-text">with NextStep AI</span>
  </div>
  <div class="hero-sub">
    Unlock your professional destiny using advanced machine learning
    and deep personality mapping. Your smart career guide to the next step.
  </div>
  <div class="hero-pills">
    <div class="hero-pill"><span>📝</span> 15-Question Assessment</div>
    <div class="hero-pill"><span>🤖</span> ML Career Prediction</div>
    <div class="hero-pill"><span>🗺️</span> Step-by-Step Roadmap</div>
    <div class="hero-pill"><span>💬</span> AI Career Mentor</div>
    <div class="hero-pill"><span>📄</span> Resume Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    st.markdown('<div class="s-eye">Cognitive Toolkit</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-title">The Neural Advantage</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Everything you need to find and pursue the right career path — all in one intelligent platform.</div>', unsafe_allow_html=True)

    _FEATURES = [
        ("🤖", "AI Prediction",     "Real-time ML analysis maps your profile to the best-fit career with confidence scoring.",          "card-pu"),
        ("🧠", "Personality Map",   "Behavioural analysis reveals your Introvert / Extrovert / Analytical / Creative profile.",          "card-bl"),
        ("🗺️", "Roadmap Generator", "Phase-by-phase learning paths with tools, skills, and projects for your exact career.",            "card-cy"),
        ("💬", "Mentor Chatbot",    "24/7 AI-powered career coaching with no API keys — fully local and private.",                      "card-gr"),
        ("📄", "Resume Analyzer",   "Keyword scoring and improvement suggestions tailored to your predicted career path.",              "card-pu"),
        ("📊", "Top 3 Matches",     "See the top three careers ranked by probability with animated confidence bars.",                   "card-bl"),
    ]
    c1, c2 = st.columns(2)
    for i, (em, title, desc, cls) in enumerate(_FEATURES):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
<div class="card {cls}" style="margin-bottom:1rem">
  <div style="font-size:2rem;margin-bottom:.8rem">{em}</div>
  <div style="font-size:15px;font-weight:700;color:var(--snow);margin-bottom:6px">{title}</div>
  <div style="font-size:13px;color:var(--muted);line-height:1.65">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    b1, _, _ = st.columns([1, 2, 2])
    with b1:
        if st.button("🚀  Start Neural Assessment", use_container_width=True):
            st.session_state.page    = "quiz"
            st.session_state.q_index = 0
            st.session_state.answers = []
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ═════════════════════════════════════════════════════════════
def page_quiz(model, label_encoder) -> None:
    if "q_index" not in st.session_state: st.session_state.q_index = 0
    if "answers" not in st.session_state: st.session_state.answers = []

    total = len(QUESTIONS)
    qi    = st.session_state.q_index
    pct   = int((qi / total) * 100)

    # Progress header
    st.markdown(f"""
<div class="quiz-hdr">
  <div>
    <div class="quiz-prog-label">Neural Assessment Pulse · {pct}% Complete</div>
    <div class="quiz-prog-bg">
      <div class="quiz-prog-fill" style="width:{pct}%"></div>
    </div>
  </div>
  <div class="quiz-step-badge">Question {min(qi + 1, total)} of {total}</div>
</div>""", unsafe_allow_html=True)

    # ── All questions answered → predict ──
    if qi >= total:
        st.markdown("""
<div class="card card-pu" style="text-align:center;padding:3rem">
  <div style="font-size:3.5rem;margin-bottom:1rem">🧠</div>
  <div style="font-family:var(--display);font-size:1.8rem;font-weight:700;color:var(--snow);margin-bottom:.5rem">Assessment Complete</div>
  <div style="font-size:14px;color:var(--muted)">Analysing your cognitive profile with NextStep AI…</div>
</div>""", unsafe_allow_html=True)

        with st.spinner("Running ML prediction on your profile…"):
            time.sleep(0.9)
            feat = quiz_to_features(st.session_state.answers, model)
            if model:
                career, conf, top3 = predict(model, feat, label_encoder)
            else:
                # Demo mode when career_model.pkl is absent
                career, conf = "software developer", 73.6
                top3 = [
                    ("software developer", 73.6),
                    ("data scientist",     15.8),
                    ("web developer",       7.2),
                ]

        st.session_state.career      = career
        st.session_state.confidence  = conf
        st.session_state.top3        = top3
        st.session_state.personality = infer_personality(st.session_state.answers)
        st.session_state.explanation = generate_explanation(career, st.session_state.answers)

        st.success("✅  Prediction ready — navigate to **AI Projections** in the sidebar.")
        if st.button("View My Results →"):
            st.session_state.page = "results"
            st.rerun()
        return

    # ── Current question ──
    q = QUESTIONS[qi]
    st.markdown(f"""
<div class="card">
  <div class="quiz-q-num">Assessment Pulse · Question {qi + 1}</div>
  <div class="quiz-q-text">{q["text"]}</div>
</div>""", unsafe_allow_html=True)

    labels = ["A", "B", "C", "D"]
    for oi, opt in enumerate(q["options"]):
        lc, bc = st.columns([0.055, 0.945])
        with lc:
            st.markdown(
                f'<div style="margin-top:9px;font-family:var(--mono);font-size:11px;'
                f'color:var(--cyan);padding:6px 0;font-weight:600">{labels[oi]}</div>',
                unsafe_allow_html=True,
            )
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


# ═════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ═════════════════════════════════════════════════════════════
def page_results() -> None:
    if not st.session_state.get("career"):
        st.markdown("""
<div class="card" style="text-align:center;padding:3.5rem">
  <div style="font-size:3.5rem;margin-bottom:1.2rem">📝</div>
  <div style="font-family:var(--display);font-weight:700;font-size:1.8rem;color:var(--snow);margin-bottom:.5rem">No Results Yet</div>
  <div style="font-size:14px;color:var(--muted)">Complete the Neural Assessment to see your AI career projection.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Take the Assessment →"):
            st.session_state.page = "quiz"
            st.rerun()
        return

    career  = st.session_state.career
    conf    = st.session_state.get("confidence") or 0.0
    top3    = st.session_state.get("top3", [])
    persona = st.session_state.get("personality", [])
    expl    = st.session_state.get("explanation", "")
    icon    = career_icon(career)

    # ── Primary result card ──
    st.markdown(f"""
<div class="card card-pu" style="text-align:center;padding:3.5rem 2rem">
  <div class="result-badge">✦ AI Projection · Top Match</div>
  <div class="result-icon">{icon}</div>
  <div class="result-career">{career.title()}</div>
  <div class="result-conf">● Neural Sync: {conf:.1f}% confidence</div>
  <div class="result-expl">{expl}</div>
</div>""", unsafe_allow_html=True)

    # ── Confidence ring + Personality ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card card-bl">', unsafe_allow_html=True)
        st.markdown('<div class="s-eye">Neural Sync Score</div>', unsafe_allow_html=True)
        r     = 52
        circ  = 2 * 3.14159 * r
        dash  = (conf / 100) * circ
        ring_color = ("var(--green)" if conf >= 70 else ("var(--amber)" if conf >= 45 else "var(--rose)"))
        st.markdown(f"""
<div class="conf-wrap">
  <div class="conf-outer">
    <svg class="conf-svg" viewBox="0 0 120 120" style="position:absolute;inset:0;transform:rotate(-90deg)">
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8"/>
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="{ring_color}" stroke-width="8"
              stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"/>
    </svg>
    <div class="conf-inner">
      <span class="conf-num">{int(conf)}</span>
      <span class="conf-unit">% sync</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card card-pu">', unsafe_allow_html=True)
        st.markdown('<div class="s-eye">Personality Profile</div>', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="p-chip {cls}">{em} {lbl}</span>'
            for lbl, cls, em in persona
        )
        st.markdown(f'<div style="margin-top:.8rem;line-height:2.4">{chips}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top 3 predictions ──
    if top3:
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        st.markdown('<div class="s-eye">AI Projections</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-title">Your Top 3 Matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-sub">Ranked by neural profile match score based on your assessment answers.</div>', unsafe_allow_html=True)

        labels = ["TOP MATCH", "ALTERNATIVE", "ALTERNATIVE"]
        max_p  = top3[0][1] if top3 else 1.0
        for rank, (name, pct) in enumerate(top3):
            bw  = int((pct / max(max_p, 1)) * 100)
            lbl = labels[min(rank, 2)]
            st.markdown(f"""
<div class="t3-row">
  <span class="t3-rank">#{rank + 1}</span>
  <span class="t3-icon">{career_icon(name)}</span>
  <div style="flex:1;margin-left:12px">
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:3px">{lbl}</div>
    <span class="t3-name">{name.title()}</span>
  </div>
  <div class="t3-bar-bg"><div class="t3-bar-fill" style="width:{bw}%"></div></div>
  <span class="t3-pct">{pct}%</span>
</div>""", unsafe_allow_html=True)

    # ── Roadmap ──
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="s-eye">The Path Forward</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="s-title">Your {career.title()} Roadmap</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="s-sub">A structured, phase-by-phase guide built for your neural profile and career trajectory.</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_roadmap(career)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Action buttons ──
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("💬  AI Mentor",        use_container_width=True):
            st.session_state.page = "chat";   st.rerun()
    with r2:
        if st.button("📄  Resume Analyzer",  use_container_width=True):
            st.session_state.page = "resume"; st.rerun()
    with r3:
        if st.button("🔄  Retake Assessment", use_container_width=True):
            for k in ["career", "confidence", "top3", "personality", "explanation", "answers", "q_index"]:
                st.session_state.pop(k, None)
            st.session_state.page = "quiz"
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PAGE: RESUME ANALYZER
# ═════════════════════════════════════════════════════════════
def page_resume() -> None:
    st.markdown('<div class="s-eye">Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-title">Score Your Resume</div>', unsafe_allow_html=True)
    career = st.session_state.get("career", "default")
    st.markdown(
        f'<div class="s-sub">Keyword analysis optimised for '
        f'<strong style="color:var(--cyan)">{career.title()}</strong>. '
        f'Paste your plain-text resume below — not a PDF.</div>',
        unsafe_allow_html=True,
    )

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
        
        # Mapping to the new Neon CSS Variables
        grade_color = {"A": "var(--green)", "B": "var(--cyan)", "C": "var(--amber)", "D": "var(--rose)"}.get(grade, "var(--purple)")
        grade_bg    = {"A": "rgba(0,255,102,0.11)", "B": "rgba(0,240,255,0.11)", "C": "rgba(255,176,0,0.11)", "D": "rgba(255,0,85,0.11)"}.get(grade, "rgba(138,43,226,0.11)")
        grade_bd    = {"A": "rgba(0,255,102,0.28)", "B": "rgba(0,240,255,0.28)", "C": "rgba(255,176,0,0.28)", "D": "rgba(255,0,85,0.28)"}.get(grade, "rgba(138,43,226,0.28)")

        cs, ck = st.columns([1, 2])
        with cs:
            st.markdown(f"""
<div class="card" style="text-align:center;padding:2rem">
  <div style="font-family:var(--mono);font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:1rem">Neural Resume Score</div>
  <div style="font-family:var(--display);font-size:5.5rem;font-weight:700;color:{grade_color};line-height:1;text-shadow:0 0 20px {grade_color}">{score}</div>
  <div style="font-family:var(--mono);font-size:1rem;color:var(--muted);margin-top:.5rem">/ 100</div>
  <div style="margin-top:1.5rem;display:inline-block;padding:6px 18px;border-radius:100px;
              background:{grade_bg};border:1px solid {grade_bd};
              font-family:var(--mono);font-size:13px;font-weight:600;color:{grade_color};box-shadow:0 0 10px {grade_bg}">Grade {grade}</div>
</div>""", unsafe_allow_html=True)

        with ck:
            matched = result["matched_kws"]
            missing = result["missing_kws"]
            mh = "".join(f'<span class="kw-found">{k}</span>'   for k in matched)
            nh = "".join(f'<span class="kw-missing">{k}</span>' for k in missing[:7])
            st.markdown(f"""
<div class="card">
  <div style="margin-bottom:1.5rem">
    <div style="font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--green);margin-bottom:.8rem;text-shadow:0 0 8px rgba(0,255,102,0.3)">✓ Keywords Found ({len(matched)})</div>
    <div>{mh if mh else '<span style="color:var(--dim);font-size:13px">None detected</span>'}</div>
  </div>
  <div>
    <div style="font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--rose);margin-bottom:.8rem;text-shadow:0 0 8px rgba(255,0,85,0.3)">✗ Keywords Missing ({len(missing)})</div>
    <div>{nh if nh else '<span style="color:var(--dim);font-size:13px">All key terms present!</span>'}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="s-eye" style="margin-bottom:1.5rem">Improvement Suggestions</div>',
                    unsafe_allow_html=True)
        for icon, tip in result["suggestions"]:
            st.markdown(
                f'<div class="sug-row"><span style="flex-shrink:0;font-size:16px">{icon}</span>'
                f'<span>{tip}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ═════════════════════════════════════════════════════════════
def page_chat() -> None:
    career = st.session_state.get("career", "")

    st.markdown('<div class="s-eye">AI Mentor</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-title">NextStep AI Career Mentor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="s-sub">Your mentor is calibrated to '
        f'<strong style="color:var(--cyan)">'
        f'{career.title() if career else "general career guidance"}'
        f'</strong>. Ask about skills, projects, interviews, or salary.</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        greeting = (
            f"Hello! I am your NextStep AI Career Mentor. Your predicted career is "
            f"**{career.title()}**. Ask me anything — skills, how to start, project ideas, "
            f"interview tips, or salary expectations. What would you like to explore?"
            if career else
            "Hello! I am your NextStep AI Career Mentor. Complete the Neural Assessment for "
            "personalised guidance, or ask me a general career question right now!"
        )
        st.session_state.chat_history.append(("bot", greeting))

    # Suggested prompts
    PROMPTS = [
        "What skills do I need?",
        "How do I get started?",
        "Suggest 3 projects",
        "Interview tips",
        "What salary can I expect?",
    ]
    pcols = st.columns(len(PROMPTS))
    for i, (col, prompt) in enumerate(zip(pcols, PROMPTS)):
        with col:
            if st.button(prompt, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append(("user", prompt))
                st.session_state.chat_history.append(
                    ("bot", chatbot_response(prompt, career or "default"))
                )
                st.rerun()

    # Conversation
    for role, msg in st.session_state.chat_history:
        meta = "You" if role == "user" else "◈ NextStep AI Mentor"
        st.markdown(f"""
<div class="chat-b {role if role == 'user' else 'bot'}">
  <div class="chat-meta">{meta}</div>
  <div class="chat-txt">{msg}</div>
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
            send = st.form_submit_button("Send →", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append(("user", user_input.strip()))
        st.session_state.chat_history.append(
            ("bot", chatbot_response(user_input.strip(), career or "default"))
        )
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()

    # Initialise session state defaults
    for key, val in [("page", "home"), ("career", ""), ("answers", []), ("q_index", 0)]:
        if key not in st.session_state:
            st.session_state[key] = val

    # Load career_model.pkl once (cached)
    model, err = load_model()
    label_encoder, le_err = load_label_encoder()
    if err and st.session_state.page != "home":
        st.warning(f"⚠️ {err}  —  Running in demo mode; predictions are illustrative.")
    elif le_err and st.session_state.page != "home":
        st.warning(f"⚠️ {le_err}.")

    # Normalize older session values that may still contain encoded class IDs.
    if st.session_state.get("career"):
        st.session_state.career = _decode_career_label(st.session_state.career, label_encoder)

    if st.session_state.get("top3"):
        st.session_state.top3 = [
            (_decode_career_label(name, label_encoder), pct)
            for name, pct in st.session_state.top3
        ]

    if st.session_state.get("career") and st.session_state.get("answers"):
        st.session_state.explanation = generate_explanation(
            st.session_state.career,
            st.session_state.answers,
        )

    render_sidebar()

    page = st.session_state.page
    if   page == "home":    page_home()
    elif page == "quiz":    page_quiz(model, label_encoder)
    elif page == "results": page_results()
    elif page == "resume":  page_resume()
    elif page == "chat":    page_chat()

    st.markdown("""
<div class="footer">
  NextStep AI · Your Smart Career Guide to the Next Step<br>
  <span style="opacity:.45">Built with ❤️ using Machine Learning &amp; Streamlit · Powered by Custom Predictive Models</span>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()