# Chapter 3: Problem Statement and Objectives

## NextStep AI — Intelligent Career Prediction System

---

## 3.1 Problem Statement

Choosing the right career is one of the most consequential decisions a student faces, yet it remains largely unsupported by data-driven tools. The current landscape presents three critical challenges:

**1. Career Choice Paralysis**
Students and fresh graduates are overwhelmed by the growing number of career options in technology and related fields. Without a structured framework to evaluate their aptitude, many make uninformed decisions based on peer pressure, parental expectations, or trending job titles rather than their actual strengths and interests.

**2. Lack of Personalised Guidance**
Traditional career counselling is generic, expensive, and appointment-based. Most students — particularly those in tier-2 and tier-3 institutions — have no access to personalised, data-backed career advice that accounts for their individual academic profile, personality traits, and skill set.

**3. No Accessible Self-Assessment Tools**
While psychometric assessments exist, they are typically paywalled, require professional interpretation, and do not provide actionable next steps. Students need a free, instant, and self-service tool that not only predicts a suitable career but also provides a learning roadmap, mentorship guidance, and resume feedback.

### Problem Definition

> *There is a need for an accessible, AI-driven career prediction system that maps a student's academic performance, personality traits, skills, and interests to a suitable career path — providing actionable guidance without requiring external API dependencies or compromising data privacy.*

---

## 3.2 Objectives

The following objectives were defined to address the stated problem:

| # | Objective | Description |
|---|---|---|
| 1 | **Build a multi-class career prediction model** | Train a machine learning classifier to predict one of five career categories (Software Developer, Data Scientist, UI/UX Designer, Entrepreneur, Research Scientist) from a 22-feature student profile. |
| 2 | **Compare classification algorithms** | Benchmark Decision Tree and LightGBM classifiers across multiple configurations to identify the best-performing algorithm. |
| 3 | **Evaluate feature selection techniques** | Apply SHAP, Mutual Information, and Boruta to identify the most predictive features, reducing model complexity without significant accuracy loss. |
| 4 | **Optimise model performance** | Use Optuna Bayesian hyperparameter tuning (40 trials) to maximise accuracy and AUC on the test set. |
| 5 | **Handle class imbalance** | Apply SMOTE oversampling to ensure fair prediction across minority career categories. |
| 6 | **Deploy an interactive web application** | Build a Streamlit-based interface providing career prediction, personalised roadmaps, AI mentorship, and resume analysis — all running locally with zero data transmission. |

---

## 3.3 Scope

| In Scope | Out of Scope |
|---|---|
| 5 career categories in technology and research domains | Non-technical career paths (law, medicine, arts) |
| 15-question behavioural quiz as input method | Detailed psychometric profiling (e.g., MBTI, Holland codes) |
| Local-only inference with no cloud dependency | Cloud-hosted multi-user deployment |
| Rule-based AI mentor chatbot | LLM-powered natural language mentoring |
| Plain-text resume analysis | PDF/DOCX resume parsing |
| Single-session predictions | User accounts and prediction history |

---

## 3.4 Expected Outcomes

1. A trained model achieving **≥ 85% accuracy** and **≥ 0.90 AUC** on the test set.
2. A comparative analysis of **10 model configurations** identifying the optimal algorithm and feature set.
3. A fully functional **Streamlit web application** with 5 integrated modules (Home, Quiz, Results, Chat, Resume).
4. A **privacy-first architecture** requiring no internet connection for inference.
