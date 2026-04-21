@echo off
echo Starting NextStep AI Career Prediction App...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python -m streamlit run app.py
) else (
    echo Virtual environment not found. Please ensure dependencies are installed.
    pause
)
