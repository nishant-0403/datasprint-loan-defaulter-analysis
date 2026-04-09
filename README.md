# datasprint-loan-defaulter-analysis


# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (you've already done this)
python model_train_and_preprocess.py

# 3. Score test set
python model_test_and_preprocess.py
# outputs to test_results/predictions.csv

# 4. Start FastAPI backend (Terminal 1)
uvicorn api:app --reload --port 8000

# 5. Start Streamlit dashboard (Terminal 2)
streamlit run dashboard.py
