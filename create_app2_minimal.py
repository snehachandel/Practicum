import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
content = content.replace("import pandas as pd", "import pandas as pd\nimport sklearn\nimport lightgbm as lgb")

# 2. Model & Label Encoder paths
content = content.replace('"career_model.pkl"', '"career_model_7.pkl"')
content = content.replace("'career_model.pkl'", "'career_model_7.pkl'")

content = content.replace('"label_encoder.pkl"', '"label_encoder_7.pkl"')
content = content.replace("'label_encoder.pkl'", "'label_encoder_7.pkl'")

# 3. Features extraction
old_feat_code = """    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(base_row.keys())

    row = {col: base_row.get(col, 0.0) for col in expected_cols}
    return pd.DataFrame([row], columns=expected_cols)"""

new_feat_code = """    try:
        import pickle, os
        with open(os.path.join(BASE_DIR, "features_7.pkl"), "rb") as f:
            expected_cols = pickle.load(f)
    except Exception as e:
        expected_cols = list(base_row.keys())
        
    row = {col: base_row.get(col, 0.0) for col in expected_cols}
    return pd.DataFrame([row], columns=expected_cols)"""

if old_feat_code in content:
    content = content.replace(old_feat_code, new_feat_code)
else:
    print("Warning: Could not find old_feat_code block")

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Minimal app2.py created successfully.")
