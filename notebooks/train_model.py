import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. Load your downloaded Kaggle dataset [cite: 104, 108]
df = pd.read_csv('../data/Gene Expression Analysis and Disease Relationship.csv')
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# 2. Preprocessing & Dimensionality Reduction [cite: 106, 117]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize mean=0, std=1 [cite: 120]

pca = PCA(n_components=0.95) # Keep 95% variance [cite: 122]
X_pca = pca.fit_transform(X_scaled)

# 3. Model Training [cite: 132, 140]
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 4. Save Artifacts for the FastAPI server [cite: 251, 288, 304]
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/standard_scaler.pkl')
joblib.dump(pca, '../models/pca_transformer.pkl')
joblib.dump(model, '../models/xgboost_model.pkl')

print("Training Complete. Models saved to /models directory.")
