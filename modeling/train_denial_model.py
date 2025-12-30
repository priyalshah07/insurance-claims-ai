import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv("/Users/priyalshah/Documents/insurance-claims-ai/data/claims_features.csv")

# Exclude claims that already failed hard rules (ML-eligible claims only)
df = df[df["rule_failed"] == False]

# Feature set
features = [
    "provider_denial_rate",
    "provider_claim_volume",
    "charge_to_avg_ratio",
    "high_charge_flag",
    "failed_prior_auth",
    "days_since_submission"
]

X = df[features]
y = df["is_denied"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

# Train logistic regression
model = LogisticRegression(max_iter = 1000, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
probs = model.predict_proba(X_test)[:, 1]
LOW_THRESHOLD = np.percentile(probs, 60)
HIGH_THRESHOLD = np.percentile(probs, 85)

print(LOW_THRESHOLD, HIGH_THRESHOLD)

auc = roc_auc_score(y_test, probs)

print(f"ROC-AUC: {auc:.3f}")
print("PR-AUC:", average_precision_score(y_test, probs))

threshold = np.percentile(probs, 85)  # top 15% riskiest claims
y_pred_custom = (probs >= threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

# Save model
os.makedirs("/Users/priyalshah/Documents/insurance-claims-ai/modeling", exist_ok=True)

with open("/Users/priyalshah/Documents/insurance-claims-ai/modeling/denial_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to modeling/denial_model.pkl")

# SHAP explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Global SHAP importance
shap.plots.bar(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig("/Users/priyalshah/Documents/insurance-claims-ai/data/shap_global_importance.png")
plt.close()

# Explain top-risk claims
top_idx = np.argsort(probs)[-1]  # highest-risk claim

shap.plots.waterfall(
    shap_values[top_idx],
    show=False
)
plt.tight_layout()
plt.savefig("/Users/priyalshah/Documents/insurance-claims-ai/data/shap_claim_top.png")
plt.close()

print("Model training & SHAP explainability complete")

