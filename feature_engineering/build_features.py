import pandas as pd
from datetime import datetime

# Load validated claims
df = pd.read_csv("/Users/priyalshah/Documents/insurance-claims-ai/data/validated_claims.csv")

# Ensure types
df["submission_date"] = pd.to_datetime(df["submission_date"])
df["cpt_code"] = df["cpt_code"].astype(str)

# ---- Rule-based features ----
df["num_rule_failures"] = df["rule_failure_reasons"].apply(
    lambda x: len(eval(x)) if isinstance(x, str) else 0
)

df["failed_prior_auth"] = df["rule_failure_reasons"].astype(str).str.contains(
    "Prior authorization", regex=False
)

# ---- Provider-level features ----
provider_stats = (
    df.groupby("provider_id")["claim_status"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

provider_stats["provider_denial_rate"] = provider_stats.get("Denied", 0)

df = df.merge(
    provider_stats["provider_denial_rate"],
    on="provider_id",
    how="left"
)

provider_volume = df.groupby("provider_id").size().rename("provider_claim_volume")
df = df.merge(provider_volume, on="provider_id", how="left")

# ---- Charge features ----
avg_charge = df.groupby("cpt_code")["charge_amount"].mean().rename("avg_cpt_charge")
df = df.merge(avg_charge, on="cpt_code", how="left")

df["charge_to_avg_ratio"] = df["charge_amount"] / df["avg_cpt_charge"]
df["high_charge_flag"] = df["charge_to_avg_ratio"] > 1.5

# ---- Temporal features ----
df["days_since_submission"] = (datetime.today() - df["submission_date"]).dt.days

# ---- Target variable (for ML later) ----
df["is_denied"] = (df["claim_status"] == "Denied").astype(int)

# Save final features
df.to_csv("/Users/priyalshah/Documents/insurance-claims-ai/data/claims_features.csv", index=False)

print("Feature engineering complete")
print(df[[
    "provider_denial_rate",
    "charge_to_avg_ratio",
    "num_rule_failures",
    "is_denied"
]].describe())
